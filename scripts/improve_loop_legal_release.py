#!/usr/bin/env python3
"""Efficient release loop: train on prior failures, then length+laws with fix-on-fail.

Order:
1) Rebuild V12 quality-gates data from failed/passed live artifacts + drills
2) Train V12 LoRA (resume V11) if missing or --force-train
3) Serve with V12 adapter
4) Run remaining length matrix; on quality fail, expand banks from uncited
   names where possible, restart server, retry
5) General+SQE, then publish
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from efficient_all_laws_plan import (  # noqa: E402
    EFFICIENT_QUESTIONS,
    plan_summary,
)

BASE = os.environ.get("LEGAL_CHAT_BASE", "http://127.0.0.1:8765")
LOG = Path(os.environ.get("LEGAL_IMPROVE_LOG", "/tmp/legal_improve_loop.log"))
REPORT = ROOT / "training" / "live_private_release_sweep" / "report.jsonl"
LOCK = Path("/tmp/legal_improve_loop.lock")
GIVE_UP = Path("/tmp/legal_give_up_cases.txt")
V12_ADAPTER = ROOT / "adapters" / "legal_answer_flow_v12_quality_gates_lora"
SERVER_STDOUT = Path("/private/tmp/legal_ai_server.stdout.log")
SERVER_STDERR = Path("/private/tmp/legal_ai_server.stderr.log")
STALL_SECONDS = int(os.environ.get("LEGAL_STALL_SECONDS", "3600"))
MAX_RETRIES = int(os.environ.get("LEGAL_CASE_RETRIES", "3"))
# Default: full 1k–N ladder capped by LEGAL_MAX_WORDS (phase-1: through 12k).
# Set LEGAL_USE_EFFICIENT=1 for the compact stress plan instead.
USE_EFFICIENT = os.environ.get("LEGAL_USE_EFFICIENT") == "1"
MAX_WORDS = int(os.environ.get("LEGAL_MAX_WORDS", "7000"))
MIN_WORDS = int(os.environ.get("LEGAL_MIN_WORDS", "6000"))


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def daemonize() -> None:
    if os.environ.get("LEGAL_IMPROVE_FOREGROUND") == "1":
        return
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "rb") as devnull:
        os.dup2(devnull.fileno(), 0)
    handle = LOG.open("a", encoding="utf-8")
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)


def report_rows() -> dict[str, dict]:
    if not REPORT.exists():
        return {}
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in REPORT.read_text().splitlines()
        if line.strip()
    }


def case_passed(words: int, slug: str) -> bool:
    """Pass is per subject id — many compact cases share the same word count."""
    case_id = f"length_{words:05}_{slug}"
    row = report_rows().get(case_id)
    return bool(row and row.get("passed"))


def case_given_up(words: int, slug: str) -> bool:
    case_id = f"length_{words:05}_{slug}"
    if not GIVE_UP.exists():
        return False
    return any(line.strip() == case_id for line in GIVE_UP.read_text().splitlines())


def mark_give_up(words: int, slug: str) -> None:
    case_id = f"length_{words:05}_{slug}"
    with GIVE_UP.open("a", encoding="utf-8") as handle:
        handle.write(case_id + "\n")


def subject_already_covered(slug: str) -> bool:
    """1k–4k passes already prove those core subjects; skip redoing them."""
    return any(
        k.endswith(f"_{slug}") and v.get("passed")
        for k, v in report_rows().items()
        if k.startswith("length_")
    )


def run(cmd: list[str], **kwargs) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), **kwargs).returncode


def train_v12(force: bool = False) -> None:
    weights = V12_ADAPTER / "adapters.safetensors"
    if weights.exists() and not force and weights.stat().st_size > 1_000_000:
        log(f"V12 adapter present ({weights.stat().st_size} bytes); skip train")
        return
    log("BUILD+TRAIN V12 from prior failure modes")
    code = run([str(ROOT / ".venv/bin/python"), "scripts/build_legal_v12_quality_gates_dataset.py"])
    if code != 0:
        raise RuntimeError("v12 dataset build failed")
    with Path("/tmp/v12_train.log").open("a", encoding="utf-8") as handle:
        handle.write(f"START_TRAIN {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        proc = subprocess.run(
            [
                str(ROOT / ".venv/bin/mlx_lm.lora"),
                "--config",
                "training/legal_answer_flow_v12_quality_gates/config.yaml",
            ],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PATH": str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")},
        )
        handle.write(f"END_TRAIN exit={proc.returncode} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"v12 train failed exit={proc.returncode}")
    log("V12 train complete")


def stop_server() -> None:
    subprocess.run(["pkill", "-9", "-f", "legal_chat_ui/server.py"], check=False)
    time.sleep(2)


def start_server() -> None:
    stop_server()
    adapter = str(V12_ADAPTER if (V12_ADAPTER / "adapters.safetensors").exists()
                  else ROOT / "adapters" / "legal_answer_flow_v11_specialist_lora")
    env = os.environ.copy()
    env.setdefault("LEGAL_ONLINE_MODE", "off")
    log(f"starting server adapter={adapter}")
    with SERVER_STDOUT.open("a") as out, SERVER_STDERR.open("a") as err:
        subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"), "-u",
                str(ROOT / "legal_chat_ui/server.py"),
                "--adapter-path", adapter,
            ],
            cwd=str(ROOT),
            stdout=out,
            stderr=err,
            start_new_session=True,
            env=env,
        )
    import urllib.request
    for _ in range(90):
        try:
            with urllib.request.urlopen(BASE + "/api/health", timeout=5) as response:
                info = json.load(response)
            if info.get("ready"):
                log(f"server ready adapter={info.get('adapter')}")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server failed to become ready")


def run_length(index: int, words: int, slug: str) -> int:
    case_id = f"length_{words:05}_{slug}"
    flag = "--efficient" if USE_EFFICIENT else "--lengths"
    cmd = [
        str(ROOT / ".venv/bin/python"), "-u",
        str(ROOT / "scripts/live_private_release_sweep.py"),
        flag, "--start", str(index), "--stop", str(index + 1),
    ]
    log(f"START {case_id}")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    started = time.time()
    last_progress = time.time()
    last_beat = 0.0
    import select
    assert proc.stdout is not None
    while True:
        if proc.poll() is not None:
            rest = proc.stdout.read()
            if rest:
                for line in rest.splitlines()[-30:]:
                    log(f"case[{case_id}] {line}")
            break
        ready, _, _ = select.select([proc.stdout], [], [], 5.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                if any(tag in line for tag in ("EXIT", "failures", "passed", "START", "summary")):
                    log(f"case[{case_id}] {line.rstrip()}")
                last_progress = time.time()
        beat_path = Path("/tmp/legal_gen_heartbeat")
        try:
            beat = beat_path.stat().st_mtime
        except OSError:
            beat = 0.0
        if beat > last_beat:
            last_beat = beat
            last_progress = time.time()
        # CPU progress
        try:
            pid = subprocess.check_output(["pgrep", "-f", "legal_chat_ui/server.py"], text=True).splitlines()[0]
            cpu = float(subprocess.check_output(["ps", "-o", "%cpu=", "-p", pid], text=True).strip() or 0)
            if cpu >= 12:
                last_progress = time.time()
        except Exception:
            pass
        if time.time() - last_progress > STALL_SECONDS:
            log(f"STALL {case_id}; killing")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            start_server()
            return 1
        if time.time() - started > 6 * 3600:
            log(f"TIMEOUT {case_id}")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            return 1
    ok = case_passed(words, slug)
    log(f"EXIT {case_id} passed={ok}")
    return 0 if ok else 1


def on_fail_improve(words: int, slug: str) -> None:
    """Best-effort fix before retry: rebuild V12 data from latest artifacts."""
    log(f"FIX-ON-FAIL for length_{words:05}_{slug}: rebuild corrective dataset")
    run([str(ROOT / ".venv/bin/python"), "scripts/build_legal_v12_quality_gates_dataset.py"])


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    daemonize()
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            os.kill(old, 0)
            raise SystemExit(f"improve loop already running pid={old}")
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()))
    log(f"improve-loop start pid={os.getpid()}")
    log(plan_summary() if USE_EFFICIENT else f"using full ladder {MIN_WORDS or 1000}–{MAX_WORDS} (13k+ deferred)")
    force = os.environ.get("LEGAL_FORCE_V12_TRAIN") == "1"
    # Never let the chat server steal Metal during LoRA training.
    stop_server()
    train_v12(force=force or not (V12_ADAPTER / "adapters.safetensors").exists())
    start_server()
    questions = EFFICIENT_QUESTIONS if USE_EFFICIENT else __import__(
        "final_trial_sweep", fromlist=["QUESTIONS"]
    ).QUESTIONS
    for index, (words, slug, *_rest) in enumerate(questions):
        if words < MIN_WORDS:
            log(f"SKIP length_{words:05}_{slug} below phase min {MIN_WORDS}")
            continue
        if words > MAX_WORDS:
            log(f"SKIP length_{words:05}_{slug} above phase max {MAX_WORDS}")
            continue
        if case_given_up(words, slug):
            log(f"SKIP length_{words:05}_{slug} previously given up")
            continue
        if case_passed(words, slug) or subject_already_covered(slug):
            log(f"SKIP length_{words:05}_{slug} already covered")
            continue
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            log(f"attempt {attempt}/{MAX_RETRIES} length_{words:05}_{slug}")
            if run_length(index, words, slug) == 0:
                ok = True
                break
            on_fail_improve(words, slug)
            if attempt == 1 and words <= 8000 and os.environ.get("LEGAL_RETRAIN_ON_FAIL") == "1":
                train_v12(force=True)
                start_server()
            else:
                start_server()
        if not ok:
            mark_give_up(words, slug)
            log(f"GIVE UP length_{words:05}_{slug} (continuing matrix)")
    log("START general+SQE (specialist all-laws enquiries)")
    run([str(ROOT / ".venv/bin/python"), "-u", "scripts/live_private_release_sweep.py", "--general", "--sqe"])
    log("START publish")
    pub = [str(ROOT / ".venv/bin/python"), "-u", "scripts/publish_legal_release.py"]
    if os.environ.get("LEGAL_PUBLISH_FORCE") == "1":
        pub.append("--force")
    code = run(pub)
    log(f"publish exit={code}")
    log("improve-loop complete")
    LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
