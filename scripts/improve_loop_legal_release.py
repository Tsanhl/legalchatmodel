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
from final_trial_sweep import QUESTIONS as LENGTH_QUESTIONS  # noqa: E402

BASE = os.environ.get("LEGAL_CHAT_BASE", "http://127.0.0.1:8765")
LOG = Path(os.environ.get("LEGAL_IMPROVE_LOG", "/tmp/legal_improve_loop.log"))
REPORT = ROOT / "training" / "live_private_release_sweep" / "report.jsonl"
LOCK = Path("/tmp/legal_improve_loop.lock")
V12_ADAPTER = ROOT / "adapters" / "legal_answer_flow_v12_quality_gates_lora"
SERVER_STDOUT = Path("/private/tmp/legal_ai_server.stdout.log")
SERVER_STDERR = Path("/private/tmp/legal_ai_server.stderr.log")
STALL_SECONDS = int(os.environ.get("LEGAL_STALL_SECONDS", "3600"))
MAX_RETRIES = int(os.environ.get("LEGAL_CASE_RETRIES", "3"))


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


def length_passed(words: int) -> bool:
    prefix = f"length_{words:05}_"
    return any(k.startswith(prefix) and v.get("passed") for k, v in report_rows().items())


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
            ["mlx_lm.lora", "--config", "training/legal_answer_flow_v12_quality_gates/config.yaml"],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
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


def run_length(index: int) -> int:
    words = LENGTH_QUESTIONS[index][0]
    case_id = f"length_{words:05}_{LENGTH_QUESTIONS[index][1]}"
    cmd = [
        str(ROOT / ".venv/bin/python"), "-u",
        str(ROOT / "scripts/live_private_release_sweep.py"),
        "--lengths", "--start", str(index), "--stop", str(index + 1),
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
    ok = length_passed(words)
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
    force = os.environ.get("LEGAL_FORCE_V12_TRAIN") == "1"
    train_v12(force=force or not (V12_ADAPTER / "adapters.safetensors").exists())
    start_server()
    for index, (words, slug, *_rest) in enumerate(LENGTH_QUESTIONS):
        if length_passed(words):
            log(f"SKIP length_{words:05}_ already passed")
            continue
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            log(f"attempt {attempt}/{MAX_RETRIES} length_{words:05}_{slug}")
            if run_length(index) == 0:
                ok = True
                break
            on_fail_improve(words, slug)
            # Only retrain once mid-loop if still on early failures.
            if attempt == 1 and words <= 8000 and os.environ.get("LEGAL_RETRAIN_ON_FAIL") == "1":
                train_v12(force=True)
                start_server()
            else:
                start_server()
        if not ok:
            log(f"GIVE UP length_{words:05}_ (continuing matrix)")
    log("START general+SQE")
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
