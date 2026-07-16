#!/usr/bin/env python3
"""Supervise the full remaining length matrix to completion, then all-law short suite + V12.

Designed to survive terminal disconnects when launched under setsid/nohup.
Each length case runs in a child process so a supervisor restart does not
immediately tear down an in-flight urllib SSE client mid-write.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_trial_sweep import QUESTIONS as LENGTH_QUESTIONS  # noqa: E402

BASE = os.environ.get("LEGAL_CHAT_BASE", "http://127.0.0.1:8765")
LOG = Path(os.environ.get("LEGAL_SUPERVISOR_LOG", "/tmp/legal_full_supervisor.log"))
REPORT = ROOT / "training" / "live_private_release_sweep" / "report.jsonl"
LOCK = Path("/tmp/legal_full_supervisor.lock")
STALL_SECONDS = int(os.environ.get("LEGAL_STALL_SECONDS", "300"))
MAX_RETRIES = int(os.environ.get("LEGAL_CASE_RETRIES", "3"))
CASE_TIMEOUT = int(os.environ.get("LEGAL_CASE_TIMEOUT", str(6 * 3600)))
SERVER_STDOUT = Path("/private/tmp/legal_ai_server.stdout.log")
SERVER_STDERR = Path("/private/tmp/legal_ai_server.stderr.log")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        same = LOG.exists() and os.fstat(1).st_ino == LOG.stat().st_ino
    except OSError:
        same = False
    if not same:
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def server_cpu_pct() -> float:
    """Return the legal server process CPU percentage, or -1 if unknown."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "%cpu=", "-p", str(_server_pid())],
            text=True,
        ).strip()
        return float(out or "-1")
    except Exception:
        return -1.0


def _server_pid() -> int:
    try:
        out = subprocess.check_output(["pgrep", "-f", "legal_chat_ui/server.py"], text=True)
        return int(out.splitlines()[0].strip())
    except Exception:
        return 0


def assistant_words(conv_id: str | None) -> int:
    if not conv_id:
        return 0
    try:
        data = http_json(f"/api/conversations/{conv_id}")
        msgs = data.get("messages") or []
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                return len((msg.get("content") or "").split())
    except Exception:
        return 0
    return 0


def http_json(path: str, method: str = "GET", payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def health() -> dict:
    try:
        return http_json("/api/health")
    except Exception as exc:
        return {"ready": False, "error": str(exc)}


def busy_info() -> dict:
    try:
        return http_json("/api/busy")
    except Exception as exc:
        return {"busy": False, "error": str(exc)}


def purge_private() -> None:
    try:
        convs = http_json("/api/conversations").get("conversations") or []
    except Exception:
        convs = []
    for conv in convs:
        if conv.get("mode") != "private":
            continue
        cid = conv.get("id")
        if not cid:
            continue
        try:
            http_json(f"/api/conversations/{cid}", method="DELETE")
            log(f"purged private {cid}")
        except Exception as exc:
            log(f"purge failed {cid}: {exc}")


def restart_server() -> None:
    log("restarting server")
    subprocess.run(["pkill", "-9", "-f", "legal_chat_ui/server.py"], check=False)
    time.sleep(2)
    env = os.environ.copy()
    # Long live sweeps are more reliable without CloudFront hangs; indexed
    # guides + local authority banks still ground answers. Override with
    # LEGAL_ONLINE_MODE=always if a live online check is required.
    env.setdefault("LEGAL_ONLINE_MODE", "off")
    with SERVER_STDOUT.open("a") as out, SERVER_STDERR.open("a") as err:
        subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "legal_chat_ui/server.py")],
            cwd=str(ROOT),
            stdout=out,
            stderr=err,
            start_new_session=True,
            env=env,
        )
    for _ in range(90):
        info = health()
        if info.get("ready"):
            log(f"server ready adapter={info.get('adapter')} online_mode={env.get('LEGAL_ONLINE_MODE')}")
            return
        time.sleep(2)
    raise RuntimeError("server failed to become ready")


def ensure_server() -> None:
    if not health().get("ready"):
        restart_server()


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


def drop_failed(words: int) -> None:
    prefix = f"length_{words:05}_"
    kept = [
        row for row in report_rows().values()
        if not (str(row.get("id", "")).startswith(prefix) and not row.get("passed"))
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")


def stdout_fingerprint() -> tuple[int, int]:
    if not SERVER_STDOUT.exists():
        return (0, 0)
    st = SERVER_STDOUT.stat()
    return (int(st.st_mtime), int(st.st_size))


def run_one(index: int) -> int:
    words, slug, register, stem = LENGTH_QUESTIONS[index]
    case_id = f"length_{words:05}_{slug}"
    ensure_server()
    if busy_info().get("busy"):
        log("busy before case; purge + wait")
        purge_private()
        time.sleep(3)
        if busy_info().get("busy"):
            restart_server()
    drop_failed(words)
    log(f"START {case_id}")
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "scripts/live_private_release_sweep.py"),
        "--lengths",
        "--start",
        str(index),
        "--stop",
        str(index + 1),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    started = time.time()
    last_progress = time.time()
    last_words = 0
    assert proc.stdout is not None
    # Non-blocking-ish read loop with stall detection.
    import select

    while True:
        if proc.poll() is not None:
            rest = proc.stdout.read()
            if rest:
                for line in rest.splitlines():
                    log(f"case[{case_id}] {line}")
            break
        ready, _, _ = select.select([proc.stdout], [], [], 5.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                log(f"case[{case_id}] {line.rstrip()}")
                last_progress = time.time()
        info = busy_info()
        # Do NOT treat server stdout growth as progress: /api/busy and
        # /api/health polls write access lines and would reset the stall timer
        # forever while generation is hung on a CloudFront CLOSE_WAIT socket.
        progressed = False
        cpu = server_cpu_pct()
        if info.get("busy") and cpu >= 12.0:
            progressed = True
        words_now = assistant_words(info.get("active_conversation_id"))
        if words_now > last_words:
            last_words = words_now
            progressed = True
        if progressed:
            last_progress = time.time()
        if time.time() - last_progress > STALL_SECONDS:
            log(
                f"STALL {case_id}; busy={info.get('busy')} cpu={cpu:.1f} "
                f"words={words_now}; killing case pid={proc.pid}"
            )
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            purge_private()
            restart_server()
            return 1
        if time.time() - started > CASE_TIMEOUT:
            log(f"TIMEOUT {case_id}")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            return 1
        if not health().get("ready"):
            log("server unready mid-case")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            restart_server()
            return 1
    code = proc.returncode or 0
    log(f"EXIT {case_id} code={code} passed={length_passed(words)}")
    return 0 if length_passed(words) else 1


def run_general_sqe() -> None:
    log("START general+SQE refresh")
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "scripts/live_private_release_sweep.py"),
            "--general",
            "--sqe",
        ],
        cwd=str(ROOT),
        check=False,
        start_new_session=True,
    )
    log("DONE general+SQE refresh")


def run_v12() -> None:
    log("START V12 dataset+train")
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/build_legal_v12_quality_gates_dataset.py")],
        cwd=str(ROOT),
        check=False,
    )
    with Path("/tmp/v12_train.log").open("a", encoding="utf-8") as handle:
        handle.write(f"START_TRAIN {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        proc = subprocess.run(
            ["mlx_lm.lora", "--config", "training/legal_answer_flow_v12_quality_gates/config.yaml"],
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.write(f"END_TRAIN exit={proc.returncode} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    log(f"DONE V12 train exit={proc.returncode}")


def acquire_lock() -> None:
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
        except Exception:
            old = 0
        if old and _alive(old):
            raise SystemExit(f"supervisor already running pid={old}")
    LOCK.write_text(str(os.getpid()))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def daemonize() -> None:
    """Detach from the controlling terminal (macOS-safe double-fork)."""
    if os.environ.get("LEGAL_SUPERVISOR_FOREGROUND") == "1":
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
    # Keep writing to the existing log path via reopen.
    log_handle = LOG.open("a", encoding="utf-8")
    os.dup2(log_handle.fileno(), 1)
    os.dup2(log_handle.fileno(), 2)


def run_publish() -> None:
    log("START publish GitHub + Hugging Face")
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/publish_legal_release.py")],
        cwd=str(ROOT),
        check=False,
    )
    log(f"DONE publish exit={proc.returncode}")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    daemonize()
    acquire_lock()
    log(f"supervisor start pid={os.getpid()}")
    ensure_server()
    purge_private()
    for index, (words, *_rest) in enumerate(LENGTH_QUESTIONS):
        if length_passed(words):
            log(f"SKIP length_{words:05}_ already passed")
            continue
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            log(f"attempt {attempt}/{MAX_RETRIES} for length_{words:05}_")
            if run_one(index) == 0:
                ok = True
                break
        if not ok:
            log(f"GIVE UP length_{words:05}_")
    ensure_server()
    run_general_sqe()
    run_v12()
    rows = [r for k, r in report_rows().items() if k.startswith("length_")]
    log(f"supervisor complete length_passed={sum(1 for r in rows if r.get('passed'))}/{len(rows)}")
    run_publish()
    LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink(missing_ok=True)
