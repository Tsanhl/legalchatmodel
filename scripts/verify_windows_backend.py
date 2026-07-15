#!/usr/bin/env python3
"""Cross-platform smoke test for the local llama-server inference adapter."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legal_chat_ui"))

from llama_server_backend import LlamaServerModelHolder  # noqa: E402


class FakeLlamaServer(BaseHTTPRequestHandler):
    last_payload: dict = {}
    last_authorization = ""

    def log_message(self, _format, *_args):
        return

    def do_GET(self):
        type(self).last_authorization = self.headers.get("Authorization", "")
        if self.path == "/health":
            body = {"status": "ok"}
        elif self.path == "/v1/models":
            body = {"object": "list", "data": [{"id": "test-local-gguf"}]}
        else:
            self.send_error(404)
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        type(self).last_authorization = self.headers.get("Authorization", "")
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_payload = json.loads(self.rfile.read(length))
        if type(self).last_payload.get("stream"):
            data = (
                'data: {"choices":[{"delta":{"content":"Local "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        data = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Complete local answer"}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    try:
        LlamaServerModelHolder(
            "https://external.example/v1", "", "base", 100, 0.1, 0.9, "system", {}
        )
        raise AssertionError("remote inference URL was accepted without explicit opt-in")
    except ValueError as exc:
        assert "loopback" in str(exc)

    old_key = os.environ.get("LEGAL_LLAMA_API_KEY")
    os.environ["LEGAL_LLAMA_API_KEY"] = "transport-test-key"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        backend = LlamaServerModelHolder(
            f"http://127.0.0.1:{httpd.server_port}/v1",
            "",
            "base",
            1200,
            0.15,
            0.9,
            "Legal system prompt.",
            {"england_wales": "England & Wales"},
        )
        backend.load()
        assert backend.ready and not backend.error
        assert FakeLlamaServer.last_authorization == "Bearer transport-test-key"
        assert backend.base_model == "test-local-gguf"
        progress: list[int] = []
        answer = backend.complete(
            [{"role": "user", "content": "Question"}], max_tokens=321,
            on_progress=progress.append,
        )
        assert answer == "Local answer" and progress[-1] == 2
        assert FakeLlamaServer.last_payload["max_tokens"] == 321
        assert FakeLlamaServer.last_payload["model"] == "test-local-gguf"
        assert FakeLlamaServer.last_payload["stream"] is True
        streamed = "".join(backend.stream(
            [{"role": "user", "content": "Question"}], "england_wales"
        ))
        assert streamed == "Local answer"
        messages = FakeLlamaServer.last_payload["messages"]
        assert messages[0]["role"] == "system" and "England & Wales" in messages[0]["content"]

        # Exercise the real application wiring and public health payload, not
        # only the backend class in isolation.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            ui_port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update({
                "LEGAL_CHAT_DB": str(Path(temp_dir) / "chat.sqlite3"),
                "LEGAL_FEEDBACK_ROOT": str(Path(temp_dir) / "feedback"),
                "LEGAL_PRIVATE_UPLOAD_ROOT": str(Path(temp_dir) / "uploads"),
            })
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "legal_chat_ui" / "server.py"),
                    "--backend", "llama-server",
                    "--llama-base-url", f"http://127.0.0.1:{httpd.server_port}/v1",
                    "--host", "127.0.0.1",
                    "--port", str(ui_port),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                health = None
                for _attempt in range(100):
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{ui_port}/api/health", timeout=1
                        ) as response:
                            health = json.loads(response.read().decode("utf-8"))
                        if health.get("ready"):
                            break
                    except Exception:
                        pass
                    time.sleep(0.05)
                assert health and health["ready"]
                assert health["backend"] == "llama-server"
                assert health["model_profile"] == "base"
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        print("Windows/llama-server backend and application smoke test: PASS")
    finally:
        httpd.shutdown()
        httpd.server_close()
        if old_key is None:
            os.environ.pop("LEGAL_LLAMA_API_KEY", None)
        else:
            os.environ["LEGAL_LLAMA_API_KEY"] = old_key


if __name__ == "__main__":
    main()
