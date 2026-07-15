"""Cross-platform local inference through llama.cpp's OpenAI-compatible server.

This module intentionally uses only the Python standard library.  It lets the
legal application keep its RAG, privacy, supervision and storage layers on
Windows/Linux while a local ``llama-server`` process performs generation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class LlamaServerModelHolder:
    """Model-holder interface compatible with ``server.ModelHolder``."""

    backend = "llama-server"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        model_profile: str,
        max_tokens: int,
        temp: float,
        top_p: float,
        system_prompt: str,
        jurisdiction_labels: dict[str, str],
    ):
        base_url = base_url.strip().rstrip("/")
        if not re.match(r"^https?://", base_url, re.I):
            raise ValueError("llama-server URL must begin with http:// or https://")
        hostname = (urlparse(base_url).hostname or "").lower()
        allow_remote = os.environ.get("LEGAL_ALLOW_REMOTE_MODEL", "").lower() in {
            "1", "true", "yes", "on"
        }
        if hostname not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
            raise ValueError(
                "llama-server must use a loopback URL; set LEGAL_ALLOW_REMOTE_MODEL=1 "
                "only for a deliberately configured private inference host"
            )
        self.base_url = base_url if base_url.endswith("/v1") else base_url + "/v1"
        self.server_root = self.base_url[:-3].rstrip("/")
        self.remote_model = model_name.strip()
        self.base_model = self.remote_model or "local GGUF"
        self.model_profile = model_profile
        # The health/UI adapter indicator is truthy only for an explicitly
        # operator-labelled fused V11 model. It is not inferred from a filename.
        self.adapter_path = "v11-fused-gguf" if model_profile == "v11-fused" else None
        self.max_tokens = max_tokens
        self.temp = temp
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.jurisdiction_labels = jurisdiction_labels
        self.ready = False
        self.error: str | None = None
        self._gen_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self.active_conversation_id: str | None = None
        self.timeout = int(os.environ.get("LEGAL_LLAMA_TIMEOUT_SECONDS", "1800"))
        self.api_key = os.environ.get("LEGAL_LLAMA_API_KEY", "").strip()

    def _request(self, url: str, payload: dict | None = None, timeout: int | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="GET" if payload is None else "POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach llama-server at {self.base_url}: {exc.reason}") from exc

    def _request_json(self, url: str, payload: dict | None = None, timeout: int | None = None):
        with self._request(url, payload, timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def load(self):
        """Confirm the local inference process is ready and discover its model id."""
        try:
            try:
                health = self._request_json(self.server_root + "/health", timeout=10)
                if isinstance(health, dict) and health.get("status") not in (None, "ok"):
                    raise RuntimeError(f"llama-server is not ready: {health.get('status')}")
            except RuntimeError:
                # Some compatible local servers expose only the OpenAI routes.
                pass
            models = self._request_json(self.base_url + "/models", timeout=15)
            available = models.get("data", []) if isinstance(models, dict) else []
            if not self.remote_model and available and isinstance(available[0], dict):
                self.remote_model = str(available[0].get("id", "")).strip()
                self.base_model = self.remote_model or self.base_model
            self.ready = True
            self.error = None
            print(
                f"[model] llama-server ready at {self.base_url}; "
                f"profile={self.model_profile} model={self.base_model}",
                flush=True,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[model] FAILED: {self.error}", flush=True)

    def _payload(self, messages: list[dict], max_tokens: int, stream: bool) -> dict:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temp,
            "top_p": self.top_p,
            "stream": stream,
        }
        if self.remote_model:
            payload["model"] = self.remote_model
        return payload

    @staticmethod
    def _content(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("llama-server returned no assistant message") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            )
        return str(content or "")

    def complete(self, messages: list[dict], max_tokens: int | None = None, on_progress=None) -> str:
        """Return one completion while forwarding count-only keepalive progress."""
        generated: list[str] = []
        with self._gen_lock:
            generated.extend(self._iter_stream_unlocked(
                messages, max_tokens or self.max_tokens, on_progress
            ))
        content = "".join(generated).strip()
        if on_progress:
            on_progress(len(content.split()))
        return content

    def _iter_stream_unlocked(self, messages: list[dict], max_tokens: int, on_progress=None):
        generated: list[str] = []
        last_progress = time.monotonic()
        with self._request(
            self.base_url + "/chat/completions",
            self._payload(messages, max_tokens, True),
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                event = json.loads(value)
                choices = event.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                text = delta.get("content", "") if isinstance(delta, dict) else ""
                if text:
                    text = str(text)
                    generated.append(text)
                    yield text
                if on_progress and time.monotonic() - last_progress >= 1.5:
                    on_progress(len("".join(generated).split()))
                    last_progress = time.monotonic()

    def stream_messages(self, messages: list[dict], max_tokens: int | None = None, on_progress=None):
        """Yield OpenAI-compatible server-sent-event text deltas."""
        with self._gen_lock:
            yield from self._iter_stream_unlocked(
                messages, max_tokens or self.max_tokens, on_progress
            )

    def stream(self, history: list[dict], jurisdiction: str | None):
        system = self.system_prompt
        if jurisdiction and jurisdiction in self.jurisdiction_labels:
            system += (
                "\nThe user's selected jurisdiction is "
                + self.jurisdiction_labels[jurisdiction]
                + "."
            )
        messages = [{"role": "system", "content": system}] + history
        yield from self.stream_messages(messages, self.max_tokens)
