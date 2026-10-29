"""OpenAI-compatible hosted API adapter (Groq, Together, OpenRouter, vLLM server, ...).

The API key is read from the environment variable named by --api-key-env or
hosted_api.api_key_env. The key is never written anywhere; raw responses are stored
without request headers.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from adapters.base import Adapter, RunSpec

SKIP_PARAMS = {"messages", "model", "stream", "n"}


class HostedApiAdapter(Adapter):
    name = "hosted_api"
    libraries = ()

    def __init__(self, spec: RunSpec, max_retries: int = 2):
        super().__init__(spec)
        if not spec.base_url or not spec.model_name:
            raise SystemExit("ERROR: hosted_api needs base_url and model_name (manifest or --base-url/--model-name)")
        base = spec.base_url.rstrip("/")
        if not base.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise SystemExit("ERROR: use HTTPS for remote endpoints")
        self.base = base
        self.key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        if spec.api_key_env and not self.key:
            raise SystemExit(f"ERROR: environment variable {spec.api_key_env} is not set")
        self.max_retries = max_retries

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise

    def generate(self, requests: list[list[dict]]) -> list[dict]:
        s = self.spec
        base_params = {k: v for k, v in (s.hosted_parameters or {}).items() if k not in SKIP_PARAMS}
        for k in ("temperature", "top_p", "seed", "max_tokens"):
            if s.dec(k) is not None:
                base_params[k] = s.dec(k)
        out = []
        for msgs in requests:
            payload = {"model": s.model_name, "messages": msgs, "stream": False, **base_params}
            resp = self._post(payload)
            ch = (resp.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            out.append({"text": msg.get("content") or "",
                        "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                        "raw": resp,
                        "meta": {"system_fingerprint": resp.get("system_fingerprint"),
                                 "finish_reason": ch.get("finish_reason"), "created": resp.get("created"),
                                 "model": resp.get("model")}})
        return out


ADAPTER = HostedApiAdapter
