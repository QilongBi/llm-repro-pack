"""Ollama adapter (self-run, local server). Checks the model digest before running."""
from __future__ import annotations

import json
import urllib.request

from adapters.base import Adapter, RunSpec


class OllamaAdapter(Adapter):
    name = "ollama"
    libraries = ()

    def __init__(self, spec: RunSpec):
        super().__init__(spec)
        self.base = (spec.base_url or "http://127.0.0.1:11434").rstrip("/")
        if not spec.ollama_tag:
            raise SystemExit("ERROR: model.ollama_tag is unknown; pass --ollama-tag")
        tags = self._get("/api/tags")
        found = [m for m in tags.get("models", []) if spec.ollama_tag in (m.get("name"), m.get("model"))]
        if not found:
            raise SystemExit(f"ERROR: Ollama model {spec.ollama_tag} is not pulled. Run: ollama pull {spec.ollama_tag}")
        self.digest = found[0].get("digest")
        if spec.ollama_digest and self.digest and not str(self.digest).startswith(str(spec.ollama_digest)) \
                and not str(spec.ollama_digest).startswith(str(self.digest)):
            raise SystemExit(f"ERROR: Ollama digest mismatch: local {self.digest}, manifest {spec.ollama_digest}")

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def versions(self) -> dict:
        try:
            return {"ollama": self._get("/api/version").get("version")}
        except Exception:
            return {"ollama": None}

    def generate(self, requests: list[list[dict]]) -> list[dict]:
        s = self.spec
        opts = {}
        for k, ok in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                      ("seed", "seed"), ("max_tokens", "num_predict")):
            if s.dec(k) is not None:
                opts[ok] = s.dec(k)
        out = []
        for msgs in requests:
            body = {"model": s.ollama_tag, "messages": msgs, "stream": False, "options": opts}
            body.update(s.extra_options)
            req = urllib.request.Request(self.base + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            msg = resp.get("message") or {}
            out.append({"text": msg.get("content", ""), "reasoning": msg.get("thinking"), "raw": resp,
                        "meta": {"digest": self.digest}})
        return out


ADAPTER = OllamaAdapter
