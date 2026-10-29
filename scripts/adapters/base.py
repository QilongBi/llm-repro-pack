"""Adapter interface shared by all engines."""
from __future__ import annotations

import importlib.metadata as md
from dataclasses import dataclass, field
from typing import Any

UNSET = {None, "", "unknown", "not_set_provider_default", "engine_default", "default"}


@dataclass
class RunSpec:
    repo_id: str | None = None
    revision: str | None = None
    precision: str | None = None
    quantization: str | None = None
    ollama_tag: str | None = None
    ollama_digest: str | None = None
    decoding: dict = field(default_factory=dict)       # temperature, top_p, top_k, seed, max_tokens
    batch_size: int = 1
    deterministic: bool = False
    template_kwargs: dict = field(default_factory=dict)
    extra_options: dict = field(default_factory=dict)  # e.g. Ollama 'think'
    base_url: str | None = None
    model_name: str | None = None
    api_key_env: str | None = None
    hosted_parameters: dict = field(default_factory=dict)

    def dec(self, key: str) -> Any:
        v = self.decoding.get(key)
        return None if v in UNSET else v


class Adapter:
    name = "base"
    libraries: tuple[str, ...] = ()

    def __init__(self, spec: RunSpec):
        self.spec = spec

    def versions(self) -> dict:
        out = {}
        for lib in self.libraries:
            try:
                out[lib] = md.version(lib)
            except md.PackageNotFoundError:
                out[lib] = None
        return out

    def chat_template_sha256(self) -> str | None:
        """SHA-256 of the chat template the engine will use, or None if not inspectable."""
        return None

    def render(self, messages: list[dict]) -> str | None:
        """The exact prompt text after the chat template, or None if not inspectable."""
        return None

    def generate(self, requests: list[list[dict]]) -> list[dict]:
        """Return one dict per request: {text, reasoning (optional), raw (optional), meta (optional)}."""
        raise NotImplementedError
