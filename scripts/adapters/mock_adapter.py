"""Mock adapter for tests and dry runs. Never use it for a real reproduction.

Responses come from the JSON file named by LLMRP_MOCK_RESPONSES, mapping the SHA-256 of
the canonical messages to the response text. Unknown requests get LLMRP_MOCK_DEFAULT
(default: empty string, which extracts as invalid).
"""
from __future__ import annotations

import hashlib
import json
import os

from adapters.base import Adapter, RunSpec


def messages_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class MockAdapter(Adapter):
    name = "mock"

    def __init__(self, spec: RunSpec):
        super().__init__(spec)
        path = os.environ.get("LLMRP_MOCK_RESPONSES")
        self.table = json.load(open(path, encoding="utf-8")) if path else {}
        self.default = os.environ.get("LLMRP_MOCK_DEFAULT", "")

    def versions(self) -> dict:
        return {"mock": "0"}

    def chat_template_sha256(self) -> str | None:
        return os.environ.get("LLMRP_MOCK_TEMPLATE_SHA")

    def generate(self, requests: list[list[dict]]) -> list[dict]:
        return [{"text": self.table.get(messages_sha(m), self.default)} for m in requests]


ADAPTER = MockAdapter
