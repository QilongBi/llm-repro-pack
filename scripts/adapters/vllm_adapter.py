"""vLLM offline adapter (self-run). Uses LLM.chat with the model's own chat template."""
from __future__ import annotations

import hashlib
import json
import os

from adapters.base import Adapter, RunSpec


class VllmAdapter(Adapter):
    name = "vllm"
    libraries = ("vllm", "torch", "transformers")

    def __init__(self, spec: RunSpec):
        super().__init__(spec)
        if spec.deterministic:
            # Must be set before vLLM is imported.
            os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        kw = {"model": spec.repo_id, "revision": spec.revision, "tokenizer_revision": spec.revision,
              "seed": int(spec.dec("seed") or 0)}
        if spec.precision and str(spec.precision).lower() not in ("auto", "unknown"):
            kw["dtype"] = str(spec.precision)
        if spec.batch_size:
            kw["max_num_seqs"] = int(spec.batch_size)
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()

    def chat_template_sha256(self) -> str | None:
        ct = getattr(self.tok, "chat_template", None)
        if isinstance(ct, dict):
            ct = json.dumps(ct, sort_keys=True)
        return hashlib.sha256(ct.encode()).hexdigest() if isinstance(ct, str) else None

    def render(self, messages: list[dict]) -> str | None:
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                **self.spec.template_kwargs)
        except Exception:
            return None

    def generate(self, requests: list[list[dict]]) -> list[dict]:
        s = self.spec
        sp = {"temperature": float(s.dec("temperature") or 0.0), "max_tokens": int(s.dec("max_tokens") or 256)}
        if s.dec("top_p") is not None:
            sp["top_p"] = float(s.dec("top_p"))
        if s.dec("top_k") is not None:
            sp["top_k"] = int(s.dec("top_k"))
        if s.dec("seed") is not None:
            sp["seed"] = int(s.dec("seed"))
        params = self.SamplingParams(**sp)
        kwargs = {"chat_template_kwargs": s.template_kwargs} if s.template_kwargs else {}
        outs = self.llm.chat(requests, params, use_tqdm=False, **kwargs)
        return [{"text": o.outputs[0].text, "meta": {"finish_reason": o.outputs[0].finish_reason}} for o in outs]


ADAPTER = VllmAdapter
