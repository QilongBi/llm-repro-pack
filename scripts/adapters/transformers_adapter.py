"""Hugging Face transformers adapter (self-run)."""
from __future__ import annotations

import hashlib
import json
import os

from adapters.base import Adapter, RunSpec

DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16",
          "half": "float16", "float32": "float32", "fp32": "float32"}


class TransformersAdapter(Adapter):
    name = "transformers"
    libraries = ("transformers", "torch", "tokenizers", "accelerate")

    def __init__(self, spec: RunSpec):
        super().__init__(spec)
        if spec.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if spec.deterministic:
            torch.use_deterministic_algorithms(True)
        self.tok = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = DTYPES.get(str(spec.precision).lower())
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id, revision=spec.revision, device_map="auto",
            torch_dtype=getattr(torch, dtype) if dtype else "auto")
        self.model.eval()

    def chat_template_sha256(self) -> str | None:
        ct = self.tok.chat_template
        if isinstance(ct, dict):
            ct = json.dumps(ct, sort_keys=True)
        return hashlib.sha256(ct.encode()).hexdigest() if isinstance(ct, str) else None

    def render(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            **self.spec.template_kwargs)

    def generate(self, requests: list[list[dict]]) -> list[dict]:
        s = self.spec
        temp = s.dec("temperature") or 0.0
        kw = {"max_new_tokens": int(s.dec("max_tokens") or 256), "do_sample": float(temp) > 0}
        if kw["do_sample"]:
            kw["temperature"] = float(temp)
            if s.dec("top_p") is not None:
                kw["top_p"] = float(s.dec("top_p"))
            if s.dec("top_k") is not None and int(s.dec("top_k")) > 0:
                kw["top_k"] = int(s.dec("top_k"))
        out = []
        bs = max(1, int(s.batch_size or 1))
        for i in range(0, len(requests), bs):
            batch = requests[i:i + bs]
            if s.dec("seed") is not None:
                from transformers import set_seed
                set_seed(int(s.dec("seed")))
            texts = [self.render(m) for m in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, pad_token_id=self.tok.pad_token_id, **kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row in new:
                raw = self.tok.decode(row, skip_special_tokens=False)
                out.append({"text": self.tok.decode(row, skip_special_tokens=True), "raw_text": raw})
        return out


ADAPTER = TransformersAdapter
