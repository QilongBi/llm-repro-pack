"""Build the fully rendered model inputs, deterministically.

Two sources:
  - archived request payloads (run-time evidence): the exact messages that were sent
  - a prompt.yaml spec plus the data file: messages rebuilt from the template

Optionally applies the model's chat template (needs transformers and the tokenizer files)
so the exact text fed to the model is recorded and hashed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from common import canonical_json, load_yaml, read_records, sha256_text

_PH = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill(template: str, values: dict) -> str:
    """Replace {name} placeholders present in values. Other braces are left untouched."""
    def rep(m: re.Match) -> str:
        tok = m.group(0)
        if tok == "{{":
            return "{"
        if tok == "}}":
            return "}"
        key = m.group(1)
        return str(values[key]) if key in values else tok
    return _PH.sub(rep, template)


def label_names(spec: dict) -> list[str]:
    out = []
    for lab in spec.get("labels") or []:
        out.append(str(lab["name"]) if isinstance(lab, dict) else str(lab))
    return out


def codebook_text(spec: dict) -> str:
    lines = []
    for lab in spec.get("labels") or []:
        if isinstance(lab, dict):
            d = lab.get("definition", "")
            lines.append(f"- {lab['name']}: {d}".rstrip(": ").rstrip())
        else:
            lines.append(f"- {lab}")
    return "\n".join(lines)


def request_record(request_id: str, item_ids: list[str], messages: list[dict], extra: dict | None = None) -> dict:
    rec = {"request_id": request_id, "item_ids": item_ids, "messages": messages,
           "sha256": sha256_text(canonical_json(messages))}
    if extra:
        rec["params"] = extra
    return rec


def render_from_spec(spec: dict, records: list[dict], id_col: str) -> list[dict]:
    base = {"label_list": ", ".join(label_names(spec)), "codebook": codebook_text(spec)}
    system = spec.get("system")
    user_t = spec.get("user_template")
    if not user_t:
        raise ValueError("prompt spec has no user_template")
    shots: list[dict] = []
    for ex in spec.get("few_shot") or []:
        inp = ex.get("input")
        vals = dict(base, **(inp if isinstance(inp, dict) else {"text": inp}))
        out = ex.get("output")
        if isinstance(out, list):
            out = "; ".join(str(x) for x in out)
        shots.append({"role": "user", "content": fill(user_t, vals)})
        shots.append({"role": "assistant", "content": str(out)})
    out_recs = []
    for r in records:
        vals = dict(base)
        vals.update({str(k): ("" if v is None else v) for k, v in r.items() if k is not None})
        msgs = []
        if system:
            msgs.append({"role": "system", "content": fill(system, base)})
        msgs.extend(shots)
        msgs.append({"role": "user", "content": fill(user_t, vals)})
        rid = str(r[id_col])
        out_recs.append(request_record(rid, [rid], msgs))
    return out_recs


def _ids_in_messages(messages: list[dict]) -> list[str]:
    for m in messages:
        if m.get("role") != "user" or not isinstance(m.get("content"), str):
            continue
        try:
            obj = json.loads(m["content"])
        except ValueError:
            continue
        for key in ("records", "items", "data"):
            if isinstance(obj, dict) and isinstance(obj.get(key), list):
                return [str(x["id"]) for x in obj[key] if isinstance(x, dict) and "id" in x]
    return []


def load_archived_requests(paths: list[tuple[Path, str]]) -> list[dict]:
    """paths: (file, package-relative name). Returns request records in file order."""
    out = []
    for p, rel in paths:
        text = p.read_text(encoding="utf-8-sig")
        objs: list[Any]
        try:
            obj = json.loads(text)
            objs = obj if isinstance(obj, list) else [obj]
        except ValueError:
            objs = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        for i, obj in enumerate(objs):
            if not isinstance(obj, dict) or "messages" not in obj:
                continue
            rid = obj.get("request_id") or obj.get("custom_id") or (rel if len(objs) == 1 else f"{rel}#{i}")
            msgs = obj["messages"]
            ids = [str(x) for x in obj.get("item_ids", [])] or _ids_in_messages(msgs)
            if not ids and obj.get("id") is not None:
                ids = [str(obj["id"])]
            extra = {k: v for k, v in obj.items() if k not in ("messages", "request_id", "custom_id", "item_ids")}
            out.append(request_record(str(rid), ids, msgs, extra))
    return out


def deterministic_sample(reqs: list[dict], n: int) -> list[dict]:
    if n >= len(reqs):
        return list(reqs)
    keyed = sorted(reqs, key=lambda r: hashlib.sha256(r["request_id"].encode()).hexdigest())
    chosen = {r["request_id"] for r in keyed[:n]}
    return [r for r in reqs if r["request_id"] in chosen]


def apply_chat_template(reqs: list[dict], tokenizer_source: str, revision: str | None,
                        template_kwargs: dict) -> str | None:
    """Add rendered_text/rendered_sha256 to each record. Returns an error string on failure."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "transformers is not installed"
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision)
        for r in reqs:
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                           **template_kwargs)
            r["rendered_text"] = text
            r["rendered_sha256"] = sha256_text(text)
    except Exception as exc:  # template errors, missing files
        return f"chat template rendering failed: {exc}"
    return None


def template_kwargs_from(manifest_like: dict, profile: dict) -> dict:
    """Map profile fields with template_kwarg to their recorded values."""
    out = {}
    for fname, spec in (profile.get("fields") or {}).items():
        kw = spec.get("template_kwarg")
        f = (manifest_like.get("model_profile") or {}).get(fname)
        if kw and isinstance(f, dict) and f.get("value") not in (None, "unknown"):
            out[kw] = f["value"]
    return out


def load_spec(path: Path) -> dict | None:
    if path.suffix.lower() not in (".yaml", ".yml"):
        return None
    spec = load_yaml(path, {})
    if isinstance(spec, dict) and "user_template" in spec:
        return spec
    return None


def render_spec_file(spec_path: Path, data_path: Path, id_col: str) -> list[dict]:
    spec = load_spec(spec_path)
    if spec is None:
        raise ValueError(f"{spec_path} is not a prompt spec")
    return render_from_spec(spec, read_records(data_path), id_col)
