"""Deterministic label extraction rules. No fuzzy repair: invalid output stays invalid.

Rule types (manifest field outputs.label_extraction_rule):
  {type: exact}                         whole text is the label (trim, strip quotes and final period,
                                        case-insensitive match against the label set)
  {type: first_line}                    first non-empty line, then exact
  {type: regex, pattern: "..."}         first capture group, then exact
  {type: json_field, field: label}      text is JSON; read one field
  {type: json_predictions, list_key: predictions, id_key: id, label_key: label}
                                        text is JSON with a list of {id, label}
  {type: harmony_final, then: {...}}    gpt-oss: keep the final channel, then apply 'then'
  {type: after_think, then: {...}}      drop <think>...</think>, then apply 'then'
Multi-label: add multi_label: true and separator: ";" (or the label is a JSON list).
"""
from __future__ import annotations

import json
import re
from typing import Any

INVALID = "__INVALID__"


def _match_label(s: str, label_set: list[str] | None) -> str | None:
    t = s.strip().strip("`\"' ").rstrip(".").strip()
    if not label_set:
        return t or None
    lower = {lab.lower(): lab for lab in label_set}
    return lower.get(t.lower())


def _labels(value: Any, rule: dict, label_set: list[str] | None) -> Any:
    if rule.get("multi_label"):
        if isinstance(value, list):
            parts = [str(x) for x in value]
        else:
            parts = str(value).split(rule.get("separator", ";"))
        out = []
        for p in parts:
            if not p.strip():
                continue
            m = _match_label(p, label_set)
            if m is None:
                return INVALID
            out.append(m)
        return tuple(sorted(set(out))) if out else INVALID
    if isinstance(value, list):
        return INVALID
    m = _match_label(str(value), label_set)
    return m if m is not None else INVALID


def harmony_final(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker not in text:
        return text
    tail = text.rsplit(marker, 1)[1]
    for end in ("<|return|>", "<|end|>", "<|call|>"):
        tail = tail.split(end, 1)[0]
    return tail


def after_think(text: str) -> str:
    return text.split("</think>", 1)[1] if "</think>" in text else text


def extract(text: str, rule: dict, label_set: list[str] | None, item_ids: list[str]) -> dict[str, Any]:
    """Return {item_id: label or INVALID}."""
    rtype = (rule or {}).get("type", "exact")
    if rtype == "harmony_final":
        return extract(harmony_final(text), rule.get("then", {"type": "exact"}), label_set, item_ids)
    if rtype == "after_think":
        return extract(after_think(text), rule.get("then", {"type": "exact"}), label_set, item_ids)
    if rtype == "json_predictions":
        lk, ik, vk = rule.get("list_key", "predictions"), rule.get("id_key", "id"), rule.get("label_key", "label")
        try:
            obj = json.loads(text)
            preds = obj[lk] if isinstance(obj, dict) else obj
            got = {str(p[ik]): _labels(p[vk], rule, label_set) for p in preds}
        except (ValueError, KeyError, TypeError):
            return {i: INVALID for i in item_ids}
        if not item_ids:
            return got
        return {i: got.get(i, INVALID) for i in item_ids}
    one: Any
    if rtype == "exact":
        one = _labels(text, rule, label_set)
    elif rtype == "first_line":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        one = _labels(lines[0], rule, label_set) if lines else INVALID
    elif rtype == "regex":
        m = re.search(rule["pattern"], text, re.DOTALL)
        one = _labels(m.group(1), rule, label_set) if m else INVALID
    elif rtype == "json_field":
        try:
            one = _labels(json.loads(text)[rule.get("field", "label")], rule, label_set)
        except (ValueError, KeyError, TypeError):
            one = INVALID
    else:
        raise ValueError(f"Unknown extraction rule type: {rtype}")
    return {i: one for i in item_ids}
