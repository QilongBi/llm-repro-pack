"""Validation against a human-labeled set, following Törnberg (2024), section 4.

Computes, deterministically:
  - accuracy, macro-F1, weighted F1, per-class precision/recall/F1/support, confusion matrix
  - Cohen's kappa and Krippendorff's alpha (LLM vs reference)
  - human inter-coder reliability when several coder columns are given, and LLM vs each coder
  - per-subset metrics (e.g. language, period) when a subset column is given
  - class support check (at least 20 items per class)
  - leakage checks: validation items that also appear as few-shot examples or in the
    prompt-development set
  - a file of misclassified items for failure analysis
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from common import read_records, to_label
from compare import (compare_labels, krippendorff_alpha_nominal, per_class_report)

MIN_CLASS_SUPPORT = 20


def _norm_text(s: Any) -> str:
    return " ".join(str(s).lower().split())


def run_validation(gold_path: Path, gold_id: str, gold_label: str, preds: dict[str, Any], multi: bool,
                   sep: str | None, coder_cols: list[str] | None = None, subset_col: str | None = None,
                   data_rows: list[dict] | None = None, data_id: str | None = None, data_text: str | None = None,
                   few_shot_texts: list[str] | None = None, dev_ids: set[str] | None = None) -> dict:
    rows = read_records(gold_path)
    gold = {str(r[gold_id]): to_label(r[gold_label], multi, sep) for r in rows if r.get(gold_label) not in (None, "")}
    res: dict[str, Any] = {"n_gold": len(gold)}
    base = compare_labels(gold, preds, multi)
    res.update({"n_compared": base["n_compared"], "accuracy": base.get("accuracy_vs_a"),
                "macro_f1": base.get("macro_f1_vs_a"), "cohens_kappa": base.get("cohens_kappa"),
                "krippendorff_alpha": base.get("krippendorff_alpha"),
                "missing_predictions": base.get("missing_in_b", [])})
    ids = sorted(set(gold) & set(preds))
    if not multi and ids:
        rep = per_class_report([gold[i] for i in ids], [preds[i] for i in ids])
        res.update(rep)
        support = {c: v["support"] for c, v in rep["per_class"].items() if v["support"] or c in set(gold.values())}
    else:
        support = {}
        for i in ids:
            for c in gold[i] if isinstance(gold[i], tuple) else (gold[i],):
                support[c] = support.get(c, 0) + 1
        res["per_label"] = base.get("per_label")
    res["per_class_support"] = support
    res["min_class_support"] = min(support.values()) if support else 0
    res["small_classes"] = sorted(c for c, n in support.items() if n < MIN_CLASS_SUPPORT)

    # Human inter-coder reliability and LLM vs each coder.
    if coder_cols:
        coders = {c: {str(r[gold_id]): to_label(r.get(c), multi, sep) for r in rows if r.get(c) not in (None, "")}
                  for c in coder_cols}
        all_ids = sorted(set().union(*[set(v) for v in coders.values()]))
        res["human_intercoder_alpha"] = krippendorff_alpha_nominal(
            [[coders[c].get(i) for c in coder_cols] for i in all_ids]) if len(coder_cols) > 1 else None
        res["llm_vs_coders"] = {c: compare_labels(coders[c], preds, multi).get("krippendorff_alpha")
                                for c in coder_cols}
        res["llm_and_humans_alpha"] = krippendorff_alpha_nominal(
            [[coders[c].get(i) for c in coder_cols] + [preds.get(i)] for i in all_ids])

    # Subsets.
    if subset_col:
        sub_of: dict[str, str] = {}
        for r in rows:
            if subset_col in r:
                sub_of[str(r[gold_id])] = str(r[subset_col])
        for r in data_rows or []:
            if subset_col in r and data_id:
                sub_of.setdefault(str(r[data_id]), str(r[subset_col]))
        subs: dict[str, dict] = {}
        for s in sorted(set(sub_of.values())):
            g = {i: gold[i] for i in gold if sub_of.get(i) == s}
            p = {i: preds[i] for i in preds if sub_of.get(i) == s}
            c = compare_labels(g, p, multi)
            subs[s] = {"n": c["n_compared"], "accuracy": c.get("accuracy_vs_a"),
                       "macro_f1": c.get("macro_f1_vs_a"), "cohens_kappa": c.get("cohens_kappa")}
        res["subsets"] = {"column": subset_col, "results": subs}

    # Leakage checks.
    texts = {}
    if data_rows and data_id and data_text:
        texts = {str(r[data_id]): r.get(data_text) for r in data_rows}
    leak: dict[str, Any] = {"checked": []}
    if few_shot_texts and texts:
        shots = {_norm_text(t) for t in few_shot_texts}
        hit = sorted(i for i in gold if i in texts and _norm_text(texts[i]) in shots)
        leak["checked"].append("few_shot_examples")
        leak["few_shot_in_validation"] = hit
    if dev_ids is not None:
        leak["checked"].append("prompt_development_set")
        leak["dev_set_in_validation"] = sorted(set(gold) & dev_ids)
    res["leakage"] = leak

    res["errors"] = [{"id": i, "gold": _s(gold[i]), "llm": _s(preds[i]), "text": texts.get(i, "")}
                     for i in ids if gold[i] != preds[i]]
    return res


def _s(v: Any) -> str:
    return "; ".join(v) if isinstance(v, tuple) else str(v)


def write_errors(path: Path, errors: list[dict], include_text: bool, include_gold: bool = True) -> None:
    """Text is written only when the data may be shared; gold labels only when the gold set may be shared."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id"] + (["gold"] if include_gold else []) + ["llm"] + (["text"] if include_text else [])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for e in errors:
            w.writerow(e)
