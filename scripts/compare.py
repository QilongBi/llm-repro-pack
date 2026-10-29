"""Agreement and validation metrics, implemented without third-party packages.

Usage:
  python compare.py --a archived.csv --b reproduced.jsonl [--a-id id --a-label label]
                    [--b-id id --b-label label] [--multi-label --sep ";"] [--json out.json]
  Treat --a as the reference (gold or archived labels) for accuracy and macro-F1.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from common import load_labels


def cohens_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(units: Sequence[Sequence[Any]]) -> float | None:
    """units: one list of coder values per item; None means missing."""
    o: Counter = Counter()
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, x in enumerate(vals):
            for j, y in enumerate(vals):
                if i != j:
                    o[(x, y)] += 1.0 / (m - 1)
    n_c: Counter = Counter()
    for (x, _), w in o.items():
        n_c[x] += w
    n = sum(n_c.values())
    if n <= 1:
        return None
    d_o = sum(w for (x, y), w in o.items() if x != y)
    d_e = sum(n_c[x] * n_c[y] for x in n_c for y in n_c if x != y) / (n - 1)
    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1.0 - d_o / d_e


def accuracy(ref: Sequence[Any], pred: Sequence[Any]) -> float | None:
    return sum(x == y for x, y in zip(ref, pred)) / len(ref) if ref else None


def macro_f1(ref: Sequence[Any], pred: Sequence[Any]) -> float | None:
    labels = sorted(set(ref) | set(pred), key=str)
    if not labels:
        return None
    f1s = []
    for c in labels:
        tp = sum(1 for x, y in zip(ref, pred) if x == c and y == c)
        fp = sum(1 for x, y in zip(ref, pred) if x != c and y == c)
        fn = sum(1 for x, y in zip(ref, pred) if x == c and y != c)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def per_class_report(ref: Sequence[Any], pred: Sequence[Any]) -> dict:
    """Precision, recall, F1 and support per class, plus weighted F1 and a confusion matrix."""
    labels = sorted(set(ref) | set(pred), key=str)
    per, weighted, n = {}, 0.0, len(ref)
    for c in labels:
        tp = sum(1 for x, y in zip(ref, pred) if x == c and y == c)
        fp = sum(1 for x, y in zip(ref, pred) if x != c and y == c)
        fn = sum(1 for x, y in zip(ref, pred) if x == c and y != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        support = sum(1 for x in ref if x == c)
        per[str(c)] = {"precision": p, "recall": r, "f1": f, "support": support}
        weighted += f * support
    confusion = {str(a): {str(b): sum(1 for x, y in zip(ref, pred) if x == a and y == b) for b in labels}
                 for a in labels}
    return {"per_class": per, "weighted_f1": weighted / n if n else None, "confusion_matrix": confusion}


def _binary(labels: Sequence[tuple], c: str) -> list[int]:
    return [1 if c in s else 0 for s in labels]


def compare_labels(a: dict[str, Any], b: dict[str, Any], multi_label: bool = False) -> dict:
    ids = sorted(set(a) & set(b))
    ra, rb = [a[i] for i in ids], [b[i] for i in ids]
    res: dict[str, Any] = {"n_a": len(a), "n_b": len(b), "n_compared": len(ids),
                           "missing_in_b": sorted(set(a) - set(b)), "missing_in_a": sorted(set(b) - set(a))}
    if not ids:
        return res
    res["exact_match_rate"] = accuracy(ra, rb)
    if not multi_label:
        res["cohens_kappa"] = cohens_kappa(ra, rb)
        res["krippendorff_alpha"] = krippendorff_alpha_nominal([[x, y] for x, y in zip(ra, rb)])
        res["accuracy_vs_a"] = res["exact_match_rate"]
        res["macro_f1_vs_a"] = macro_f1(ra, rb)
        res["disagreements"] = [{"id": i, "a": x, "b": y} for i, x, y in zip(ids, ra, rb) if x != y][:50]
        return res
    sa = [tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in ra]
    sb = [tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in rb]
    labels = sorted({c for s in sa + sb for c in s})
    jac = [len(set(x) & set(y)) / len(set(x) | set(y)) if set(x) | set(y) else 1.0 for x, y in zip(sa, sb)]
    res["mean_jaccard"] = sum(jac) / len(jac)
    per = {}
    for c in labels:
        xa, xb = _binary(sa, c), _binary(sb, c)
        per[c] = {"cohens_kappa": cohens_kappa(xa, xb),
                  "krippendorff_alpha": krippendorff_alpha_nominal([[x, y] for x, y in zip(xa, xb)]),
                  "f1_vs_a": macro_f1(xa, xb)}
    res["per_label"] = per
    ks = [v["cohens_kappa"] for v in per.values() if v["cohens_kappa"] is not None]
    al = [v["krippendorff_alpha"] for v in per.values() if v["krippendorff_alpha"] is not None]
    res["cohens_kappa"] = sum(ks) / len(ks) if ks else None
    res["krippendorff_alpha"] = sum(al) / len(al) if al else None
    f1 = []
    for c in labels:
        xa, xb = _binary(sa, c), _binary(sb, c)
        tp = sum(1 for x, y in zip(xa, xb) if x and y)
        fp = sum(1 for x, y in zip(xa, xb) if not x and y)
        fn = sum(1 for x, y in zip(xa, xb) if x and not y)
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 1.0)
    res["macro_f1_vs_a"] = sum(f1) / len(f1) if f1 else None
    res["accuracy_vs_a"] = res["exact_match_rate"]
    return res


def _jsonable(o: Any) -> Any:
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-id", default="id")
    ap.add_argument("--a-label", default="label")
    ap.add_argument("--b-id", default="id")
    ap.add_argument("--b-label", default="label")
    ap.add_argument("--multi-label", action="store_true")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    a = load_labels(args.a, args.a_id, args.a_label, args.multi_label, args.sep)
    b = load_labels(args.b, args.b_id, args.b_label, args.multi_label, args.sep)
    res = _jsonable(compare_labels(a, b, args.multi_label))
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
