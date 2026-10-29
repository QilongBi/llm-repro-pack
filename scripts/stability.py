"""Stability analysis after Törnberg (2024), section 8.

Re-runs a deterministic sample with the archived scripts:
  --repeats K          the same prompt K times (test-retest; informative when temperature > 0
                       or when the engine is not batch-invariant)
  --paraphrase F ...   prompt.yaml variants written by the researcher (prompt stability)
and reports Krippendorff's alpha across runs and agreement with the archived labels.

Usage:
  python stability.py --package PKG --engine ENGINE [--sample 50] [--repeats 3]
                      [--paraphrase variant1.yaml variant2.yaml] [classify options]
Exit codes as classify.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import field_value, load_labels, load_yaml, now_iso
from compare import _jsonable, compare_labels, krippendorff_alpha_nominal

PASS = ("revision", "repo_id", "base_url", "model_name", "api_key_env", "ollama_tag")


def _run(pkg: Path, args, out: Path, spec: Path | None) -> int:
    cmd = [sys.executable, str(pkg / "scripts" / "classify.py"), "--package", str(pkg), "--engine", args.engine,
           "--out", str(out), "--sample", str(args.sample)]
    if spec:
        cmd += ["--prompt-spec", str(spec)]
    for opt in PASS:
        if getattr(args, opt):
            cmd += ["--" + opt.replace("_", "-"), getattr(args, opt)]
    return subprocess.run(cmd).returncode


def _alpha(runs: list[dict], ids: list[str]) -> float | None:
    return krippendorff_alpha_nominal([[r.get(i) for r in runs] for i in ids]) if len(runs) > 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--paraphrase", type=Path, nargs="*", default=[])
    for opt in PASS:
        ap.add_argument("--" + opt.replace("_", "-"), dest=opt)
    args = ap.parse_args()
    if not args.repeats and not args.paraphrase:
        ap.error("give --repeats and/or --paraphrase")
    pkg = args.package.resolve()
    m = load_yaml(pkg / "manifest.yaml")
    base = pkg / "reproduction" / "stability"
    multi = field_value(m, "study.task_type") == "multi_label"
    sep = field_value(m, "study.multi_label_separator")
    archived = load_labels(pkg / field_value(m, "outputs.labels"), field_value(m, "outputs.id_column"),
                           field_value(m, "outputs.label_column"), multi, sep)
    groups: dict[str, dict[str, dict]] = {"repeats": {}, "paraphrases": {}}
    for k in range(1, args.repeats + 1):
        out = base / f"repeat_{k}"
        rc = _run(pkg, args, out, None)
        if rc:
            return rc
        groups["repeats"][f"repeat_{k}"] = load_labels(out / "labels.jsonl", "id", "label", multi, sep)
    for i, spec in enumerate(args.paraphrase, 1):
        out = base / f"paraphrase_{i}"
        rc = _run(pkg, args, out, spec.resolve())
        if rc:
            return rc
        groups["paraphrases"][f"paraphrase_{i} ({spec.name})"] = load_labels(out / "labels.jsonl", "id", "label",
                                                                           multi, sep)
    result: dict = {"created_at": now_iso(), "engine": args.engine, "sample": args.sample}
    lines = ["# Stability report", "", f"Engine: {args.engine}. Sample: {args.sample} items "
             "(deterministic, same items in every run).", ""]
    for name, runs in groups.items():
        if not runs:
            continue
        ids = sorted(set().union(*[set(r) for r in runs.values()]))
        sub = {i: archived[i] for i in ids if i in archived}
        alpha = _alpha(list(runs.values()), ids)
        alpha_with = _alpha([sub] + list(runs.values()), ids)
        vs = {k: compare_labels(sub, r, multi).get("exact_match_rate") for k, r in runs.items()}
        result[name] = {"krippendorff_alpha_between_runs": alpha, "krippendorff_alpha_with_archived": alpha_with,
                        "exact_match_vs_archived": vs}
        lines += [f"## {name.capitalize()}", "", "| Measure | Value |", "|---|---|",
                  f"| Krippendorff's alpha between runs | {alpha if alpha is not None else 'n/a (one run)'} |",
                  f"| Krippendorff's alpha incl. archived labels | {alpha_with} |", ""]
        lines += ["| Run | Exact match vs archived |", "|---|---|"] + [f"| {k} | {v} |" for k, v in vs.items()] + [""]
    lines += ["Large changes under small prompt variations indicate an unstable annotation; report them.",
              "Paraphrase files are part of the analysis: archive them with the paper."]
    base.mkdir(parents=True, exist_ok=True)
    (base / "stability.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    (base / "STABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Report: {base / 'STABILITY_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
