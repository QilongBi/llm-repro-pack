"""Compute the reproducibility grade (A-D) from manifest.yaml and the schema.

Usage: python grade.py <package_dir>   (prints the grade, updates manifest.yaml)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from common import (field_value, get_path, is_field, is_known, is_locked, load_yaml, norm_value,
                    resource_path, save_yaml)
from model_profiles import get_profile

GRADE_MEANING = {
    "A": "bit-identical reproduction expected on the same hardware class; fully open model",
    "B": "bit-identical reproduction expected on the same hardware class; open-weight model",
    "C": "statistical reproduction only (agreement metrics)",
    "D": "downstream analysis of archived outputs only",
}


def condition(m: dict, cond: dict | None) -> bool:
    if not cond:
        return True
    v = field_value(m, cond["field"])
    if "known" in cond:
        return (v not in (None, "", "unknown")) == bool(cond["known"])
    if v in (None, "unknown"):
        return False
    if "equals" in cond:
        return norm_value(v) == norm_value(cond["equals"])
    if "in" in cond:
        return norm_value(v) in {norm_value(x) for x in cond["in"]}
    if "gt" in cond:
        try:
            return float(v) > float(cond["gt"])
        except (TypeError, ValueError):
            return False
    return True


def applicable(m: dict, entry: dict) -> bool:
    return condition(m, entry.get("when"))


def critical_in(m: dict, entry: dict, mode: str) -> bool:
    if mode not in (entry.get("critical") or []):
        return False
    if not applicable(m, entry):
        return False
    if entry.get("unless") and condition(m, entry["unless"]):
        return False
    return True


def profile_entries(m: dict) -> list[dict]:
    prof = get_profile(field_value(m, "model_profile.name") or "default")
    out = []
    for fname, spec in (prof.get("fields") or {}).items():
        out.append({"path": f"model_profile.{fname}",
                    "critical": ["self_run", "hosted_api"] if spec.get("critical") else [],
                    "fix": f"Record {fname}: {spec.get('description', '')}".strip()})
    return out


def _reason(f: Any) -> str:
    if not is_field(f):
        return "missing"
    if f.get("conflict"):
        return "unresolved conflict"
    if not is_known(f):
        return "unknown"
    if f["provenance"] == "claude_inferred":
        return "inferred, not confirmed"
    return "not locked"


def grade(m: dict) -> dict:
    schema = load_yaml(resource_path("templates/manifest.schema.yaml"), {})
    entries = list(schema.get("fields", [])) + profile_entries(m)
    mode = field_value(m, "study.run_mode")
    mode_known = mode in ("self_run", "hosted_api")
    eval_mode = mode if mode_known else "self_run"

    core_missing = [p for p in schema.get("core_fields", []) if not is_known(get_path(m, p))]
    critical_missing, a_missing, recommended_missing = [], [], []
    for e in entries:
        f = get_path(m, e["path"])
        if critical_in(m, e, eval_mode):
            if not is_locked(f):
                item = {"field": e["path"], "reason": _reason(f), "fix": e.get("fix", "")}
                (a_missing if e.get("required_for") == "A" else critical_missing).append(item)
        elif applicable(m, e) and not e.get("critical") and not is_known(f):
            recommended_missing.append({"field": e["path"], "fix": e.get("fix", "")})

    openness = field_value(m, "model.openness")
    openness_locked = is_locked(get_path(m, "model.openness"))
    reasons = []
    if core_missing:
        g = "D"
        reasons.append("Core fields missing: " + ", ".join(core_missing) + ". The run cannot be re-executed.")
    elif not mode_known:
        g = "C"
        reasons.append("Run mode unknown.")
    elif mode == "hosted_api":
        g = "C"
        reasons.append("Hosted API mode: the provider controls weights, inference stack and hardware, so "
                       "only statistical reproduction is possible. Re-running in self_run mode with the "
                       "public weights gives a cross-mode agreement check.")
    elif critical_missing:
        g = "C"
        reasons.append(f"{len(critical_missing)} critical field(s) not locked.")
    elif openness == "fully_open" and openness_locked and not a_missing:
        g = "A"
        reasons.append("Fully open model, self-run, all critical fields locked.")
    else:
        g = "B"
        if openness == "fully_open" and a_missing:
            reasons.append("Fully open model, but the openness record is incomplete, so grade A is not reached.")
        else:
            reasons.append("Open-weight model, self-run, all critical fields locked. Grade B is the maximum "
                           "for open-weight models.")

    improve = []
    for item in critical_missing + (a_missing if openness == "fully_open" else []):
        improve.append(f"{item['field']} ({item['reason']}): {item['fix'] or 'provide or confirm the value.'}")
    if mode == "hosted_api":
        improve.append("Re-run the study in self_run mode with pinned public weights to reach grade B or A.")
    if g == "D":
        improve.insert(0, "Provide the prompt files, the input data and the model ID so the run can be re-executed.")

    acc = field_value(m, "hardware.accelerator")
    notes = [] if mode == "hosted_api" else [f"Bit-identical results are expected only on the same hardware class ({acc or 'unknown'}) with "
             "the same engine version, precision and batch size. Other hardware (e.g. Apple Silicon vs "
             "NVIDIA, or GPUs with and without MXFP4 support) gives statistical reproduction only."]
    if mode == "hosted_api":
        notes.append("Hosted API: the provider may change weights, quantization, inference stack or hardware "
                     "behind the same model name. Archived raw responses and system fingerprints are the only "
                     "record of what was served.")
    if field_value(m, "model.converted_or_quantized") is True:
        notes.append("Converted or quantized weights were used instead of the original release. Reproduce "
                     "with the same converted artifact (see model.format, model.quantization, model.ollama_digest).")
    if field_value(m, "decoding.batch_invariant") is not True and mode == "self_run":
        notes.append("Batch-invariant mode was not recorded as on. Results can depend on batch size and "
                     "request order; keep both identical when reproducing.")
    return {"grade": g, "meaning": GRADE_MEANING[g], "reasons": reasons,
            "critical_missing": critical_missing, "a_only_missing": a_missing if openness == "fully_open" else [],
            "recommended_missing": recommended_missing, "how_to_improve": improve, "notes": notes}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    pkg = Path(sys.argv[1])
    m = load_yaml(pkg / "manifest.yaml")
    res = grade(m)
    m["grade"] = res
    save_yaml(pkg / "manifest.yaml", m)
    print(f"Grade {res['grade']}: {res['meaning']}")
    for r in res["reasons"]:
        print("  " + r)
    for i in res["how_to_improve"]:
        print("  - " + i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
