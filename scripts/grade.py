"""Compute the reproducibility grade from manifest.yaml and the schema.

The skill only produces grade A or B packages:
  A  fully open model, self-run, every critical field locked (incl. the openness record)
  B  open-weight model (or fully open without a complete openness record), self-run,
     every critical field locked
Anything else is "not eligible": the package is not built, and the missing items are listed.

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
    None: "not eligible: this skill only packages runs that reach grade A or B",
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


def is_critical(m: dict, entry: dict) -> bool:
    if entry.get("critical") is not True:
        return False
    if not applicable(m, entry):
        return False
    if entry.get("unless") and condition(m, entry["unless"]):
        return False
    return True


def profile_entries(m: dict) -> list[dict]:
    prof = get_profile(field_value(m, "model_profile.name") or "default")
    return [{"path": f"model_profile.{fname}", "critical": bool(spec.get("critical")),
             "fix": f"Record {fname}: {spec.get('description', '')}".strip()}
            for fname, spec in (prof.get("fields") or {}).items()]


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
    critical_missing, a_missing, recommended_missing = [], [], []
    for e in entries:
        f = get_path(m, e["path"])
        if is_critical(m, e):
            if not is_locked(f):
                item = {"field": e["path"], "reason": _reason(f), "fix": e.get("fix", "")}
                (a_missing if e.get("required_for") == "A" else critical_missing).append(item)
        elif applicable(m, e) and e.get("critical") is not True and not is_known(f):
            recommended_missing.append({"field": e["path"], "fix": e.get("fix", "")})

    mode = field_value(m, "study.run_mode")
    if mode != "self_run":
        critical_missing.insert(0, {"field": "study.run_mode", "reason": f"run mode is {mode!r}",
                                    "fix": "Only self-run studies (a machine the researcher controlled) are supported."})
    openness = field_value(m, "model.openness")
    reasons = []
    if critical_missing:
        g = None
        reasons.append(f"{len(critical_missing)} critical field(s) not locked. This skill only packages runs "
                       "that reach grade A or B, so no package was built.")
    elif openness == "fully_open" and not a_missing:
        g = "A"
        reasons.append("Fully open model, self-run, all critical fields locked.")
    else:
        g = "B"
        if openness == "fully_open":
            reasons.append("Fully open model, but the openness record is incomplete, so grade A is not reached.")
        else:
            reasons.append("Open-weight model, self-run, all critical fields locked. Grade B is the maximum "
                           "for open-weight models.")

    improve = [f"{i['field']} ({i['reason']}): {i['fix'] or 'provide or confirm the value.'}"
               for i in critical_missing + (a_missing if openness == "fully_open" else [])]

    acc = field_value(m, "hardware.accelerator")
    notes = [f"Bit-identical results are expected only on the same hardware class ({acc or 'unknown'}) with "
             "the same engine version, precision and batch size. Other hardware (e.g. Apple Silicon vs "
             "NVIDIA, or GPUs with different kernel support for the weight format) gives statistical agreement only; judge it "
             "with Cohen's kappa and Krippendorff's alpha."]
    if field_value(m, "model.converted_or_quantized") is True:
        notes.append("Converted or quantized weights were used instead of the original release. Reproduce "
                     "with the same converted artifact (see model.format, model.quantization, model.ollama_digest).")
    if field_value(m, "decoding.batch_invariant") is not True:
        notes.append("Batch-invariant mode was not recorded as on. Results can depend on batch size and "
                     "request order; keep both identical when reproducing.")
    return {"grade": g, "eligible": g in ("A", "B"), "meaning": GRADE_MEANING[g], "reasons": reasons,
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
    print(f"Grade {res['grade'] or '-'}: {res['meaning']}")
    for r in res["reasons"]:
        print("  " + r)
    for i in res["how_to_improve"]:
        print("  - " + i)
    return 0 if res["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
