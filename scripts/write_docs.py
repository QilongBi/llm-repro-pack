"""Write README.md, REPRODUCIBILITY.md and env/REBUILD.md from manifest.yaml."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from string import Template
from typing import Any

from common import (TOOL_NAME, TOOL_VERSION, field_value, get_path, is_field, iter_fields, resource_path)
from model_profiles import get_profile


def _v(m: dict, path: str, default: str = "unknown") -> str:
    v = field_value(m, path)
    if v in (None, ""):
        return default
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _bullets(items: list[str], empty: str = "None.") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def _short(v: Any, n: int = 80) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 3] + "..."


def reproduce_command(m: dict) -> str:
    mode = field_value(m, "study.run_mode")
    if mode == "hosted_api":
        env = field_value(m, "hosted_api.api_key_env") or "PROVIDER_API_KEY"
        return (f"export {env}=...   # never write the key to a file\n"
                f"   python scripts/reproduce.py --package . --engine hosted_api --api-key-env {env}")
    eng = field_value(m, "engine.name")
    eng = eng if eng in ("transformers", "vllm", "ollama") else "transformers"
    return f"python scripts/reproduce.py --package . --engine {eng}"


def expected(m: dict, g: dict) -> str:
    if g["grade"] in ("A", "B"):
        return ("Re-running with the same engine version, precision, batch size and hardware class should "
                "reproduce the archived raw outputs exactly (exact match rate 1.0). On different hardware "
                "expect high but imperfect agreement; judge it by Cohen's kappa and Krippendorff's alpha.")
    if g["grade"] == "C":
        return ("Exact output identity is not expected. Judge the reproduction by agreement with the archived "
                "labels (exact match rate, Cohen's kappa, Krippendorff's alpha).")
    return "The run cannot be re-executed from this package. Use the archived outputs for downstream analysis."


def readme(m: dict) -> str:
    g = m["grade"]
    data_sec = ("## Data\n\nThe input data is included under `data/`." if field_value(m, "data.shareable")
                else "## Data\n\nThe input data is not included. See `data/ACCESS.md` for how to obtain it "
                     "and its SHA-256.")
    hw = _v(m, "hardware.accelerator")
    if field_value(m, "hardware.instance_type"):
        hw += f" ({_v(m, 'hardware.cloud_provider', '')} {_v(m, 'hardware.instance_type')})"
    t = Template(resource_path("templates/README.md.tmpl").read_text(encoding="utf-8"))
    return t.safe_substitute(
        title=m["package"]["title"], tool=TOOL_NAME, tool_version=TOOL_VERSION,
        created_at=m["package"]["created_at"], repo_id=_v(m, "model.repo_id"), revision=_v(m, "model.revision"),
        openness=_v(m, "model.openness"), run_mode=_v(m, "study.run_mode"),
        engine=f"{_v(m, 'engine.name')} {_v(m, 'engine.version', '')}".strip()
        if field_value(m, "study.run_mode") != "hosted_api" else f"hosted: {_v(m, 'hosted_api.provider')}",
        hardware=hw if field_value(m, "study.run_mode") != "hosted_api" else "provider-controlled",
        n_items=_v(m, "data.n_items"), grade=g["grade"], grade_meaning=g["meaning"],
        reproduce_command=reproduce_command(m), data_section=data_sec, expected_result=expected(m, g))


def reproducibility(m: dict) -> str:
    g = m["grade"]
    missing = [f"`{i['field']}`: {i['reason']}" for i in g["critical_missing"] + g["a_only_missing"]]
    rec = [f"`{i['field']}`" + (f": {i['fix']}" if i.get("fix") else "") for i in g["recommended_missing"]]
    conflicts = []
    for c in m.get("conflicts") or []:
        cands = "; ".join(f"{_short(x['value'])!s} [{x['provenance']}, {x['source']}]" for x in c["candidates"])
        status = c["status"] + (f" -> {_short(c['resolution'])}" if c.get("resolution") is not None else "")
        conflicts.append(f"`{c['field']}` ({status}): {cands}")
    inferred = [f"`{p}` = {_short(f['value'])} (basis: {f.get('basis')})" for p, f in iter_fields(m)
                if f["provenance"] == "claude_inferred"]
    confirmed = [f"`{p}` = {_short(f['value'])}" for p, f in iter_fields(m)
                 if f["provenance"] == "researcher_confirmed" and "inferred" in str(f.get("note", ""))]
    inferred_text = _bullets(inferred, "None remain unconfirmed.")
    if confirmed:
        inferred_text += "\n\nInferred and then confirmed by the researcher:\n\n" + _bullets(confirmed)
    caveats = list(g["notes"]) + list(m.get("warnings") or [])
    prof = get_profile(field_value(m, "model_profile.name") or "default")
    caveats += prof.get("notes") or []
    if field_value(m, "model.openness") == "open_weight":
        caveats.append("Open-weight model: the training data is not disclosed, so data contamination and "
                       "training-time choices cannot be audited.")
    counts = Counter(f["provenance"] for _, f in iter_fields(m))
    summary = "| Provenance | Fields |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in sorted(counts.items()))
    t = Template(resource_path("templates/REPRODUCIBILITY.md.tmpl").read_text(encoding="utf-8"))
    return t.safe_substitute(
        grade=g["grade"], grade_meaning=g["meaning"], grade_reason=" ".join(g["reasons"]),
        missing_critical=_bullets(missing), how_to_improve=_bullets(g["how_to_improve"], "Nothing to add."),
        missing_recommended=_bullets(rec), conflicts=_bullets(conflicts), inferred=inferred_text,
        caveats=_bullets(caveats), provenance_summary=summary)


def rebuild(m: dict, pkg: Path) -> str:
    lines = ["# Rebuilding the environment", ""]
    provided = sorted(p.relative_to(pkg).as_posix() for p in (pkg / "env" / "provided").rglob("*")
                      if p.is_file()) if (pkg / "env" / "provided").exists() else []
    digest = field_value(m, "engine.container_image_digest")
    if digest and digest != "unknown":
        lines += ["## Container (best)", "", f"    docker pull {digest}", ""]
    if provided:
        lines += ["## Researcher-provided environment files", ""] + [f"- `{p}`" for p in provided] + [""]
        for p in provided:
            if p.endswith((".yml", ".yaml")):
                lines += [f"    conda env create -f {p} -n repro", ""]
            elif "requirements" in p or "freeze" in p or p.endswith(".lock"):
                lines += [f"    python -m venv .venv && .venv/bin/pip install -r {p}", ""]
    lines += ["## Reconstructed requirements (fallback)", "",
              "`env/requirements.reconstructed.txt` lists library versions from the manifest with their "
              "provenance. Lines marked claude_inferred describe the packaging machine, not the run machine.",
              "", "    pip install -r env/requirements.reconstructed.txt", "",
              "## Target", "",
              f"- OS: {_v(m, 'engine.os')}", f"- Python: {_v(m, 'engine.python')}",
              f"- CUDA: {_v(m, 'engine.cuda')}, driver {_v(m, 'engine.driver')}",
              f"- Accelerator: {_v(m, 'hardware.accelerator')}", ""]
    return "\n".join(lines)


def write_all(pkg: Path, m: dict) -> None:
    (pkg / "README.md").write_text(readme(m), encoding="utf-8", newline="\n")
    (pkg / "REPRODUCIBILITY.md").write_text(reproducibility(m), encoding="utf-8", newline="\n")
    (pkg / "env").mkdir(exist_ok=True)
    (pkg / "env" / "REBUILD.md").write_text(rebuild(m, pkg), encoding="utf-8", newline="\n")
