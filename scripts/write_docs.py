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
    eng = field_value(m, "engine.name")
    eng = eng if eng in ("transformers", "vllm", "ollama") else "transformers"
    return f"python scripts/reproduce.py --package . --engine {eng}"


def expected(m: dict, g: dict) -> str:
    return ("Re-running with the same engine version, precision, batch size and hardware class should "
            "reproduce the archived raw outputs exactly (exact match rate 1.0). On different hardware "
            "expect high but imperfect agreement; judge it by Cohen's kappa and Krippendorff's alpha.")


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
        engine=f"{_v(m, 'engine.name')} {_v(m, 'engine.version', '')}".strip(),
        hardware=hw,
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


STATUS_MARK = {"met": "met", "partial": "partial", "not_met": "NOT MET", "not_documented": "not documented",
               "n/a": "n/a"}


def best_practices_doc(m: dict) -> str:
    bp = m.get("best_practices") or {}
    lines = ["# Best-practice checklist", "",
             "Checked against Törnberg (2024), *Best Practices for Text Annotation with Large Language Models* "
             "(arXiv:2402.05129). This checklist is separate from the reproducibility grade in "
             "`REPRODUCIBILITY.md`: it covers validity and research practice, not only re-execution.", ""]
    counts = bp.get("counts") or {}
    lines += ["Summary: " + ", ".join(f"{STATUS_MARK.get(k, k)} {v}" for k, v in sorted(counts.items())), ""]
    if not bp.get("validated"):
        lines += ["**No validation against human labels was provided.** Törnberg (2024) treats validation as a "
                  "basic requirement for every publication that uses LLM annotations.", ""]
    point = None
    for it in bp.get("items") or []:
        if it["point"] != point:
            point = it["point"]
            lines += ["", f"## {point}", "", "| Requirement | Status | Evidence |", "|---|---|---|"]
        ev = str(it["evidence"]).replace("|", "/")
        lines.append(f"| {it['requirement']} | {STATUS_MARK.get(it['status'], it['status'])} | {ev} |")
    lines += ["", "Items marked *not documented* can be added with `practices.yaml` "
              "(`confirm.py practices`) and a rebuild. The archived run itself is never changed.", ""]
    return "\n".join(lines)


def prompt_codebook(m: dict, pkg: Path) -> str:
    import render_inputs as ri
    from common import read_jsonl
    lines = ["# Prompt codebook", "",
             "Everything needed to reproduce the LLM coding: categories, prompts, examples and settings. "
             "Generated from `manifest.yaml`; the prompt files in `prompts/` are authoritative.", "",
             "## Model and settings", "", "| Setting | Value | Provenance |", "|---|---|---|"]
    rows = [("model", "model.repo_id"), ("revision", "model.revision"),
            ("engine", "engine.name"), ("engine version", "engine.version"),
            ("temperature", "decoding.temperature"), ("top_p", "decoding.top_p"), ("top_k", "decoding.top_k"),
            ("seed", "decoding.seed"), ("max tokens", "decoding.max_tokens"), ("batch size", "decoding.batch_size"),
            ("precision", "decoding.precision")]
    for name, path in rows:
        f = get_path(m, path)
        if is_field(f) and f["provenance"] != "unknown":
            lines.append(f"| {name} | {_short(f['value'])} | {f['provenance']} |")
    for k, f in sorted((m.get("model_profile") or {}).items()):
        if is_field(f) and k != "name" and f["provenance"] != "unknown":
            lines.append(f"| {k} | {_short(f['value'])} | {f['provenance']} |")
    spec = None
    ut = str(field_value(m, "prompt.user_template") or "")
    if "#user_template" in ut:
        spec = ri.load_spec(pkg / ut.split("#", 1)[0])
    lines += ["", "## Categories", ""]
    if spec and spec.get("labels"):
        for lab in spec["labels"]:
            if isinstance(lab, dict):
                lines.append(f"- **{lab['name']}**: {lab.get('definition', '(no definition)')}")
                for ex in lab.get("examples") or []:
                    lines.append(f"  - example: {ex}")
            else:
                lines.append(f"- **{lab}**")
    else:
        for lab in field_value(m, "study.label_set") or []:
            lines.append(f"- **{lab}**")
    cb = (m.get("files") or {}).get("codebook") or []
    if cb:
        lines += ["", "Human coder guidelines: " + ", ".join(f"`{c['package_path']}`" for c in cb if c.get("package_path"))]
    lines += ["", "## Prompt", ""]
    fence = "`" * 3
    if spec:
        lines += ["System:", "", fence, str(spec.get("system") or "(none)").rstrip(), fence, "",
                  "User template:", "", fence, str(spec.get("user_template")).rstrip(), fence]
        if spec.get("few_shot"):
            lines += ["", "Few-shot examples, in this order:", ""]
            for i, ex in enumerate(spec["few_shot"], 1):
                lines.append(f"{i}. input `{ex.get('input')}` -> `{ex.get('output')}`")
    else:
        rel = field_value(m, "prompt.rendered_inputs_file")
        if rel and rel != "unknown" and (pkg / rel).exists():
            first = read_jsonl(pkg / rel)[:1]
            for msg in first[0]["messages"] if first else []:
                body = str(msg.get("content"))
                body = body if len(body) < 1500 else body[:1500] + " ..."
                lines += [f"{msg.get('role')}:", "", fence, body, fence, ""]
            lines.append("(First archived request shown; all requests are in `" + rel + "`.)")
    lines += ["", "## Output and label extraction", "",
              f"Rule: `{_short(field_value(m, 'outputs.label_extraction_rule'), 200)}`. "
              "Invalid outputs are kept as invalid; they are never repaired.", ""]
    return "\n".join(lines)


def write_all(pkg: Path, m: dict) -> None:
    (pkg / "BEST_PRACTICES.md").write_text(best_practices_doc(m), encoding="utf-8", newline="\n")
    (pkg / "PROMPT_CODEBOOK.md").write_text(prompt_codebook(m, pkg), encoding="utf-8", newline="\n")
    (pkg / "README.md").write_text(readme(m), encoding="utf-8", newline="\n")
    (pkg / "REPRODUCIBILITY.md").write_text(reproducibility(m), encoding="utf-8", newline="\n")
    (pkg / "env").mkdir(exist_ok=True)
    (pkg / "env" / "REBUILD.md").write_text(rebuild(m, pkg), encoding="utf-8", newline="\n")
