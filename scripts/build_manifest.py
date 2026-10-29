"""Assemble the package directory and write manifest.yaml.

Reads .llmrp/selection.yaml, evidence.yaml and answers.yaml from the project.
Copies the confirmed files, renders model inputs, computes hashes, and copies the
skill's scripts so the package can reproduce itself. Weights are never copied unless
--archive-weights is given and the license allows redistribution.

Normally called through package.py.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common import (SCHEMA_VERSION, TOOL_NAME, TOOL_VERSION, WEIGHT_SUFFIXES, get_path, is_field,
                    load_yaml, make_field, mtime_iso, now_iso, read_records, resource_path,
                    resolve_candidates, save_yaml, set_path, sha256_file, state_dir, unknown_field,
                    write_jsonl, SCRIPTS_DIR, field_value, load_labels)
from model_profiles import get_profile
import render_inputs as ri

ROLE_DEST = {
    "prompt_files": ("prompts", {"prompts", "prompt"}),
    "config": ("scripts/original", set()),
    "code": ("scripts/original", set()),
    "data": ("data", {"data"}),
    "labels": ("outputs", {"outputs", "output", "results"}),
    "raw_outputs": ("outputs", {"outputs", "output", "results"}),
    "requests": ("outputs", {"outputs", "output", "results"}),
    "logs": ("logs", {"logs", "log"}),
    "env_package": ("env/provided", {"env", "environment"}),
    "gold": ("validation", {"validation"}),
    "dev_set": ("validation/development", {"validation", "dev"}),
    "codebook": ("prompts/codebook", {"prompts", "docs"}),
    "stability_runs": ("outputs/stability", {"outputs", "output", "results"}),
    "practices": ("docs", {"docs"}),
}
MAX_COPY_BYTES = 500 * 1024 * 1024
REDISTRIBUTABLE_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "openrail"}


class Builder:
    def __init__(self, project: Path, out: Path, opts: argparse.Namespace):
        self.project = project.resolve()
        self.out = out.resolve()
        self.opts = opts
        st = state_dir(self.project)
        self.sel = load_yaml(st / "selection.yaml", {})
        self.ev = load_yaml(st / "evidence.yaml", {})
        self.answers = load_yaml(st / "answers.yaml", {})
        if not self.sel.get("confirmed_by_researcher"):
            raise SystemExit("ERROR: file selection is not confirmed (confirm.py select).")
        if not self.ev:
            raise SystemExit("ERROR: no evidence.yaml. Run collect_env.py first.")
        self.roles = self.sel.get("roles", {})
        self.copied: dict[str, list[dict]] = {}
        self.path_map: dict[str, str] = {}
        self.warnings: list[str] = list(self.ev.get("warnings", []))
        self.m: dict[str, Any] = {}
        self.spec: dict | None = None
        self._reqs: list[dict] = []

    # ------------------------------------------------------------ copying
    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.project / p

    def _dest(self, role: str, src: Path) -> Path:
        dest_dir, drop = ROLE_DEST[role]
        try:
            rel = src.resolve().relative_to(self.project)
            parts = list(rel.parts)
        except ValueError:
            parts = [src.name]
        if len(parts) > 1 and parts[0].lower() in drop:
            parts = parts[1:]
        return self.out / dest_dir / Path(*parts)

    def copy_role(self, role: str) -> list[dict]:
        entry = self.roles.get(role)
        if not entry:
            return []
        entries = entry if isinstance(entry, list) else [entry]
        out = []
        for e in entries:
            src = self._abs(e["path"])
            files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
            for f in files:
                if f.name.lower().endswith(WEIGHT_SUFFIXES):
                    self.warnings.append(f"Skipped weight file {f.name}; weights are referenced by hash only.")
                    continue
                if f.stat().st_size > MAX_COPY_BYTES:
                    self.warnings.append(f"Skipped {f} (> 500 MB); hash recorded only.")
                    out.append(self._record(role, f, None, e["origin"]))
                    continue
                dst = self._dest(role, f)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                out.append(self._record(role, f, dst, e["origin"]))
        self.copied[role] = out
        return out

    def _record(self, role: str, src: Path, dst: Path | None, origin: str) -> dict:
        try:
            orig = src.resolve().relative_to(self.project).as_posix()
        except ValueError:
            orig = str(src)
        pkg = dst.relative_to(self.out).as_posix() if dst else None
        if pkg:
            self.path_map[orig] = pkg
        return {"package_path": pkg, "original_path": orig, "sha256": sha256_file(src),
                "size": src.stat().st_size, "mtime": mtime_iso(src), "origin": origin}

    # ------------------------------------------------------------ fields
    def resolve_all(self) -> list[dict]:
        conflicts = []
        cands = self.ev.get("candidates", {})
        for path in sorted(set(cands) | set(self.answers)):
            best, conflict = resolve_candidates(path, cands.get(path, []))
            if path in self.answers:
                ans = {k: v for k, v in self.answers[path].items() if k != "recorded_at"}
                if conflict:
                    conflict["status"] = "resolved_by_researcher"
                    conflict["resolution"] = ans["value"]
                    conflicts.append(conflict)
                elif best["provenance"] == "claude_inferred" and ans["provenance"] == "researcher_confirmed":
                    ans.setdefault("note", f"confirmed inferred value (basis: {best.get('basis')})")
                best = ans
            elif conflict:
                conflicts.append(conflict)
            set_path(self.m, path, best)
        return conflicts

    def put(self, path: str, value: Any, prov: str, source: str, basis: str | None = None,
            note: str | None = None, override: bool = True) -> None:
        if not override and is_field(get_path(self.m, path)) and get_path(self.m, path)["provenance"] != "unknown":
            return
        if path in self.answers:
            return
        set_path(self.m, path, make_field(value, prov, source, basis=basis, note=note))

    # ------------------------------------------------------------ build
    def build(self) -> dict:
        if self.out.exists() and any(self.out.iterdir()):
            if not self.opts.overwrite:
                raise SystemExit(f"ERROR: {self.out} is not empty. Pass --overwrite to replace it.")
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        for d in ("env", "prompts", "scripts", "outputs", "logs"):
            (self.out / d).mkdir(exist_ok=True)

        self.m = {"schema_version": SCHEMA_VERSION,
                  "package": {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "created_at": now_iso(),
                              "title": self.opts.title or self.project.name}}
        conflicts = self.resolve_all()
        run_mode = field_value(self.m, "study.run_mode") or "self_run"

        for role in ROLE_DEST:
            if role == "data" and not self.opts.data_shareable:
                continue
            if role in ("gold", "dev_set") and not self.opts.gold_shareable:
                continue
            self.copy_role(role)
        self._data_fields()
        self._output_fields()
        self._prompt_fields(run_mode)
        self._lint()
        self._validation()
        self._stability()
        self._profile_fields()
        self._env_files()
        self._fix_paths()
        self._copy_scripts()
        if self.opts.archive_weights:
            self._archive_weights()
        self._fill_schema()
        files = {}
        for role, recs in self.copied.items():
            if recs:
                files[role] = recs
        for role in ("data", "gold"):
            if role in self.roles and role not in files:
                e = self.roles[role]
                src = self._abs(e["path"])
                files[role] = [self._record(role, src, None, e["origin"])]
        self.m["files"] = files
        self.m["conflicts"] = conflicts
        self.m["warnings"] = sorted(set(self.warnings))
        save_yaml(self.out / "manifest.yaml", self.m)
        return self.m

    def _data_fields(self) -> None:
        e = self.roles.get("data")
        if not e:
            self.warnings.append("No data file selected.")
            return
        src = self._abs(e["path"])
        basis_src = f"confirmed data file {e['path']}"
        rows = read_records(src)
        self.put("data.sha256", sha256_file(src), "researcher_confirmed", basis_src)
        self.put("data.n_items", len(rows), "researcher_confirmed", basis_src)
        self.put("data.original_path", e["path"], "researcher_confirmed", basis_src)
        self.put("data.origin", e["origin"], "researcher_confirmed", "selection.yaml")
        self.put("data.shareable", bool(self.opts.data_shareable), "researcher_provided", "researcher input")
        if self.opts.data_shareable:
            pkg = self.path_map.get(Path(e["path"]).as_posix()) or self._dest("data", src).relative_to(self.out).as_posix()
            self.put("data.path", pkg, "researcher_confirmed", basis_src)
        else:
            self.put("data.path", "not_shared", "researcher_provided", "researcher input",
                     note="data withheld; see data/ACCESS.md")
            acc = self.opts.data_access or ""
            if acc:
                self.put("data.access_instructions", acc, "researcher_provided", "researcher input")
            d = self.out / "data"
            d.mkdir(exist_ok=True)
            (d / "ACCESS.md").write_text(
                "# Data access\n\nThe input data is not included in this package.\n\n"
                f"- Original file name: `{Path(e['path']).name}`\n- SHA-256: `{sha256_file(src)}`\n"
                f"- Items: {len(rows)}\n\n## How to obtain it\n\n{acc or 'Not provided by the researcher.'}\n\n"
                "Place the file at `data/` with the same name and verify the SHA-256 before reproducing.\n",
                encoding="utf-8", newline="\n")
        trunc = [r for r in self.copied.get("logs", []) + self.copied.get("raw_outputs", [])
                 if r["package_path"] and "trunc" in r["package_path"].lower()]
        if trunc:
            self.put("data.truncation", trunc[0]["package_path"], "run_time_evidence", trunc[0]["original_path"])

    def _output_fields(self) -> None:
        lab = self.copied.get("labels") or []
        if lab:
            self.put("outputs.labels", lab[0]["package_path"], "researcher_confirmed", lab[0]["original_path"])
            self.put("outputs.labels_sha256", lab[0]["sha256"], "researcher_confirmed", lab[0]["original_path"])
            self.put("outputs.labels_origin", lab[0]["origin"], "researcher_confirmed", "selection.yaml")
        raw = [r["package_path"] for r in self.copied.get("raw_outputs", []) if r["package_path"]]
        if raw:
            self.put("outputs.raw_outputs", raw, "researcher_confirmed", "selection.yaml")
        logs = [r["package_path"] for r in self.copied.get("logs", []) if r["package_path"]]
        if logs:
            self.put("outputs.logs", logs, "researcher_confirmed", "selection.yaml")

    def _prompt_fields(self, run_mode: str) -> None:
        recs = self.copied.get("prompt_files") or []
        if recs:
            self.put("prompt.files_sha256", {r["package_path"]: r["sha256"] for r in recs},
                     "researcher_confirmed", "SHA-256 of confirmed prompt files")
        spec, spec_rec = None, None
        for r in recs:
            s = ri.load_spec(self.out / r["package_path"])
            if s:
                spec, spec_rec = s, r
                self.spec = s
                break
        requests: list[dict] = []
        source, complete = None, False
        req_files = [(self.out / r["package_path"], r["package_path"])
                     for r in self.copied.get("requests", []) if r["package_path"]]
        n_items = field_value(self.m, "data.n_items")
        if req_files:
            requests = ri.load_archived_requests(req_files)
            source = "archived request payloads"
            covered = {i for r in requests for i in r["item_ids"]}
            complete = bool(n_items) and (len(covered) >= n_items or len(requests) >= n_items)
        if spec is not None:
            p = spec_rec["package_path"]
            self.put("prompt.system_prompt", f"{p}#system" if spec.get("system") else "none", "code_declared", p)
            self.put("prompt.user_template", f"{p}#user_template", "code_declared", p)
            if spec.get("labels"):
                self.put("prompt.codebook", f"{p}#labels", "code_declared", p)
            self.put("prompt.few_shot", {"location": f"{p}#few_shot", "count": len(spec.get("few_shot") or []),
                                         "order": "as listed"}, "code_declared", p)
            if not requests and "data" in self.roles:
                id_col = field_value(self.m, "data.id_column") or "id"
                requests = ri.render_spec_file(self.out / p, self._abs(self.roles["data"]["path"]), id_col)
                source, complete = "prompt spec + data", True
        elif requests and recs:
            sys_msgs = {m["content"] for r in requests for m in r["messages"] if m.get("role") == "system"}
            for r in recs:
                text = (self.out / r["package_path"]).read_text(encoding="utf-8")
                if text in sys_msgs:
                    self.put("prompt.system_prompt", r["package_path"], "run_time_evidence",
                             r["package_path"], note="identical to the system message in archived requests")
            self.put("prompt.user_template", "built by original code; see rendered inputs", "claude_inferred",
                     "scripts/original", basis="no template file; user messages come from the run script")
        self._reqs = requests
        if not requests:
            self.warnings.append("No rendered inputs: no prompt.yaml spec and no archived requests.")
            return
        total = len(requests)
        sampled = requests
        if self.opts.max_rendered and total > self.opts.max_rendered:
            sampled = ri.deterministic_sample(requests, self.opts.max_rendered)
        applied = False
        if run_mode == "self_run" and self.opts.render_chat_template:
            snap = self._snapshot_path()
            if snap:
                prof = get_profile(field_value(self.m, "model_profile.name") or "default")
                err = ri.apply_chat_template(sampled, str(snap), None, ri.template_kwargs_from(self.m, prof))
                if err:
                    self.warnings.append("Chat template not applied: " + err)
                else:
                    applied = True
            else:
                self.warnings.append("Chat template not applied: no local tokenizer snapshot found.")
        out_file = self.out / "prompts" / "rendered" / "rendered_inputs.jsonl"
        write_jsonl(out_file, sampled)
        prov = "run_time_evidence" if source == "archived request payloads" else "code_declared"
        rel = out_file.relative_to(self.out).as_posix()
        self.put("prompt.rendered_inputs_file", rel, prov, source)
        self.put("prompt.rendered_inputs_sha256", sha256_file(out_file), prov, rel)
        self.put("prompt.rendered_inputs_count", {"archived": len(sampled), "total": total}, prov, source)
        self.put("prompt.rendered_inputs_complete", bool(complete and len(sampled) == total), prov, source)
        self.put("prompt.chat_template_applied", applied, "code_declared" if applied else "claude_inferred",
                 "render_inputs.py", basis=None if applied else "chat template rendering was not run")

    def _lint(self) -> None:
        from prompt_lint import lint
        labels = field_value(self.m, "study.label_set")
        labels = labels if isinstance(labels, list) else []
        roles = [m.get("role") for m in self._reqs[0]["messages"]] if self._reqs else None
        if self.spec:
            res = lint(self.spec.get("system") or "", self.spec.get("user_template") or "", labels,
                       self.spec.get("labels") or None, self.spec.get("few_shot"), roles)
        elif self._reqs:
            msgs = self._reqs[0]["messages"]
            system = "\n".join(m["content"] for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str))
            users = [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
            res = lint(system, users[-1] if users else "", labels, None, None, roles)
        else:
            return
        self.put("prompt.lint", res, "code_declared", "prompt_lint.py on the archived prompt",
                 note="advisory checks after Törnberg (2024); the archived prompt is not changed")

    def _snapshot_path(self) -> Path | None:
        rev = field_value(self.m, "model.revision")
        for s in (self.ev.get("info") or {}).get("hf_cache_snapshots") or []:
            if rev and str(s["commit"]).startswith(str(rev)):
                return Path(s["path"])
        if self.ev.get("inputs", {}).get("model_path"):
            p = Path(self.ev["inputs"]["model_path"])
            return p if p.is_dir() else None
        return None

    def _validation(self) -> None:
        e = self.roles.get("gold")
        if not e:
            return
        gold = self._abs(e["path"])
        src_g = e["path"]
        self.put("validation.gold_sha256", sha256_file(gold), "researcher_provided", src_g)
        lab = self.copied.get("labels") or []
        idc = field_value(self.m, "outputs.id_column")
        lc = field_value(self.m, "outputs.label_column")
        gid = getattr(self.opts, "gold_id_column", None) or idc
        glc = getattr(self.opts, "gold_label_column", None) or lc
        if not (lab and idc and lc and gid and glc):
            self.warnings.append("Validation skipped: label or gold columns unknown.")
            return
        from validate import run_validation, write_errors
        multi = field_value(self.m, "study.task_type") == "multi_label"
        sep = field_value(self.m, "study.multi_label_separator")
        preds = load_labels(self.out / lab[0]["package_path"], idc, lc, multi, sep)
        data_rows = read_records(self._abs(self.roles["data"]["path"])) if "data" in self.roles else None
        shots = []
        for ex in (self.spec or {}).get("few_shot") or []:
            inp = ex.get("input")
            shots.append(inp.get("text", "") if isinstance(inp, dict) else str(inp))
        dev_ids = None
        if "dev_set" in self.roles:
            dev_ids = set()
            for d in self.roles["dev_set"]:
                for r in read_records(self._abs(d["path"])):
                    key = gid if gid in r else (field_value(self.m, "data.id_column") or "id")
                    if key in r:
                        dev_ids.add(str(r[key]))
        coders = [c.strip() for c in (getattr(self.opts, "gold_coder_columns", None) or "").split(",") if c.strip()]
        res = run_validation(gold, gid, glc, preds, multi, sep, coder_cols=coders or None,
                             subset_col=getattr(self.opts, "subset_column", None), data_rows=data_rows,
                             data_id=field_value(self.m, "data.id_column"),
                             data_text=field_value(self.m, "data.text_column"),
                             few_shot_texts=shots, dev_ids=dev_ids)
        src = "validate.py: archived labels vs researcher-provided gold set"
        for key in ("n_compared", "accuracy", "macro_f1", "weighted_f1", "cohens_kappa", "krippendorff_alpha",
                    "per_class", "per_class_support", "min_class_support", "small_classes",
                    "human_intercoder_alpha", "llm_vs_coders", "llm_and_humans_alpha", "subsets", "leakage",
                    "confusion_matrix"):
            if res.get(key) is not None:
                self.put(f"validation.{key}", res[key], "researcher_provided", src)
        vdir = self.out / "validation"
        include_text = bool(self.opts.data_shareable)
        write_errors(vdir / "errors.csv", res["errors"], include_text, bool(self.opts.gold_shareable))
        self.put("validation.errors_file", "validation/errors.csv", "researcher_provided", src,
                 note=f"{len(res['errors'])} misclassified items for failure analysis")
        import json as _json
        (vdir / "metrics.json").write_text(_json.dumps({k: v for k, v in res.items() if k != "errors"},
                                                       indent=2, default=list) + "\n", encoding="utf-8")
        if res["small_classes"]:
            self.warnings.append(f"Validation: classes with fewer than 20 items: {', '.join(res['small_classes'])}. "
                                 "Törnberg (2024) recommends at least 20-30 per class.")
        lk = res["leakage"]
        if lk.get("few_shot_in_validation") or lk.get("dev_set_in_validation"):
            self.warnings.append("Validation leakage: validation items also used as few-shot examples or in "
                                 "prompt development: " + ", ".join(lk.get("few_shot_in_validation", []) +
                                                                     lk.get("dev_set_in_validation", [])))

    def _stability(self) -> None:
        runs = self.copied.get("stability_runs") or []
        lab = self.copied.get("labels") or []
        idc = field_value(self.m, "outputs.id_column")
        lc = field_value(self.m, "outputs.label_column")
        if not runs or not lab or not (idc and lc):
            return
        from compare import krippendorff_alpha_nominal, compare_labels
        multi = field_value(self.m, "study.task_type") == "multi_label"
        sep = field_value(self.m, "study.multi_label_separator")
        main = load_labels(self.out / lab[0]["package_path"], idc, lc, multi, sep)
        others = {}
        for r in runs:
            if r["package_path"]:
                try:
                    others[r["package_path"]] = load_labels(self.out / r["package_path"], idc, lc, multi, sep)
                except SystemExit:
                    self.warnings.append(f"Stability run {r['package_path']} has no {idc}/{lc} columns; skipped.")
        if not others:
            return
        ids = sorted(set(main).union(*others.values()))
        alpha = krippendorff_alpha_nominal([[main.get(i)] + [o.get(i) for o in others.values()] for i in ids])
        exact = {k: compare_labels(main, o, multi).get("exact_match_rate") for k, o in others.items()}
        self.put("stability.archived_runs", {"n_runs": len(others) + 1, "files": sorted(others),
                                             "krippendorff_alpha": alpha, "exact_match_vs_main": exact},
                 "run_time_evidence", "archived repeated or paraphrased runs")

    def _profile_fields(self) -> None:
        prof = get_profile(field_value(self.m, "model_profile.name") or "default")
        for fname in (prof.get("fields") or {}):
            path = f"model_profile.{fname}"
            if not is_field(get_path(self.m, path)):
                set_path(self.m, path, unknown_field())

    def _env_files(self) -> None:
        cur = (self.ev.get("info") or {}).get("current_environment") or {}
        save_yaml(self.out / "env" / "current_environment.yaml",
                  {"note": "Captured at packaging time, not at run time. Values here are claude_inferred.", **cur})
        libs = get_path(self.m, "engine.libraries") or {}
        lines = ["# Reconstructed from manifest.yaml (engine.libraries). Provenance per line.",
                 "# Prefer env/provided/ files when present."]
        for lib, f in sorted(libs.items()):
            if is_field(f) and f["value"] not in (None, "unknown"):
                lines.append(f"{lib}=={f['value']}  # {f['provenance']}")
        (self.out / "env" / "requirements.reconstructed.txt").write_text("\n".join(lines) + "\n",
                                                                          encoding="utf-8", newline="\n")
        hw = []
        for k in ("hardware.accelerator", "hardware.accelerator_memory", "hardware.cpu", "hardware.cloud_provider",
                  "hardware.instance_type", "engine.os", "engine.cuda", "engine.driver"):
            f = get_path(self.m, k)
            if is_field(f):
                hw.append(f"{k}: {f['value']}  [{f['provenance']}; source {f['source']}]")
        (self.out / "env" / "hardware.txt").write_text("\n".join(hw) + "\n", encoding="utf-8", newline="\n")

    def _fix_paths(self) -> None:
        f = get_path(self.m, "hosted_api.raw_responses")
        if is_field(f) and isinstance(f["value"], list):
            f["value"] = [self.path_map.get(v, v) for v in f["value"]]

    def _copy_scripts(self) -> None:
        dst = self.out / "scripts"
        for p in sorted(SCRIPTS_DIR.rglob("*.py")):
            if "__pycache__" in p.parts or "resources" in p.relative_to(SCRIPTS_DIR).parts:
                continue
            t = dst / p.relative_to(SCRIPTS_DIR)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
        res = dst / "resources"
        (res / "templates").mkdir(parents=True, exist_ok=True)
        for name in ("manifest.schema.yaml", "REPRODUCTION_REPORT.md.tmpl"):
            shutil.copy2(resource_path(f"templates/{name}"), res / "templates" / name)
        prof_dst = res / "profiles"
        prof_dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(Path(resource_path("profiles")).glob("*.yaml")):
            shutil.copy2(p, prof_dst / p.name)
        sh = resource_path("templates/reproduce.sh.tmpl").read_text(encoding="utf-8")
        (dst / "reproduce.sh").write_text(sh, encoding="utf-8", newline="\n")

    def _archive_weights(self) -> None:
        lic = str(field_value(self.m, "model.license") or "").lower()
        prof = get_profile(field_value(self.m, "model_profile.name") or "default")
        if prof.get("redistribution_allowed") is False or lic not in REDISTRIBUTABLE_LICENSES:
            raise SystemExit(f"ERROR: --archive-weights refused: license '{lic}' is not on the redistribution "
                             "allowlist. Reference weights by ID and hash instead.")
        snap = self._snapshot_path()
        if not snap:
            raise SystemExit("ERROR: --archive-weights: no local snapshot found.")
        d = self.out / "weights"
        d.mkdir(exist_ok=True)
        for p in sorted(snap.iterdir()):
            if p.is_file():
                shutil.copy2(p, d / p.name)
        self.put("model.weights_archived", True, "researcher_provided", "researcher request (--archive-weights)",
                 note=f"license {lic} permits redistribution")

    def _fill_schema(self) -> None:
        schema = load_yaml(resource_path("templates/manifest.schema.yaml"), {})
        for entry in schema.get("fields", []):
            if not is_field(get_path(self.m, entry["path"])):
                set_path(self.m, entry["path"], unknown_field())


def build(project: Path, out: Path, opts: argparse.Namespace) -> dict:
    return Builder(project, out, opts).build()
