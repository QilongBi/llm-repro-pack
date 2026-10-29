"""Reproduce mode: verify a package, rebuild its environment, re-run the archived scripts,
compare with the archived labels and write a reproduction report.

Usage:
  python reproduce.py --package PKG --verify-only
  python reproduce.py --package PKG --rebuild --dry-run
  python reproduce.py --package PKG --engine ENGINE [--sample N] [classify options]
Exit codes: 0 ok, 3 verification failed, 4 inputs incomplete, 5 run failed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (TOOL_NAME, TOOL_VERSION, field_value, get_path, is_field, load_labels, load_yaml,
                    now_iso, resource_path, sha256_file, values_equal)
from compare import _jsonable, compare_labels
import hash_files


def verify(pkg: Path, m: dict) -> list[str]:
    errors = []
    chk = hash_files.verify(pkg)
    for k in ("mismatched", "missing", "unlisted"):
        errors += [f"checksum {k}: {r}" for r in chk[k]]
    for rel, h in (field_value(m, "prompt.files_sha256") or {}).items():
        p = pkg / rel
        if not p.exists() or sha256_file(p) != h:
            errors.append(f"prompt file hash mismatch: {rel}")
    rel = field_value(m, "prompt.rendered_inputs_file")
    want = field_value(m, "prompt.rendered_inputs_sha256")
    if rel and rel != "unknown" and want and want != "unknown":
        if not (pkg / rel).exists() or sha256_file(pkg / rel) != want:
            errors.append(f"rendered inputs hash mismatch: {rel}")
    lab = field_value(m, "outputs.labels")
    want = field_value(m, "outputs.labels_sha256")
    if lab and lab != "unknown" and want and want != "unknown":
        if sha256_file(pkg / lab) != want:
            errors.append(f"archived labels hash mismatch: {lab}")
    data = field_value(m, "data.path")
    want = field_value(m, "data.sha256")
    if data == "not_shared":
        hits = [p for p in (pkg / "data").glob("*") if p.name != "ACCESS.md"]
        if hits and not any(sha256_file(p) == want for p in hits):
            errors.append("data file placed under data/ does not match data.sha256")
    return errors


def rebuild_commands(pkg: Path, m: dict) -> list[list[str]]:
    digest = field_value(m, "engine.container_image_digest")
    if digest and digest != "unknown":
        return [["docker", "pull", str(digest)]]
    prov = pkg / "env" / "provided"
    files = sorted(p for p in prov.rglob("*") if p.is_file()) if prov.exists() else []
    for p in files:
        if p.suffix in (".yml", ".yaml"):
            return [["conda", "env", "create", "-f", str(p), "-n", "llmrp-repro"]]
    for p in files:
        if "requirements" in p.name or "freeze" in p.name or p.suffix == ".lock" and p.name != "poetry.lock":
            return [[sys.executable, "-m", "pip", "install", "-r", str(p)]]
    return [[sys.executable, "-m", "pip", "install", "-r", str(pkg / "env" / "requirements.reconstructed.txt")]]


def env_diff(m: dict, versions_now: dict) -> list[str]:
    out = []
    libs = get_path(m, "engine.libraries") or {}
    for lib, f in sorted(libs.items()):
        if not is_field(f) or f["value"] in (None, "unknown"):
            continue
        now = versions_now.get(lib)
        if now is None:
            out.append(f"{lib}: run {f['value']} [{f['provenance']}], now not installed")
        elif not values_equal(now, f["value"]):
            out.append(f"{lib}: run {f['value']} [{f['provenance']}], now {now}")
    return out


def current_versions() -> dict:
    import importlib.metadata as md
    from collect_env import KEY_LIBS
    out = {}
    for lib in KEY_LIBS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def report(pkg: Path, m: dict, args, verification: list[str], diff: list[str], metrics: dict | None,
           hw_now: str) -> str:
    original = field_value(m, "study.run_mode")
    cross = original == "hosted_api" and args.engine not in ("hosted_api", "mock")
    g = (m.get("grade") or {}).get("grade", "?")
    if metrics:
        mt = "| Metric | Value |\n|---|---|\n" + "\n".join(
            f"| {k} | {_fmt(metrics.get(k))} |" for k in
            ("n_compared", "exact_match_rate", "cohens_kappa", "krippendorff_alpha", "macro_f1_vs_a", "mean_jaccard")
            if k in metrics)
        if metrics.get("missing_in_b"):
            mt += f"\n\nItems without a reproduced label: {len(metrics['missing_in_b'])}."
    else:
        mt = "Not computed."
    same_hw = values_equal(hw_now, field_value(m, "hardware.accelerator") or "")
    if not metrics:
        interp = "No comparison was made."
    elif metrics.get("exact_match_rate") == 1.0:
        interp = "All compared labels match the archived labels exactly."
    elif g in ("A", "B") and same_hw and not cross and not diff:
        interp = ("Labels differ although the package is grade " + g + " and the hardware class and library "
                  "versions match. Check batch size, request order and engine settings.")
    else:
        interp = ("Exact identity was not expected here (" + ("cross-mode run; " if cross else "") +
                  ("different hardware; " if not same_hw else "") + ("library versions differ; " if diff else "") +
                  f"package grade {g}). Judge agreement by kappa and alpha.")
    t = Template(resource_path("templates/REPRODUCTION_REPORT.md.tmpl").read_text(encoding="utf-8"))
    return t.safe_substitute(
        tool=TOOL_NAME, tool_version=TOOL_VERSION, created_at=now_iso(), package=str(pkg), grade=g,
        original_mode=original, engine=args.engine,
        cross_mode="yes: hosted API package re-run with public weights" if cross else "no",
        n_compared=(metrics or {}).get("n_compared", 0), sampled=args.sample or "no",
        hardware_now=hw_now, hardware_then=field_value(m, "hardware.accelerator") or "unknown",
        verification="All checks passed." if not verification else "\n".join(f"- {e}" for e in verification),
        env_diff="\n".join(f"- {d}" for d in diff) or "No differences in recorded library versions.",
        metrics=mt, interpretation=interp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--engine")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path)
    for opt in ("--revision", "--repo-id", "--base-url", "--model-name", "--api-key-env", "--ollama-tag"):
        ap.add_argument(opt)
    args = ap.parse_args()
    pkg = args.package.resolve()
    m = load_yaml(pkg / "manifest.yaml")
    if not m:
        print("ERROR: manifest.yaml not found", file=sys.stderr)
        return 3
    errors = verify(pkg, m)
    if errors:
        for e in errors:
            print("VERIFY FAIL: " + e, file=sys.stderr)
        print("Stopping: the package does not match its checksums and hashes.", file=sys.stderr)
        return 3
    print("Verification passed.")
    if args.verify_only:
        return 0
    if args.rebuild:
        cmds = rebuild_commands(pkg, m)
        for c in cmds:
            print("REBUILD: " + " ".join(c))
            if not args.dry_run:
                r = subprocess.run(c)
                if r.returncode != 0:
                    return 5
        if args.dry_run or not args.engine:
            return 0
    if not args.engine:
        print("ERROR: --engine is required to re-run", file=sys.stderr)
        return 2
    out = (args.out or pkg / "reproduction").resolve()
    cmd = [sys.executable, str(pkg / "scripts" / "classify.py"), "--package", str(pkg), "--engine", args.engine,
           "--out", str(out)]
    if args.sample:
        cmd += ["--sample", str(args.sample)]
    for opt in ("revision", "repo_id", "base_url", "model_name", "api_key_env", "ollama_tag"):
        if getattr(args, opt):
            cmd += ["--" + opt.replace("_", "-"), getattr(args, opt)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"Archived runner failed with exit code {r.returncode}.", file=sys.stderr)
        return r.returncode if r.returncode in (3, 4) else 5
    multi = field_value(m, "study.task_type") == "multi_label"
    sep = field_value(m, "study.multi_label_separator")
    a = load_labels(pkg / field_value(m, "outputs.labels"), field_value(m, "outputs.id_column"),
                    field_value(m, "outputs.label_column"), multi, sep)
    b = load_labels(out / "labels.jsonl", "id", "label", multi, sep)
    if args.sample:
        a = {k: v for k, v in a.items() if k in b}
    metrics = _jsonable(compare_labels(a, b, multi))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    from collect_env import current_environment
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    hw_now = current_environment(False).get("accelerator", "unknown")
    diff = env_diff(m, {**current_versions(), **{k: v for k, v in info.get("versions", {}).items() if v}})
    (out / "REPRODUCTION_REPORT.md").write_text(report(pkg, m, args, [], diff, metrics, hw_now),
                                                encoding="utf-8", newline="\n")
    print(f"Exact match rate: {_fmt(metrics.get('exact_match_rate'))}, kappa: {_fmt(metrics.get('cohens_kappa'))}, "
          f"alpha: {_fmt(metrics.get('krippendorff_alpha'))}")
    print(f"Report: {out / 'REPRODUCTION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
