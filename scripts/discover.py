"""Find candidate files in a project directory and list them for researcher confirmation.

Nothing is hashed or locked here. The researcher must confirm the selection with
`confirm.py select` before evidence collection.

Usage:
  python discover.py --project P [--labels F] [--data F] [--prompt F ...] [--config F]
                     [--raw-outputs F ...] [--logs F ...] [--env-package F ...] [--gold F]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from common import (ID_COLUMN_NAMES, LABEL_COLUMN_NAMES, TEXT_COLUMN_NAMES, WEIGHT_SUFFIXES,
                    guess_column, mtime_iso, now_iso, posix_rel, read_records, save_yaml, state_dir)

SKIP_DIRS = {".git", ".llmrp", "__pycache__", "node_modules", ".venv", "venv", "site-packages",
             ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
MAX_FILES = 5000

DATA_EXT = {".csv", ".tsv", ".jsonl", ".ndjson", ".json", ".parquet", ".xlsx", ".txt"}
TEXT_CFG_EXT = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
PROMPT_EXT = {".yaml", ".yml", ".txt", ".md", ".json", ".jinja", ".j2", ".tmpl"}
CODE_EXT = {".py", ".sh", ".ipynb", ".r", ".jl", ".bat", ".ps1"}

OUTPUT_DIRS = {"outputs", "output", "results", "predictions", "runs", "preds"}
LOG_DIRS = {"logs", "log"}
ENV_DIRS = {"env", "environment"}

# Roles that take one file versus a list.
SINGLE_ROLES = ("config", "data", "labels", "gold")
LIST_ROLES = ("prompt_files", "raw_outputs", "requests", "logs", "code", "env_package")
ALL_ROLES = SINGLE_ROLES + LIST_ROLES


def _roles_for(rel: str) -> list[str]:
    parts = [p.lower() for p in rel.split("/")]
    name = parts[-1]
    stem, ext = os.path.splitext(name)
    dirs = set(parts[:-1])
    in_out, in_log, in_env = bool(dirs & OUTPUT_DIRS), bool(dirs & LOG_DIRS), bool(dirs & ENV_DIRS)
    roles = []
    if "example" in name or name.startswith("readme"):
        return roles
    if (ext in PROMPT_EXT and re.search(r"prompt|codebook|system|instruction|few[_-]?shot|template", stem)
            and not in_out and not in_log) or ("prompts" in dirs and ext in PROMPT_EXT):
        roles.append("prompt_files")
    if ext in TEXT_CFG_EXT and re.search(r"config|settings|params|hparams|^task|endpoint", stem) \
            and not in_out and not in_log:
        roles.append("config")
    if ext in DATA_EXT and not (in_out or in_log or in_env) and "prompt_files" not in roles \
            and "config" not in roles \
            and ("data" in dirs or re.search(r"data|input|corpus|texts?$|items|records|sample", stem)):
        roles.append("data")
    if ext in DATA_EXT and (re.search(r"label|pred|annot|result|classif", stem)
                            or (in_out and re.search(r"output", stem))) \
            and not re.search(r"gold|human|request|raw|response", stem):
        roles.append("labels")
    if ext in DATA_EXT | {".log"} and re.search(r"raw|response|generation|completion|model_output", stem):
        roles.append("raw_outputs")
    if ext in {".json", ".jsonl"} and re.search(r"request", stem):
        roles.append("requests")
    if ext == ".log" or in_log or (ext in {".json", ".yaml", ".yml"}
                                   and re.search(r"provenance|run_info|metadata|run_log|validation", stem)):
        roles.append("logs")
    if ext in CODE_EXT and "tests" not in dirs:
        roles.append("code")
    if in_env or re.search(r"^environment|^requirements|pip[_-]?freeze|^conda|^dockerfile|digest|"
                           r"nvidia[_-]?smi|^hardware|^uv\.lock$|^poetry\.lock$|\.lock$", name):
        if not name.endswith(".md"):
            roles.append("env_package")
    if ext in DATA_EXT and re.search(r"gold|human|manual|ground[_-]?truth|reference_label", stem):
        roles.append("gold")
    return roles


def _columns(path: Path) -> dict:
    try:
        rows = read_records(path)
    except BaseException:  # read_records exits on unreadable files
        return {}
    if not rows:
        return {}
    cols = list(rows[0].keys())
    return {"columns": cols, "n_rows": len(rows),
            "id_column": guess_column(cols, ID_COLUMN_NAMES),
            "text_column": guess_column(cols, TEXT_COLUMN_NAMES),
            "label_column": guess_column(cols, LABEL_COLUMN_NAMES)}


def discover(project: Path, specified: dict[str, list[str]]) -> dict:
    project = project.resolve()
    candidates: dict[str, list[dict]] = {r: [] for r in ALL_ROLES}
    weights_seen = []
    n = 0
    for root, dirs, files in os.walk(project):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fname in sorted(files):
            p = Path(root) / fname
            rel = posix_rel(p, project)
            if fname.lower().endswith(WEIGHT_SUFFIXES):
                weights_seen.append(rel)
                continue
            n += 1
            if n > MAX_FILES:
                break
            for role in _roles_for(rel):
                candidates[role].append({"path": rel, "mtime": mtime_iso(p), "size": p.stat().st_size,
                                         "origin": "auto_discovered"})
    # Researcher-specified files win and are marked as such.
    for role, paths in specified.items():
        for raw in paths:
            p = Path(raw)
            if not p.is_absolute():
                p = project / p
            if not p.exists():
                raise SystemExit(f"ERROR: specified {role} file does not exist: {raw}")
            rel = posix_rel(p, project) if str(p.resolve()).startswith(str(project)) else str(p.resolve())
            candidates[role] = [c for c in candidates[role] if c["path"] != rel]
            candidates[role].insert(0, {"path": rel, "mtime": mtime_iso(p), "size": p.stat().st_size if p.is_file() else 0,
                                        "origin": "researcher_specified"})
    columns = {}
    for role in ("data", "labels", "gold"):
        for c in candidates[role][:5]:
            p = project / c["path"] if not Path(c["path"]).is_absolute() else Path(c["path"])
            if p.is_file():
                columns[c["path"]] = _columns(p)
    return {"project": str(project), "generated_at": now_iso(), "candidates": candidates,
            "columns": columns, "weights_seen": weights_seen,
            "specified": {k: v for k, v in specified.items() if v}}


def print_table(result: dict) -> None:
    print(f"Project: {result['project']}")
    for role in ALL_ROLES:
        cands = result["candidates"][role]
        print(f"\n[{role}] {len(cands)} candidate(s)")
        for c in cands:
            flag = "*" if c["origin"] == "researcher_specified" else " "
            print(f" {flag} {c['mtime']}  {c['size']:>10}  {c['path']}")
    if result["columns"]:
        print("\nColumn guesses (confirm with the researcher):")
        for path, info in result["columns"].items():
            if info:
                print(f"  {path}: id={info['id_column']} text={info['text_column']} "
                      f"label={info['label_column']} rows={info['n_rows']} columns={info['columns']}")
    if result["weights_seen"]:
        print("\nWeight files seen (never copied by default):")
        for w in result["weights_seen"]:
            print("  " + w)
    print("\n* = researcher-specified. Ask the researcher to confirm each role, then run confirm.py select.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--prompt", dest="prompt_files", action="append", default=[])
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--raw-outputs", dest="raw_outputs", action="append", default=[])
    ap.add_argument("--requests", action="append", default=[])
    ap.add_argument("--logs", action="append", default=[])
    ap.add_argument("--code", action="append", default=[])
    ap.add_argument("--env-package", dest="env_package", action="append", default=[])
    ap.add_argument("--gold", action="append", default=[])
    args = ap.parse_args()
    specified = {r: getattr(args, r) for r in ALL_ROLES}
    result = discover(args.project, specified)
    save_yaml(state_dir(args.project) / "candidates.yaml", result)
    print_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
