"""Record researcher decisions: file selection, confirmed values and corrections.

Subcommands:
  select   Confirm which file plays each role (after discover.py).
             python confirm.py select --project P labels=outputs/labels.csv data=data/items.csv \
                 prompt_files=prompt.yaml prompt_files=codebook.md ...
             Use role= (empty) to clear a role.
  field    Record a value for a manifest field.
             python confirm.py field --project P decoding.seed=0 --as researcher_provided
             python confirm.py field --project P model.revision --as researcher_confirmed
               (no '=value' confirms the currently resolved value)
             --as claude_inferred requires --basis.
  columns  Record column names: python confirm.py columns --project P data.id_column=id ...
  practices  Load a filled-in best-practice record (templates/practices.yaml):
             python confirm.py practices --project P --file practices.yaml
  show     Print the selection and recorded answers.
  pending  List fields that still need the researcher (inferred values and conflicts).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (PROVENANCE_LABELS, load_yaml, make_field, now_iso, parse_scalar,
                    resolve_candidates, save_yaml, state_dir)
from discover import ALL_ROLES, LIST_ROLES


def _paths(project: Path) -> dict[str, Path]:
    s = state_dir(project)
    return {"candidates": s / "candidates.yaml", "selection": s / "selection.yaml",
            "answers": s / "answers.yaml", "evidence": s / "evidence.yaml"}


def cmd_select(project: Path, pairs: list[str]) -> None:
    paths = _paths(project)
    cands = load_yaml(paths["candidates"], {}).get("candidates", {})
    sel = load_yaml(paths["selection"], {"roles": {}})
    roles = sel.setdefault("roles", {})
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"ERROR: expected role=path, got {pair}")
        role, path = pair.split("=", 1)
        if role not in ALL_ROLES:
            raise SystemExit(f"ERROR: unknown role {role}. Roles: {', '.join(ALL_ROLES)}")
        if path == "":
            roles.pop(role, None)
            continue
        p = Path(path) if Path(path).is_absolute() else project / path
        if not p.exists():
            raise SystemExit(f"ERROR: {path} does not exist")
        origin = "researcher_specified"
        for c in cands.get(role, []):
            if c["path"] == path:
                origin = c["origin"]
        entry = {"path": path, "origin": origin}
        if role in LIST_ROLES:
            lst = roles.setdefault(role, [])
            if not any(e["path"] == path for e in lst):
                lst.append(entry)
        else:
            roles[role] = entry
    sel["confirmed_by_researcher"] = True
    sel["confirmed_at"] = now_iso()
    save_yaml(paths["selection"], sel)
    print("Selection saved.")
    cmd_show(project)


def _current_value(project: Path, field_path: str):
    ev = load_yaml(_paths(project)["evidence"], {})
    cands = ev.get("candidates", {}).get(field_path, [])
    if not cands:
        return None
    best, _ = resolve_candidates(field_path, cands)
    return best["value"]


def cmd_field(project: Path, items: list[str], as_label: str, basis: str | None, source: str | None) -> None:
    if as_label not in PROVENANCE_LABELS or as_label == "unknown":
        raise SystemExit(f"ERROR: --as must be one of {PROVENANCE_LABELS[:-1]}")
    paths = _paths(project)
    answers = load_yaml(paths["answers"], {})
    for item in items:
        if "=" in item:
            key, raw = item.split("=", 1)
            value = parse_scalar(raw)
        else:
            key = item
            value = _current_value(project, key)
            if value is None:
                raise SystemExit(f"ERROR: no collected value for {key}; give key=value")
        answers[key] = make_field(value, as_label, source or "researcher input", basis=basis)
        answers[key]["recorded_at"] = now_iso()
        print(f"{key} = {value!r} [{as_label}]")
    save_yaml(paths["answers"], answers)


def cmd_practices(project: Path, file: Path) -> None:
    from common import flatten, is_blank
    paths = _paths(project)
    answers = load_yaml(paths["answers"], {})
    n = 0
    for key, val in flatten(load_yaml(file, {}) or {}).items():
        if is_blank(val):
            continue
        answers[key] = make_field(val, "researcher_provided", f"best-practice record {file.name}")
        answers[key]["recorded_at"] = now_iso()
        n += 1
    save_yaml(paths["answers"], answers)
    print(f"Recorded {n} best-practice answers from {file}.")


def cmd_show(project: Path) -> None:
    paths = _paths(project)
    sel = load_yaml(paths["selection"], {})
    print("Selection:")
    for role, entry in (sel.get("roles") or {}).items():
        entries = entry if isinstance(entry, list) else [entry]
        for e in entries:
            print(f"  {role:<13} {e['path']}  ({e['origin']})")
    answers = load_yaml(paths["answers"], {})
    if answers:
        print("Answers:")
        for k, v in answers.items():
            print(f"  {k} = {v['value']!r} [{v['provenance']}]")


def pending(project: Path) -> tuple[list, list]:
    paths = _paths(project)
    ev = load_yaml(paths["evidence"], {})
    answers = load_yaml(paths["answers"], {})
    inferred, conflicts = [], []
    for key, cands in (ev.get("candidates") or {}).items():
        if key in answers:
            continue
        best, conflict = resolve_candidates(key, cands)
        if conflict:
            conflicts.append(conflict)
        elif best["provenance"] == "claude_inferred":
            inferred.append((key, best))
    return inferred, conflicts


def cmd_pending(project: Path) -> int:
    inferred, conflicts = pending(project)
    if conflicts:
        print("Conflicts (ask the researcher which value is right):")
        for c in conflicts:
            print(f"  {c['field']}:")
            for cand in c["candidates"]:
                print(f"    - {cand['value']!r} [{cand['provenance']}] from {cand['source']}")
    if inferred:
        print("Inferred values (ask the researcher to confirm or correct each):")
        for key, f in inferred:
            print(f"  {key} = {f['value']!r}  basis: {f.get('basis')}")
    if not conflicts and not inferred:
        print("Nothing pending.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--project", type=Path, required=True)
    s.add_argument("pairs", nargs="+")
    f = sub.add_parser("field")
    f.add_argument("--project", type=Path, required=True)
    f.add_argument("items", nargs="+")
    f.add_argument("--as", dest="as_label", default="researcher_confirmed")
    f.add_argument("--basis")
    f.add_argument("--source")
    c = sub.add_parser("columns")
    c.add_argument("--project", type=Path, required=True)
    c.add_argument("items", nargs="+")
    pr = sub.add_parser("practices")
    pr.add_argument("--project", type=Path, required=True)
    pr.add_argument("--file", type=Path, required=True)
    for name in ("show", "pending"):
        p = sub.add_parser(name)
        p.add_argument("--project", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "select":
        cmd_select(args.project, args.pairs)
    elif args.cmd == "field":
        cmd_field(args.project, args.items, args.as_label, args.basis, args.source)
    elif args.cmd == "columns":
        cmd_field(args.project, args.items, "researcher_provided", None, "researcher input")
    elif args.cmd == "practices":
        cmd_practices(args.project, args.file)
    elif args.cmd == "show":
        cmd_show(args.project)
    elif args.cmd == "pending":
        return cmd_pending(args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
