"""Build a reproducibility package from a confirmed project (package mode, final step).

Steps: build manifest and copy files -> grade -> README/REPRODUCIBILITY -> checksums -> self-check.

Usage:
  python package.py --project P --data-shareable yes|no [--data-access "how to get it"]
                    [--out DIR] [--title T] [--gold-shareable yes|no]
                    [--gold-id-column C] [--gold-label-column C]
                    [--gold-coder-columns c1,c2] [--subset-column C]
                    [--render-chat-template] [--max-rendered N] [--archive-weights] [--overwrite]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import save_yaml, state_dir
import best_practices
import build_manifest
import grade as grade_mod
import hash_files
import selfcheck
import write_docs


def yes_no(s: str) -> bool:
    if s.lower() in ("yes", "y", "true", "1"):
        return True
    if s.lower() in ("no", "n", "false", "0"):
        return False
    raise argparse.ArgumentTypeError("expected yes or no")


def run(args: argparse.Namespace) -> dict:
    project = args.project.resolve()
    out = (args.out or project.parent / f"{project.name}-repro-pack").resolve()
    if str(out).startswith(str(project) + "\\") or str(out).startswith(str(project) + "/"):
        raise SystemExit("ERROR: put the package outside the project directory.")
    m = build_manifest.build(project, out, args)
    m["grade"] = grade_mod.grade(m)
    m["best_practices"] = best_practices.evaluate(m)
    save_yaml(out / "manifest.yaml", m)
    write_docs.write_all(out, m)
    hash_files.write(out)
    res = selfcheck.selfcheck(out)
    save_yaml(state_dir(project) / "last_package.yaml", {"package": str(out), "grade": m["grade"]["grade"],
                                                         "selfcheck": res})
    return {"package": out, "grade": m["grade"], "selfcheck": res, "best_practices": m["best_practices"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title")
    ap.add_argument("--data-shareable", type=yes_no, required=True)
    ap.add_argument("--data-access")
    ap.add_argument("--gold-shareable", type=yes_no, default=False)
    ap.add_argument("--gold-id-column")
    ap.add_argument("--gold-label-column")
    ap.add_argument("--gold-coder-columns", help="Comma-separated human coder columns in the gold file.")
    ap.add_argument("--subset-column", help="Column (gold or data) for per-subset validation, e.g. language.")
    ap.add_argument("--render-chat-template", action="store_true")
    ap.add_argument("--max-rendered", type=int, default=1000)
    ap.add_argument("--archive-weights", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    res = run(args)
    g = res["grade"]
    print(f"Package: {res['package']}")
    print(f"Grade {g['grade']}: {g['meaning']}")
    for r in g["reasons"]:
        print("  " + r)
    if g["how_to_improve"]:
        print("To raise the grade:")
        for i in g["how_to_improve"]:
            print("  - " + i)
    for e in res["selfcheck"]["errors"]:
        print("SELF-CHECK FAIL: " + e)
    bp = res["best_practices"]
    print("Best-practice checklist (Törnberg 2024): " + ", ".join(f"{k} {v}" for k, v in sorted(bp["counts"].items())))
    if not bp["validated"]:
        print("  WARNING: no validation against human labels. Provide a gold set if one exists.")
    print("Self-check passed." if res["selfcheck"]["ok"] else "Self-check FAILED.")
    return 0 if res["selfcheck"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
