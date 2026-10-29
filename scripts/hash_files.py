"""Write or verify checksums.sha256 for a package (sha256sum format).

Usage:
  python hash_files.py write  <package_dir>
  python hash_files.py verify <package_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import sha256_file

CHECKSUM_FILE = "checksums.sha256"
EXCLUDE_DIRS = {"__pycache__", "reproduction", ".pytest_cache"}


def package_files(pkg: Path) -> list[Path]:
    out = []
    for p in sorted(pkg.rglob("*")):
        if not p.is_file() or p.name == CHECKSUM_FILE:
            continue
        rel_parts = p.relative_to(pkg).parts
        if set(rel_parts[:-1]) & EXCLUDE_DIRS or p.suffix == ".pyc":
            continue
        out.append(p)
    return out


def write(pkg: Path) -> int:
    lines = [f"{sha256_file(p)}  {p.relative_to(pkg).as_posix()}" for p in package_files(pkg)]
    (pkg / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


def verify(pkg: Path) -> dict:
    """Return {'ok': bool, 'mismatched': [...], 'missing': [...], 'unlisted': [...]}."""
    cf = pkg / CHECKSUM_FILE
    if not cf.exists():
        return {"ok": False, "mismatched": [], "missing": [CHECKSUM_FILE], "unlisted": []}
    listed = {}
    for line in cf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, rel = line.split("  ", 1)
            listed[rel] = h
    mismatched, missing = [], []
    for rel, h in listed.items():
        p = pkg / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != h:
            mismatched.append(rel)
    present = {p.relative_to(pkg).as_posix() for p in package_files(pkg)}
    unlisted = sorted(present - set(listed))
    return {"ok": not (mismatched or missing or unlisted), "mismatched": mismatched, "missing": missing,
            "unlisted": unlisted}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "verify"):
        print(__doc__)
        return 2
    pkg = Path(sys.argv[2])
    if sys.argv[1] == "write":
        print(f"Wrote {write(pkg)} checksums.")
        return 0
    res = verify(pkg)
    for k in ("mismatched", "missing", "unlisted"):
        for rel in res[k]:
            print(f"{k.upper()}: {rel}")
    print("Checksums OK." if res["ok"] else "Checksum verification FAILED.")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
