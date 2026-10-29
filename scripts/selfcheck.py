"""Integrity self-check for a finished package.

Checks: schema completeness, provenance labels, checksums, secret scan, no stray weights,
required files present.

Usage: python selfcheck.py <package_dir>   (exit code 1 on any failure)
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import (PROVENANCE_LABELS, WEIGHT_SUFFIXES, field_value, get_path, is_field, iter_fields,
                    load_yaml, resource_path)
import hash_files
import secret_scan

REQUIRED_FILES = ("manifest.yaml", "README.md", "REPRODUCIBILITY.md", "checksums.sha256",
                  "scripts/reproduce.py", "scripts/reproduce.sh", "scripts/classify.py", "scripts/compare.py")


def selfcheck(pkg: Path) -> dict:
    errors: list[str] = []
    m = load_yaml(pkg / "manifest.yaml")
    if not m:
        return {"ok": False, "errors": ["manifest.yaml missing or empty"]}
    schema = load_yaml(resource_path("templates/manifest.schema.yaml"), {})
    for e in schema.get("fields", []):
        if not is_field(get_path(m, e["path"])):
            errors.append(f"schema: field {e['path']} missing from manifest")
    for path, f in iter_fields(m):
        if f["provenance"] not in PROVENANCE_LABELS:
            errors.append(f"provenance: {path} has invalid label {f['provenance']}")
        if f["provenance"] == "claude_inferred" and not f.get("basis"):
            errors.append(f"provenance: {path} is claude_inferred without a basis")
        if "source" not in f:
            errors.append(f"provenance: {path} has no source")
    if "grade" not in m:
        errors.append("manifest has no grade")
    for rel in REQUIRED_FILES:
        if not (pkg / rel).exists():
            errors.append(f"missing required file {rel}")
    chk = hash_files.verify(pkg)
    for k in ("mismatched", "missing", "unlisted"):
        for rel in chk[k]:
            errors.append(f"checksum {k}: {rel}")
    for f in secret_scan.scan_paths([pkg]):
        errors.append(f"secret scan: {Path(f['file']).relative_to(pkg).as_posix()}:{f['line']} [{f['kind']}]")
    archived = field_value(m, "model.weights_archived") is True
    for p in pkg.rglob("*"):
        if p.is_file() and p.name.lower().endswith(WEIGHT_SUFFIXES) and not archived:
            errors.append(f"weights: {p.relative_to(pkg).as_posix()} present but weights were not approved")
    return {"ok": not errors, "errors": errors}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    res = selfcheck(Path(sys.argv[1]))
    for e in res["errors"]:
        print("FAIL: " + e)
    print("Self-check passed." if res["ok"] else f"Self-check FAILED ({len(res['errors'])} problem(s)).")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
