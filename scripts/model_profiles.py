"""Load and match per-model profiles (profiles/*.yaml)."""
from __future__ import annotations

import fnmatch
from pathlib import Path

from common import load_yaml, resource_path


def load_profiles() -> list[dict]:
    d = resource_path("profiles")
    out = []
    for p in sorted(Path(d).glob("*.yaml")):
        prof = load_yaml(p, {})
        prof.setdefault("name", p.stem)
        prof.setdefault("fields", {})
        prof.setdefault("match", [])
        out.append(prof)
    return out


def get_profile(name: str) -> dict:
    for p in load_profiles():
        if p["name"] == name:
            return p
    return default_profile()


def default_profile() -> dict:
    for p in load_profiles():
        if p["name"] == "default":
            return p
    return {"name": "default", "fields": {}, "match": [], "openness": "open_weight"}


def match_profile(*names: str | None) -> tuple[dict, str | None]:
    """Return (profile, matched pattern). Falls back to the default profile."""
    cands = [n.lower() for n in names if n]
    for prof in load_profiles():
        if prof["name"] == "default":
            continue
        for pat in prof.get("match", []):
            for n in cands:
                if fnmatch.fnmatch(n, pat.lower()):
                    return prof, pat
    return default_profile(), None


def profile_hints(profile: dict, repo_id: str | None) -> dict:
    hints = profile.get("hints") or {}
    if not repo_id:
        return {}
    return hints.get(repo_id.lower(), {})
