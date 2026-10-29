"""Scan files for API keys, tokens and other secrets. Exit code 1 if anything is found.

Checks known token formats and the values of secret-looking environment variables
(names containing KEY, TOKEN, SECRET, PASSWORD or AUTH) set in the current shell.

Usage: python secret_scan.py <dir_or_file> [...]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PATTERNS = [
    ("openai_key", r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"),
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    ("groq_key", r"\bgsk_[A-Za-z0-9]{20,}"),
    ("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}"),
    ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    ("private_key", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ("bearer_header", r"(?i)authorization[\"']?\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._\-]{16,}"),
    ("generic_assignment", r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token)[\"']?\s*[:=]\s*"
                           r"[\"']([A-Za-z0-9_\-./+=]{24,})[\"']"),
]
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH", re.IGNORECASE)
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".llmrp"}
MAX_BYTES = 50 * 1024 * 1024
# The scanner's own source contains example patterns, not secrets.
SELF_NAMES = {"secret_scan.py", "test_selfcheck.py"}


def env_secret_values() -> list[tuple[str, str]]:
    out = []
    for k, v in os.environ.items():
        if SECRET_ENV.search(k) and v and len(v) >= 12 and not v.lower() in ("true", "false"):
            out.append((k, v))
    return out


def scan_text(text: str, env_values: list[tuple[str, str]], patterns: bool = True) -> list[tuple[int, str]]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, rx in PATTERNS if patterns else ():
            if re.search(rx, line):
                hits.append((lineno, kind))
        for name, val in env_values:
            if val in line:
                hits.append((lineno, f"value_of_env_{name}"))
    return hits


def scan_paths(paths: list[Path], include_self: bool = False) -> list[dict]:
    env_values = env_secret_values()
    findings = []
    for root in paths:
        files = [root] if root.is_file() else [p for p in sorted(root.rglob("*"))
                                                if p.is_file() and not (set(p.parts) & SKIP_DIRS)]
        for p in files:
            # Our own pattern definitions are only checked against env-var values.
            use_patterns = include_self or p.name not in SELF_NAMES
            try:
                if p.stat().st_size > MAX_BYTES:
                    findings.append({"file": str(p), "line": 0, "kind": "too_large_to_scan"})
                    continue
                raw = p.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            text = raw.decode("utf-8", errors="replace")
            for lineno, kind in scan_text(text, env_values, use_patterns):
                findings.append({"file": str(p), "line": lineno, "kind": kind})
    return findings


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    findings = scan_paths([Path(a) for a in sys.argv[1:]])
    for f in findings:
        print(f"SECRET? {f['file']}:{f['line']} [{f['kind']}]")
    if findings:
        print(f"{len(findings)} potential secret(s) found. Remove them before sharing or pushing.")
        return 1
    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
