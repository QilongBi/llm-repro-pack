"""Shared helpers for llm-repro-pack scripts.

Requires Python >= 3.9 and PyYAML. Everything here is deterministic.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("ERROR: PyYAML is required. Install it with: pip install pyyaml\n")
    raise

TOOL_NAME = "llm-repro-pack"
TOOL_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"

SCRIPTS_DIR = Path(__file__).resolve().parent

PROVENANCE_LABELS = (
    "run_time_evidence",
    "researcher_provided",
    "code_declared",
    "claude_inferred",
    "researcher_confirmed",
    "unknown",
)
# Provenance labels that count as "locked" for grading.
LOCKED_PROVENANCE = frozenset(
    {"run_time_evidence", "researcher_provided", "code_declared", "researcher_confirmed"}
)
# Lower rank wins when candidates are resolved (evidence hierarchy).
EVIDENCE_RANK = {
    "researcher_confirmed": 0,
    "researcher_provided": 1,
    "run_time_evidence": 2,
    "code_declared": 3,
    "claude_inferred": 4,
    "unknown": 5,
}
UNKNOWN = "unknown"
STATE_DIRNAME = ".llmrp"

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt", ".npz", ".h5", ".onnx")


# ---------------------------------------------------------------- resources

def resource_path(rel: str) -> Path:
    """Locate a bundled resource in the skill layout or in a package's scripts/resources copy."""
    for base in (SCRIPTS_DIR.parent, SCRIPTS_DIR / "resources"):
        p = base / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"Bundled resource not found: {rel}")


def state_dir(project: Path) -> Path:
    d = Path(project) / STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- fields

def make_field(value: Any, provenance: str, source: str, basis: str | None = None,
               note: str | None = None) -> dict:
    if provenance not in PROVENANCE_LABELS:
        raise ValueError(f"Invalid provenance label: {provenance}")
    if provenance == "claude_inferred" and not basis:
        raise ValueError("claude_inferred values must state their basis")
    f: dict[str, Any] = {"value": value, "provenance": provenance, "source": source}
    if basis:
        f["basis"] = basis
    if note:
        f["note"] = note
    return f


def unknown_field(note: str | None = None) -> dict:
    return make_field(UNKNOWN, "unknown", "none", note=note)


def is_field(obj: Any) -> bool:
    return isinstance(obj, dict) and "provenance" in obj and "value" in obj


def is_known(f: Any) -> bool:
    if not is_field(f):
        return False
    if f["provenance"] == "unknown":
        return False
    v = f["value"]
    return v not in (None, "", UNKNOWN) and v != [] and v != {}


def is_locked(f: Any) -> bool:
    return is_known(f) and f["provenance"] in LOCKED_PROVENANCE and not f.get("conflict")


# ---------------------------------------------------------------- value normalization

_DTYPE_ALIASES = {
    "bf16": "bfloat16", "torch.bfloat16": "bfloat16",
    "fp16": "float16", "half": "float16", "torch.float16": "float16",
    "fp32": "float32", "float": "float32", "torch.float32": "float32",
}


def norm_value(v: Any) -> Any:
    """Normalize a value so equivalent spellings compare equal."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return round(float(s), 6)
        except ValueError:
            pass
        s = _DTYPE_ALIASES.get(s, s)
        if re.fullmatch(r"v\d[\w.+\-]*", s):
            s = s[1:]
        return s
    if isinstance(v, (list, tuple)):
        return tuple(norm_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((str(k), norm_value(x)) for k, x in v.items()))
    return v


def values_equal(a: Any, b: Any) -> bool:
    na, nb = norm_value(a), norm_value(b)
    if na == nb:
        return True
    # Short and full git/HF commit hashes refer to the same commit.
    if isinstance(na, str) and isinstance(nb, str) and re.fullmatch(r"[0-9a-f]{7,40}", na) \
            and re.fullmatch(r"[0-9a-f]{7,40}", nb):
        return na.startswith(nb) or nb.startswith(na)
    return False


def resolve_candidates(path: str, cands: list[dict]) -> tuple[dict, dict | None]:
    """Pick the highest-ranked candidate. Report a conflict when candidates disagree.

    The conflict is never silently dropped: the chosen field is marked conflict=True
    until the researcher resolves it.
    """
    if not cands:
        return unknown_field(), None
    ordered = sorted(cands, key=lambda c: EVIDENCE_RANK[c["provenance"]])
    best = dict(ordered[0])
    distinct: list[Any] = []
    for c in ordered:
        if not any(values_equal(c["value"], d) for d in distinct):
            distinct.append(c["value"])
    if len(distinct) > 1:
        best["conflict"] = True
        conflict = {"field": path, "status": "unresolved", "candidates": ordered}
        return best, conflict
    return best, None


# ---------------------------------------------------------------- dotted paths

def get_path(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict) or is_field(nxt):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def iter_fields(d: dict, prefix: str = "") -> Iterable[tuple[str, dict]]:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if is_field(v):
            yield p, v
        elif isinstance(v, dict):
            yield from iter_fields(v, p)


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Nested dict -> {dotted.path: leaf}. Lists are leaves."""
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, p))
        else:
            out[p] = v
    return out


def is_blank(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def field_value(manifest: dict, dotted: str) -> Any:
    f = get_path(manifest, dotted)
    return f["value"] if is_field(f) else None


# ---------------------------------------------------------------- hashing and io

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_yaml(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return default if data is None else data


def save_yaml(path: Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, width=100)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mtime_iso(path: Path) -> str:
    ts = Path(path).stat().st_mtime
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(s: str) -> dt.datetime | None:
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        m = re.fullmatch(r"(\d{8})T(\d{6})(\d*)Z", str(s))
        if not m:
            return None
        d = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d


def posix_rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def die(msg: str, code: int = 2) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
    raise SystemExit(code)


def parse_scalar(text: str) -> Any:
    """Parse a CLI value with YAML rules (numbers, booleans, lists, dicts)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


# ---------------------------------------------------------------- tables and labels

def read_records(path: Path) -> list[dict]:
    """Read a table of records from csv, tsv, jsonl, json or a text file containing JSON."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix in (".jsonl", ".ndjson"):
        return read_jsonl(path)
    if suffix == ".parquet":
        try:
            import pandas as pd  # optional
        except ImportError:
            die("Reading parquet needs pandas and pyarrow")
        return pd.read_parquet(path).to_dict(orient="records")
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        try:
            return [json.loads(ln) for ln in lines]
        except json.JSONDecodeError:
            die(f"Cannot parse records from {path}")
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("predictions", "records", "data", "items", "labels", "results"):
            if isinstance(data.get(key), list):
                return [r for r in data[key] if isinstance(r, dict)]
        for v in data.values():
            if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
                return v
    die(f"No record list found in {path}")
    return []


def to_label(value: Any, multi_label: bool, sep: str | None) -> Any:
    if value is None:
        return None
    if multi_label:
        if isinstance(value, (list, tuple)):
            items = [str(x).strip() for x in value]
        else:
            s = str(value)
            items = [x.strip() for x in (s.split(sep) if sep else re.split(r"[;|,]", s))]
        return tuple(sorted(x for x in items if x))
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if len(value) == 1 else tuple(sorted(str(x) for x in value))
    return str(value).strip()


def load_labels(path: Path, id_col: str, label_col: str, multi_label: bool = False,
                sep: str | None = None) -> dict[str, Any]:
    rows = read_records(path)
    out: dict[str, Any] = {}
    for r in rows:
        if id_col not in r:
            die(f"Column '{id_col}' not found in {path}")
        if label_col not in r:
            die(f"Column '{label_col}' not found in {path}")
        out[str(r[id_col])] = to_label(r[label_col], multi_label, sep)
    return out


ID_COLUMN_NAMES = ("id", "item_id", "doc_id", "uid", "record_id", "idx", "index", "text_id")
TEXT_COLUMN_NAMES = ("text", "content", "body", "sentence", "document", "message", "post", "tweet")
LABEL_COLUMN_NAMES = ("label", "labels", "prediction", "predicted_label", "pred", "category",
                      "class", "answer", "classification")


def guess_column(columns: Iterable[str], names: tuple[str, ...]) -> str | None:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    return None
