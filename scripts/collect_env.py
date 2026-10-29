"""Collect evidence for every manifest field, following the evidence hierarchy.

Sources, best first:
  1. researcher-provided environment package  -> researcher_provided
  2. run-time evidence (logs, output metadata) -> run_time_evidence
  3. code and config (at the run commit if git) -> code_declared
  4. current environment, model cache, profiles -> claude_inferred (with basis)
  5. nothing                                    -> unknown

All candidates are kept. Conflicts are reported, never silently resolved.

Usage:
  python collect_env.py --project P --model-id REPO [--revision SHA] --run-mode self_run|hosted_api
                        [--hf-cache DIR] [--model-path DIR_OR_FILE] [--hub] [--probe-cloud]
                        [--hash-large-weights]
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import re
import urllib.request
from pathlib import Path
from typing import Any

from common import (UNKNOWN, WEIGHT_SUFFIXES, ID_COLUMN_NAMES, LABEL_COLUMN_NAMES, TEXT_COLUMN_NAMES,
                    guess_column, load_yaml, make_field, mtime_iso, now_iso, parse_iso, read_records,
                    resolve_candidates, run_cmd, save_yaml, sha256_file, sha256_text, state_dir,
                    values_equal)
from model_profiles import match_profile, profile_hints

KEY_LIBS = ("torch", "transformers", "tokenizers", "vllm", "accelerate", "bitsandbytes", "flash-attn",
            "xformers", "triton", "llama-cpp-python", "mlx", "mlx-lm", "ollama", "openai",
            "openai-harmony", "safetensors", "huggingface-hub", "numpy", "sentencepiece",
            "auto-gptq", "autoawq", "sglang")
ENGINE_LIB = {"vllm": "vllm", "transformers": "transformers", "ollama": "ollama", "mlx": "mlx-lm",
              "llama.cpp": "llama-cpp-python", "sglang": "sglang"}

# Leaf keys in JSON/YAML (configs, logs, requests, metadata) mapped to manifest fields.
KEY_MAP = {
    "temperature": "decoding.temperature", "top_p": "decoding.top_p", "top_k": "decoding.top_k",
    "seed": "decoding.seed", "max_tokens": "decoding.max_tokens",
    "max_new_tokens": "decoding.max_tokens", "max_completion_tokens": "decoding.max_tokens",
    "num_predict": "decoding.max_tokens", "max_output_tokens": "decoding.max_tokens",
    "batch_size": "decoding.batch_size", "batch_invariant": "decoding.batch_invariant",
    "deterministic": "decoding.batch_invariant",
    "dtype": "decoding.precision", "torch_dtype": "decoding.precision", "precision": "decoding.precision",
    "revision": "model.revision", "model_revision": "model.revision",
    "quantization": "model.quantization", "quant_method": "model.quantization",
    "engine": "engine.name", "inference_engine": "engine.name", "backend": "engine.name",
    "engine_version": "engine.version",
    "provider": "hosted_api.provider", "base_url": "hosted_api.base_url", "api_base": "hosted_api.base_url",
    "api_key_env": "hosted_api.api_key_env",
    "text_column": "data.text_column", "id_column": "data.id_column", "label_column": "outputs.label_column",
    "task_type": "study.task_type", "run_mode": "study.run_mode",
    "allowed_labels": "study.label_set", "label_set": "study.label_set", "labels": "study.label_set",
    "label_extraction": "outputs.label_extraction_rule",
    "label_extraction_rule": "outputs.label_extraction_rule",
}
MODEL_KEYS = ("model", "model_id", "model_name", "repo_id", "requested_model_id")
TIME_KEYS = ("created", "started_at_utc", "ended_at_utc", "timestamp", "start_time", "end_time",
             "started_at", "ended_at", "finished_at", "created_at")
SKIP_KEYS = {"messages", "content", "records", "predictions", "data", "items", "usage", "choices",
             "prompt", "system", "user_template", "few_shot", "logprobs", "input", "output"}
SECTIONS = ("study", "model", "engine", "decoding", "data", "outputs", "hosted_api", "hardware", "prompt")
FILE_KEYS = {"data.path", "outputs.labels", "outputs.raw_outputs", "outputs.logs"}

NUM = r"([0-9]*\.?[0-9]+)"
CODE_PATTERNS = {
    "decoding.temperature": [r"\btemperature[\"']?\s*[:=]\s*[\"']?" + NUM],
    "decoding.top_p": [r"\btop_p[\"']?\s*[:=]\s*[\"']?" + NUM],
    "decoding.top_k": [r"\btop_k[\"']?\s*[:=]\s*[\"']?(-?[0-9]+)"],
    "decoding.seed": [r"\bseed[\"']?\s*[:=]\s*[\"']?([0-9]+)",
                      r"(?:manual_seed|set_seed)\(\s*([0-9]+)\s*\)"],
    "decoding.max_tokens": [r"\b(?:max_tokens|max_new_tokens|max_completion_tokens|num_predict)"
                            r"[\"']?\s*[:=]\s*[\"']?([0-9]+)"],
    "decoding.batch_size": [r"\bbatch_size[\"']?\s*[:=]\s*[\"']?([0-9]+)"],
    "decoding.precision": [r"\b(?:torch_dtype|dtype)[\"']?\s*[:=]\s*[\"']?(?:torch\.)?"
                           r"(bfloat16|float16|float32|half|bf16|fp16|fp32|auto)\b"],
    "model.revision": [r"\brevision[\"']?\s*[:=]\s*[\"']([0-9a-f]{7,40})[\"']"],
}
LOG_PATTERNS = {
    "engine.libraries.vllm": [r"vLLM (?:API server )?version:?\s*v?([0-9]+\.[0-9]+[\w.+\-]*)",
                              r"LLM engine \(v([0-9]+\.[0-9]+[\w.+\-]*)\)"],
    "engine.libraries.transformers": [r"\btransformers(?:==|\s+version[:\s]+|:\s+|\s+v)([0-9]+\.[0-9]+[\w.+\-]*)"],
    "engine.libraries.torch": [r"\btorch(?:==|\s+version[:\s]+|:\s+|\s+v)([0-9]+\.[0-9]+[\w.+\-]*)"],
    "engine.libraries.ollama": [r"ollama version (?:is )?v?([0-9]+\.[0-9]+[\w.\-]*)"],
    "engine.cuda": [r"CUDA Version:\s*([0-9]+\.[0-9]+)", r"\bcuda(?:==|\s+version[:\s]+)([0-9]+\.[0-9]+)"],
    "engine.driver": [r"Driver Version:\s*([0-9.]+)"],
    "engine.os": [r"^(Linux [\w.\-]+ [0-9][\w.\-+]*)", r"\b(Ubuntu [0-9]+\.[0-9]+(?:\.[0-9]+)?(?: LTS)?)",
                  r"\b(macOS [0-9]+(?:\.[0-9]+)*)", r"^OS:\s*(.+)$"],
    "hardware.accelerator": [r"\b(NVIDIA (?:GeForce |Tesla |RTX )?[A-Z][\w\-]*(?: [\w\-]+){0,2}?)"
                             r"(?=\s{2,}|\s*\||,|\s*$|\s+On\b|\s+Off\b)",
                             r"\b(Apple M[1-9](?: Pro| Max| Ultra)?)\b"],
    "hardware.accelerator_memory": [r"[0-9]+MiB\s*/\s*([0-9]+MiB)", r"Memory:\s*([0-9]+\s*GB)"],
    "hardware.instance_type": [r"instance[-_ ]type[\"']?\s*[:=]\s*[\"']?([a-z0-9][a-z0-9.\-]+)"],
    "model.revision": [r"revision[=:\s\"']+([0-9a-f]{40})"],
    "engine.name": [],
}
ENGINE_SIGNATURES = [
    ("vllm", r"\bfrom vllm\b|\bimport vllm\b|vllm serve|vllm\.entrypoints"),
    ("mlx", r"\bmlx_lm\b"),
    ("llama.cpp", r"\bllama_cpp\b|llama-server|llama-cli"),
    ("sglang", r"\bsglang\b"),
    ("ollama", r"\bollama\b"),
    ("transformers", r"AutoModelForCausalLM|pipeline\(\s*[\"']text-generation"),
]
HOSTED_SIGNATURE = r"chat/completions|from openai import|import openai|\bgroq\b|together\.ai|api\.together"
PROVIDERS = {"groq.com": "Groq", "together.xyz": "Together AI", "together.ai": "Together AI",
             "fireworks.ai": "Fireworks AI", "openrouter.ai": "OpenRouter", "deepinfra.com": "DeepInfra",
             "cerebras.ai": "Cerebras", "sambanova.ai": "SambaNova", "hyperbolic.xyz": "Hyperbolic",
             "huggingface.co": "Hugging Face Inference", "nebius": "Nebius", "novita.ai": "Novita"}
TEXT_EXT = {".txt", ".log", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".cfg", ".ini", ".py",
            ".sh", ".lock", ".csv", ".tsv", ".r", ".ipynb", ""}
MAX_TEXT = 20 * 1024 * 1024


def _num(s: Any) -> Any:
    if isinstance(s, (int, float, bool)) or s is None:
        return s
    t = str(s).strip()
    if re.fullmatch(r"-?[0-9]+", t):
        return int(t)
    try:
        return float(t)
    except ValueError:
        return t


class Evidence:
    def __init__(self, project: Path, run_mode: str):
        self.project = project
        self.run_mode = run_mode
        self.cands: dict[str, list[dict]] = {}
        self.warnings: list[str] = []
        self.info: dict[str, Any] = {}
        self.timestamps: list[tuple[str, str, str]] = []
        self.fingerprints: dict[str, set] = {}
        self.reasoning_seen: list[str] = []
        self.profile_keys: dict[str, str] = {}

    def add(self, path: str, value: Any, prov: str, source: str, basis: str | None = None,
            note: str | None = None) -> None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        if isinstance(value, str) and value.strip().lower() == UNKNOWN:
            return
        lst = self.cands.setdefault(path, [])
        for c in lst:
            if c["provenance"] == prov and values_equal(c["value"], value) and c["source"] == source:
                return
        lst.append(make_field(value, prov, source, basis=basis, note=note))

    def rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.project.resolve()).as_posix()
        except ValueError:
            return str(p)


# ---------------------------------------------------------------- generic scanners

def iter_files(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and not p.name.lower().endswith(WEIGHT_SUFFIXES) and ".git" not in p.parts:
                yield p


def read_text(p: Path) -> str | None:
    if p.suffix.lower() not in TEXT_EXT or p.stat().st_size > MAX_TEXT:
        return None
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


def load_structured(p: Path, text: str) -> list[Any]:
    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            return [load_yaml(p, {})]
        if suffix == ".jsonl":
            out = []
            for line in text.splitlines()[:200]:
                if line.strip():
                    out.append(json.loads(line))
            return out
        if suffix == ".json" or text.lstrip().startswith(("{", "[")):
            obj = json.loads(text)
            return obj[:200] if isinstance(obj, list) else [obj]
    except (ValueError, TypeError):
        return []
    return []


def _labels_from(v: Any) -> list[str] | None:
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return list(v)
    if isinstance(v, list) and v and all(isinstance(x, dict) and "name" in x for x in v):
        return [str(x["name"]) for x in v]
    return None


def scan_obj(ev: Evidence, obj: Any, prov: str, source: str, kind: str, depth: int = 0,
             prefix: str = "") -> None:
    """Map known keys in a JSON/YAML object to manifest fields."""
    if depth > 4:
        return
    if isinstance(obj, list):
        for x in obj[:200]:
            scan_obj(ev, x, prov, source, kind, depth + 1, prefix)
        return
    if not isinstance(obj, dict):
        return
    is_response = "choices" in obj or str(obj.get("object", "")).startswith("chat.completion")
    if is_response:
        for ch in obj.get("choices") or []:
            msg = (ch or {}).get("message") or {}
            if any(msg.get(k) for k in ("reasoning", "reasoning_content", "thinking")):
                ev.reasoning_seen.append(source)
            if isinstance(msg.get("content"), str) and "<|channel|>analysis" in msg["content"]:
                ev.reasoning_seen.append(source)
    for k, v in obj.items():
        kl = str(k).lower()
        dotted = f"{prefix}.{kl}" if prefix else kl
        # Our own config layout: section.key maps directly.
        top = dotted.split(".")[0]
        if kind == "config" and dotted in ("outputs.label_extraction_rule", "hosted_api.parameters") \
                and isinstance(v, dict):
            ev.add(dotted, v, prov, source)
            continue
        if kind == "config" and top in SECTIONS and not isinstance(v, dict) and dotted not in FILE_KEYS:
            if dotted == "study.label_set" or kl in ("labels", "allowed_labels"):
                labels = _labels_from(v)
                if labels:
                    ev.add("study.label_set", labels, prov, source)
                continue
            ev.add(dotted, _num(v) if not isinstance(v, (list, dict)) else v, prov, source)
            continue
        if kind == "config" and dotted in ("task.type",):
            ev.add("study.task_type", v, prov, source)
            continue
        if kind == "config" and dotted in ("task.multi_label_separator",):
            ev.add("study.multi_label_separator", v, prov, source)
            continue
        if kl in ev.profile_keys and not isinstance(v, (dict, list)):
            ev.add(ev.profile_keys[kl], v, prov, source)
        if kl == "system_fingerprint" and v:
            ev.fingerprints.setdefault(str(v), set()).add(source)
            continue
        if kl in TIME_KEYS and v not in (None, ""):
            iso = _to_iso(v)
            if iso:
                ev.timestamps.append((iso, prov, source))
            continue
        if kl in MODEL_KEYS and isinstance(v, str) and v:
            _model_key(ev, kl, v, prov, source, kind, is_response)
            continue
        if kl == "reported_model" and isinstance(v, str):
            ev.add("hosted_api.reported_model", v, prov, source)
            continue
        if kl in ("multi_label",) and isinstance(v, bool):
            ev.add("study.task_type", "multi_label" if v else "single_label", prov, source)
            continue
        if kl in KEY_MAP:
            target = KEY_MAP[kl]
            if target == "study.label_set":
                labels = _labels_from(v)
                if labels:
                    ev.add(target, labels, prov, source)
                continue
            if target == "outputs.label_extraction_rule" and isinstance(v, (dict, str)):
                ev.add(target, v if isinstance(v, dict) else {"type": v}, prov, source)
                continue
            if target == "hosted_api.provider" and ev.run_mode != "hosted_api":
                continue
            if not isinstance(v, (dict, list)):
                ev.add(target, _num(v), prov, source)
                continue
        if kl in SKIP_KEYS:
            continue
        if isinstance(v, (dict, list)):
            scan_obj(ev, v, prov, source, kind, depth + 1, dotted if kind == "config" else "")


def _model_key(ev: Evidence, key: str, v: str, prov: str, source: str, kind: str, is_response: bool) -> None:
    if kind in ("request", "meta", "response") and ev.run_mode == "hosted_api":
        if is_response:
            ev.add("hosted_api.reported_model", v, prov, source)
        else:
            ev.add("hosted_api.model_name", v, prov, source)
        return
    if re.fullmatch(r"[\w.\-]+:[\w.\-]+", v) and "/" not in v:
        ev.add("model.ollama_tag", v, prov, source)
    elif "/" in v and not v.startswith(("http", "/", ".")):
        ev.add("model.repo_id_declared", v, prov, source)


def _to_iso(v: Any) -> str | None:
    if isinstance(v, (int, float)) and v > 1e9:
        return dt.datetime.fromtimestamp(float(v), tz=dt.timezone.utc).replace(microsecond=0).isoformat()
    d = parse_iso(str(v))
    return d.replace(microsecond=0).isoformat() if d else None


def scan_text(ev: Evidence, text: str, prov: str, source: str, patterns: dict[str, list[str]],
              skip_comments: bool = False) -> None:
    lines = text.splitlines()
    if skip_comments:
        lines = [ln for ln in lines if not ln.lstrip().startswith(("#", "//"))]
    body = "\n".join(lines)
    for path, regs in patterns.items():
        for rx in regs:
            for m in re.finditer(rx, body, flags=re.MULTILINE | re.IGNORECASE if path != "hardware.accelerator" else re.MULTILINE):
                val = m.group(1).strip()
                if path == "hardware.accelerator" and re.match(r"NVIDIA (Corporation|Driver|SMI)", val):
                    continue
                ev.add(path, _num(val) if path.startswith("decoding.") else val, prov, source)


def detect_engine(text: str) -> str | None:
    for name, rx in ENGINE_SIGNATURES:
        if re.search(rx, text):
            return name
    return None


# ---------------------------------------------------------------- environment package

def parse_requirements(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def parse_conda(path: Path) -> tuple[dict[str, str], str | None]:
    data = load_yaml(path, {}) or {}
    libs, python = {}, None
    for dep in data.get("dependencies") or []:
        if isinstance(dep, str):
            parts = dep.split("=")
            if len(parts) >= 2:
                name = parts[0].lower().replace("_", "-")
                libs[name] = parts[1]
                if name == "python":
                    python = parts[1]
        elif isinstance(dep, dict) and "pip" in dep:
            libs.update(parse_requirements("\n".join(dep["pip"] or [])))
    return libs, python


def parse_lockfile(text: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r'name\s*=\s*"([^"]+)"\s*\n\s*version\s*=\s*"([^"]+)"', text):
        out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def collect_env_package(ev: Evidence, files: list[Path], run_end: dt.datetime | None) -> None:
    prov = "researcher_provided"
    for p in files:
        text = read_text(p)
        if text is None:
            continue
        src = ev.rel(p)
        note = None
        if run_end and dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc) > run_end + dt.timedelta(days=1):
            note = "post-hoc export, may not match run-time environment"
            ev.warnings.append(f"{src}: modified {mtime_iso(p)}, after the run ended ({run_end.isoformat()}). "
                               "Post-hoc export, may not match run-time environment.")
        name = p.name.lower()
        libs: dict[str, str] = {}
        if name.endswith((".yml", ".yaml")) and "dependencies" in text:
            libs, py = parse_conda(p)
            if py:
                ev.add("engine.python", py, prov, src, note=note)
            ev.add("engine.environment_file", src, prov, src, note=note)
        elif name.endswith(".lock") and 'name = "' in text:
            libs = parse_lockfile(text)
            ev.add("engine.environment_file", src, prov, src, note=note)
        elif "==" in text and re.search(r"^[A-Za-z0-9_.\-]+==", text, re.MULTILINE):
            libs = parse_requirements(text)
            ev.add("engine.environment_file", src, prov, src, note=note)
        for lib, ver in libs.items():
            if lib in KEY_LIBS:
                ev.add(f"engine.libraries.{lib}", ver, prov, src, note=note)
        if name.startswith("dockerfile"):
            m = re.search(r"^FROM\s+(\S+@sha256:[0-9a-f]{64})", text, re.MULTILINE)
            if m:
                ev.add("engine.container_base_image", m.group(1), prov, src, note=note)
        if "digest" in name or '"RepoDigests"' in text:
            m = re.search(r"(\S*@?sha256:[0-9a-f]{64})", text)
            if m:
                ev.add("engine.container_image_digest", m.group(1).strip('"[],'), prov, src, note=note)
        scan_text(ev, text, prov, src, LOG_PATTERNS)
        for obj in load_structured(p, text):
            scan_obj(ev, obj, prov, src, "meta")


# ---------------------------------------------------------------- git and code

def git_run_commit(project: Path, run_end: dt.datetime | None) -> dict | None:
    if run_cmd(["git", "-C", str(project), "rev-parse", "--is-inside-work-tree"]) is None:
        return None
    info: dict[str, Any] = {"repo": True}
    if run_end:
        out = run_cmd(["git", "-C", str(project), "log", "-1", f"--before={run_end.isoformat()}",
                       "--format=%H|%cI"])
        if out and "|" in out:
            sha, when = out.strip().split("|", 1)
            info.update(commit=sha, committed_at=when)
    head = run_cmd(["git", "-C", str(project), "rev-parse", "HEAD"])
    info["head"] = head.strip() if head else None
    return info


def git_show(project: Path, commit: str, rel: str) -> str | None:
    return run_cmd(["git", "-C", str(project), "show", f"{commit}:{rel}"])


def git_changed_since(project: Path, commit: str, rels: list[str]) -> list[str]:
    out = run_cmd(["git", "-C", str(project), "diff", "--name-only", commit, "--", *rels]) or ""
    return [ln for ln in out.splitlines() if ln.strip()]


def collect_code(ev: Evidence, code_files: list[Path], config_files: list[Path], prompt_files: list[Path],
                 run_end: dt.datetime | None, profile: dict) -> None:
    prov = "code_declared"
    git = git_run_commit(ev.project, run_end)
    ev.info["git"] = git
    commit = (git or {}).get("commit")
    if commit:
        ev.add("code.git_commit", commit, "claude_inferred", "git log",
               basis=f"latest commit before the last output time ({run_end.isoformat() if run_end else '?'})")
        rels = [ev.rel(p) for p in code_files + config_files + prompt_files]
        changed = git_changed_since(ev.project, commit, rels)
        if changed:
            ev.warnings.append("Code or config changed after the run commit " + commit[:10] + ": "
                               + ", ".join(changed) + ". Values were read from the run commit.")
    elif git is None and run_end:
        late = [ev.rel(p) for p in code_files + config_files + prompt_files
                if dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc) > run_end]
        if late:
            ev.warnings.append("No git history. These files were modified after the outputs, so values read "
                               "from them have lower confidence: " + ", ".join(late))
    engines = []
    all_code_text = []
    for p in code_files + config_files:
        rel = ev.rel(p)
        text = git_show(ev.project, commit, rel) if commit else None
        src = f"{rel}@{commit[:10]}" if text is not None and commit else rel
        if text is None:
            text = read_text(p)
        if text is None:
            continue
        all_code_text.append(text)
        if p in config_files:
            for obj in load_structured(p, text):
                scan_obj(ev, obj, prov, src, "config")
        else:
            scan_text(ev, text, prov, src, CODE_PATTERNS, skip_comments=True)
            scan_text(ev, text, prov, src, _profile_patterns(profile), skip_comments=True)
            eng = detect_engine(text)
            if eng:
                engines.append((eng, src))
            if re.search(r"VLLM_BATCH_INVARIANT|use_deterministic_algorithms\(\s*True|batch_invariant\s*=\s*True",
                         text):
                ev.add("decoding.batch_invariant", True, prov, src)
            if ev.run_mode == "hosted_api" and re.search(HOSTED_SIGNATURE, text):
                ev.info.setdefault("hosted_client_code", []).append(src)
    if engines and ev.run_mode == "self_run":
        order = [n for n, _ in ENGINE_SIGNATURES]
        engines.sort(key=lambda e: order.index(e[0]))
        ev.add("engine.name", engines[0][0], prov, engines[0][1])
        others = sorted({e for e, _ in engines[1:] if e != engines[0][0]})
        if others:
            ev.info["engine_signatures_also_seen"] = others
    joined = "\n".join(all_code_text)
    for fname, spec in (profile.get("fields") or {}).items():
        for d in spec.get("detect") or []:
            if re.search(d["pattern"], joined):
                ev.add(f"model_profile.{fname}", d["value"], prov, "code scan",
                       note=f"code matches /{d['pattern']}/")
                break
    for p in prompt_files:
        if p.suffix.lower() in (".yaml", ".yml"):
            spec = load_yaml(p, {}) or {}
            if isinstance(spec, dict):
                labels = _labels_from(spec.get("labels"))
                if labels:
                    ev.add("study.label_set", labels, prov, ev.rel(p))
                if isinstance(spec.get("label_extraction"), dict):
                    ev.add("outputs.label_extraction_rule", spec["label_extraction"], prov, ev.rel(p))


def _profile_patterns(profile: dict) -> dict[str, list[str]]:
    return {f"model_profile.{k}": v.get("patterns", []) for k, v in (profile.get("fields") or {}).items()
            if v.get("patterns")}


# ---------------------------------------------------------------- run-time evidence

def collect_runtime(ev: Evidence, files: list[Path], kind_hint: str, profile: dict) -> None:
    prov = "run_time_evidence"
    pats = dict(LOG_PATTERNS)
    pats.update({k: v for k, v in CODE_PATTERNS.items() if k != "model.revision"})
    pats.update(_profile_patterns(profile))
    for p in files:
        text = read_text(p)
        if text is None:
            continue
        src = ev.rel(p)
        objs = load_structured(p, text)
        if objs:
            for obj in objs:
                kind = "response" if isinstance(obj, dict) and "choices" in obj else kind_hint
                scan_obj(ev, obj, prov, src, kind)
            if "<|channel|>analysis" in text:
                ev.reasoning_seen.append(src)
        else:
            scan_text(ev, text, prov, src, pats)
            if re.search(r"vLLM|LLM engine", text):
                ev.add("engine.name", "vllm", prov, src)
            if "<|channel|>analysis" in text:
                ev.reasoning_seen.append(src)


def collect_requests(ev: Evidence, files: list[Path]) -> list[dict]:
    """Read archived request payloads. Returns parameter dicts (messages removed)."""
    params = []
    for p in files:
        text = read_text(p)
        if text is None:
            continue
        for obj in load_structured(p, text):
            if isinstance(obj, dict) and "messages" in obj:
                params.append({k: v for k, v in obj.items() if k != "messages"})
                scan_obj(ev, obj, "run_time_evidence", ev.rel(p), "request")
    return params


# ---------------------------------------------------------------- model resources

def hf_hub_dirs(explicit: str | None) -> list[Path]:
    dirs = []
    if explicit:
        dirs.append(Path(explicit))
    if os.environ.get("HF_HUB_CACHE"):
        dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen, out = set(), []
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def find_snapshots(repo_id: str, hub_dirs: list[Path]) -> tuple[Path | None, list[dict]]:
    folder = "models--" + repo_id.replace("/", "--")
    for d in hub_dirs:
        repo = d / folder
        snaps = repo / "snapshots"
        if snaps.is_dir():
            out = []
            refs = {}
            if (repo / "refs").is_dir():
                for r in (repo / "refs").iterdir():
                    if r.is_file():
                        refs[r.read_text(encoding="utf-8").strip()] = r.name
            for s in sorted(snaps.iterdir()):
                if s.is_dir():
                    out.append({"commit": s.name, "path": str(s), "mtime": mtime_iso(s), "ref": refs.get(s.name)})
            return repo, out
    return None, []


def _weight_hash(p: Path, hash_large: bool) -> tuple[str, str]:
    try:
        target = Path(os.path.realpath(p))
    except OSError:
        target = p
    if re.fullmatch(r"[0-9a-f]{64}", target.name):
        return target.name, "HF cache blob name (LFS SHA-256)"
    if p.stat().st_size <= 2 * 1024 ** 3 or hash_large:
        return sha256_file(p), "computed SHA-256"
    return "not_computed", "file larger than 2 GiB; rerun with --hash-large-weights"


def collect_model_files(ev: Evidence, model_dir: Path, basis_prefix: str, hash_large: bool,
                        profile: dict) -> None:
    prov = "claude_inferred"
    src = str(model_dir)
    weights = []
    for p in sorted(model_dir.iterdir()) if model_dir.is_dir() else [model_dir]:
        if p.is_file() and p.name.lower().endswith(WEIGHT_SUFFIXES):
            h, how = _weight_hash(p, hash_large)
            weights.append({"file": p.name, "sha256": h, "size": p.stat().st_size, "method": how})
    if weights:
        ev.add("model.weight_files", weights, prov, src, basis=f"{basis_prefix}; weight files listed and hashed")
    if model_dir.is_file() and model_dir.suffix.lower() == ".gguf":
        ev.add("model.format", "gguf", prov, src, basis="weights file has .gguf extension")
        m = re.search(r"(I?Q[0-9]_[A-Z0-9_]+|Q[0-9]_[0-9]|F16|BF16|F32)", model_dir.name, re.IGNORECASE)
        if m:
            ev.add("model.quantization", m.group(1).upper(), prov, src, basis="quantization tag in GGUF file name")
        ev.add("model.converted_or_quantized", True, prov, src, basis="GGUF is a converted format")
        return
    tok = model_dir / "tokenizer.json"
    if not tok.exists():
        tok = model_dir / "tokenizer.model"
    if tok.exists():
        ev.add("model.tokenizer_sha256", sha256_file(tok), prov, str(tok), basis=f"{basis_prefix}; {tok.name}")
    template, tsrc = None, None
    if (model_dir / "chat_template.jinja").exists():
        template, tsrc = (model_dir / "chat_template.jinja").read_text(encoding="utf-8"), "chat_template.jinja"
    elif (model_dir / "tokenizer_config.json").exists():
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        ct = cfg.get("chat_template")
        if isinstance(ct, list):
            ct = json.dumps(ct, sort_keys=True)
        if isinstance(ct, str):
            template, tsrc = ct, "tokenizer_config.json:chat_template"
    if template is not None:
        ev.add("model.chat_template_sha256", sha256_text(template), prov, str(model_dir / tsrc.split(":")[0]),
               basis=f"{basis_prefix}; SHA-256 of {tsrc}")
        ev.add("model.chat_template_source", tsrc, prov, src, basis=basis_prefix)
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        q = (cfg.get("quantization_config") or {}).get("quant_method")
        ev.add("model.quantization", q or "none", prov, str(cfg_path),
               basis=f"{basis_prefix}; config.json quantization_config")
        ev.add("model.format", "original", prov, src,
               basis=f"{basis_prefix}; Hugging Face safetensors snapshot")
        native = profile.get("native_quantization")
        ev.add("model.converted_or_quantized", bool(q) and q != native, prov, str(cfg_path),
               basis="quantization_config compared with the profile's native format")
        dt_ = cfg.get("torch_dtype")
        if dt_:
            ev.info["config_torch_dtype"] = dt_
    readme = model_dir / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="replace")
        m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if m:
            try:
                import yaml
                card = yaml.safe_load(m.group(1)) or {}
            except Exception:
                card = {}
            if card.get("license"):
                ev.add("model.license", str(card["license"]), prov, str(readme), basis="model card front matter")
            if card.get("datasets"):
                ev.info["model_card_datasets"] = card["datasets"]
            if card.get("base_model"):
                ev.info["model_card_base_model"] = card["base_model"]


def collect_model(ev: Evidence, repo_id: str, args: argparse.Namespace, run_start, run_end, profile: dict) -> None:
    prov = "claude_inferred"
    if args.model_path:
        mp = Path(args.model_path)
        collect_model_files(ev, mp, f"researcher-supplied model path {mp}", args.hash_large_weights, profile)
        m = re.search(r"snapshots[\\/]([0-9a-f]{40})", str(mp.resolve()))
        if m:
            ev.add("model.revision", m.group(1), prov, str(mp), basis="commit hash in HF cache snapshot folder name")
    else:
        repo_dir, snaps = find_snapshots(repo_id, hf_hub_dirs(args.hf_cache))
        ev.info["hf_cache_snapshots"] = snaps
        if snaps:
            known = [c["value"] for c in ev.cands.get("model.revision", [])
                     if c["provenance"] != "claude_inferred"]
            chosen, basis = None, None
            for s in snaps:
                if any(values_equal(s["commit"], k) for k in known):
                    chosen, basis = s, "snapshot matches the revision from higher-ranked evidence"
            if chosen is None and len(snaps) == 1:
                chosen, basis = snaps[0], "only snapshot of this repo in the local HF cache"
            if chosen is None and run_end:
                before = [s for s in snaps if parse_iso(s["mtime"]) and parse_iso(s["mtime"]) <= run_end]
                if len(before) == 1:
                    chosen, basis = before[0], "only cached snapshot created before the outputs were written"
            if chosen is None:
                ev.warnings.append(f"Ambiguous: {len(snaps)} snapshots of {repo_id} in the HF cache "
                                   f"({', '.join(s['commit'][:10] + ' ' + s['mtime'] for s in snaps)}). "
                                   "Ask the researcher which commit was used.")
                for s in snaps:
                    ev.add("model.revision", s["commit"], prov, s["path"],
                           basis=f"cached snapshot (modified {s['mtime']}); one of several")
            else:
                ev.add("model.revision", chosen["commit"], prov, chosen["path"], basis=basis)
                collect_model_files(ev, Path(chosen["path"]), f"HF cache snapshot {chosen['commit'][:10]}",
                                    args.hash_large_weights, profile)
    if args.hub:
        _collect_hub(ev, repo_id)


def _collect_hub(ev: Evidence, repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        ev.warnings.append("--hub requested but huggingface_hub is not installed.")
        return
    revs = [c["value"] for c in ev.cands.get("model.revision", [])]
    rev = revs[0] if len(revs) == 1 else None
    try:
        info = HfApi().model_info(repo_id, revision=rev, files_metadata=True)
    except Exception as exc:  # network or auth errors
        ev.warnings.append(f"Hugging Face Hub lookup failed: {exc}")
        return
    basis = f"Hugging Face Hub API at packaging time (revision {rev or 'main'})"
    if rev is None:
        ev.add("model.revision", info.sha, "claude_inferred", "huggingface hub",
               basis="current main on the Hub at packaging time; the run may have used an older commit")
    weights = []
    for s in info.siblings or []:
        if s.rfilename.lower().endswith(WEIGHT_SUFFIXES) and getattr(s, "lfs", None):
            sha = s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
            weights.append({"file": s.rfilename, "sha256": sha, "size": s.size, "method": "Hub LFS metadata"})
    if weights:
        ev.add("model.weight_files", weights, "claude_inferred", "huggingface hub", basis=basis)
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = (card or {}).get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if lic:
        ev.add("model.license", lic, "claude_inferred", "huggingface hub", basis=basis)


def collect_ollama(ev: Evidence) -> None:
    tags = [c["value"] for c in ev.cands.get("model.ollama_tag", [])]
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
            listing = json.loads(r.read())
    except Exception:
        listing = None
    if listing:
        for m in listing.get("models", []):
            if m.get("name") in tags or m.get("model") in tags:
                ev.add("model.ollama_digest", m.get("digest"), "claude_inferred", "ollama /api/tags",
                       basis="digest of the local Ollama model at packaging time")
                det = m.get("details") or {}
                if det.get("quantization_level"):
                    ev.add("model.quantization", det["quantization_level"], "claude_inferred",
                           "ollama /api/tags", basis="Ollama model details")
    ver = run_cmd(["ollama", "-v"])
    if ver:
        m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", ver)
        if m:
            ev.add("engine.libraries.ollama", m.group(1), "claude_inferred", "ollama -v",
                   basis="Ollama installed at packaging time")
    ev.add("model.format", "ollama", "claude_inferred", "engine.name",
           basis="Ollama serves its own converted model blobs")
    ev.add("model.converted_or_quantized", True, "claude_inferred", "engine.name",
           basis="Ollama models are converted (GGUF-based) copies of the original weights")


# ---------------------------------------------------------------- current environment

def current_environment(probe_cloud: bool) -> dict:
    env: dict[str, Any] = {"captured_at": now_iso(), "os": platform.platform(),
                           "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                           "libraries": {}}
    for lib in KEY_LIBS:
        try:
            env["libraries"][lib] = md.version(lib)
        except md.PackageNotFoundError:
            pass
    q = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if q:
        first = q.strip().splitlines()[0].split(",")
        env["accelerator"] = first[0].strip()
        env["accelerator_memory"] = first[1].strip() if len(first) > 1 else None
        env["driver"] = first[2].strip() if len(first) > 2 else None
        full = run_cmd(["nvidia-smi"]) or ""
        m = re.search(r"CUDA Version:\s*([0-9.]+)", full)
        env["cuda"] = m.group(1) if m else None
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        env["accelerator"] = chip.strip() if chip else "Apple Silicon"
    else:
        env["accelerator"] = "cpu"
    if probe_cloud:
        env.update(_probe_cloud())
    return env


def _http(url: str, headers: dict, method: str = "GET", timeout: float = 1.0) -> str | None:
    try:
        req = urllib.request.Request(url, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode().strip()
    except Exception:
        return None


def _probe_cloud() -> dict:
    tok = _http("http://169.254.169.254/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "60"}, "PUT")
    if tok:
        it = _http("http://169.254.169.254/latest/meta-data/instance-type", {"X-aws-ec2-metadata-token": tok})
        if it:
            return {"cloud_provider": "AWS", "instance_type": it}
    mt = _http("http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
               {"Metadata-Flavor": "Google"})
    if mt:
        return {"cloud_provider": "GCP", "instance_type": mt.rsplit("/", 1)[-1]}
    az = _http("http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text",
               {"Metadata": "true"})
    if az:
        return {"cloud_provider": "Azure", "instance_type": az}
    return {}


def add_current_env(ev: Evidence, env: dict) -> None:
    """Use the current environment only to fill gaps (last resort).

    Where run-time or researcher evidence exists, differences are recorded in
    info.current_vs_run instead of being reported as conflicts.
    """
    if ev.run_mode == "hosted_api":
        ev.info["current_vs_run"] = {}
        return  # the provider's stack is not the packaging machine
    prov, basis = "claude_inferred", "current environment at packaging time; may differ from the run"
    src = "env/current_environment.yaml"
    pairs = [("engine.os", env.get("os")), ("engine.python", env.get("python")), ("hardware.cpu", env.get("cpu"))]
    pairs += [(f"engine.libraries.{lib}", ver) for lib, ver in env.get("libraries", {}).items()]
    if ev.run_mode == "self_run":
        pairs += [("hardware.accelerator", env.get("accelerator")),
                  ("hardware.accelerator_memory", env.get("accelerator_memory")),
                  ("engine.driver", env.get("driver")), ("engine.cuda", env.get("cuda")),
                  ("hardware.cloud_provider", env.get("cloud_provider")),
                  ("hardware.instance_type", env.get("instance_type"))]
    diffs = {}
    for path, val in pairs:
        if val in (None, ""):
            continue
        existing = ev.cands.get(path)
        if existing:
            if not any(values_equal(c["value"], val) for c in existing):
                diffs[path] = {"now": val, "run": existing[0]["value"]}
            continue
        b = "cloud metadata endpoint" if path.startswith(("hardware.cloud", "hardware.instance")) else basis
        ev.add(path, val, prov, src, basis=b)
    ev.info["current_vs_run"] = diffs


def add_if_absent(ev: Evidence, path: str, value: Any, prov: str, source: str, basis: str) -> None:
    if not ev.cands.get(path):
        ev.add(path, value, prov, source, basis=basis)


# ---------------------------------------------------------------- derived fields

def vendor_of(acc: str) -> str | None:
    a = str(acc).lower()
    if re.search(r"nvidia|geforce|tesla|rtx|\ba100\b|\bh100\b|\bl4\b|\bt4\b", a):
        return "nvidia"
    if "apple" in a:
        return "apple"
    if re.search(r"\bamd\b|instinct|radeon|\bmi[0-9]{3}", a):
        return "amd"
    if a == "cpu":
        return "cpu"
    return None


def derive(ev: Evidence) -> None:
    for c in list(ev.cands.get("hardware.accelerator", [])):
        v = vendor_of(c["value"])
        if v:
            ev.add("hardware.accelerator_vendor", v, c["provenance"], c["source"], basis=c.get("basis"),
                   note="derived from hardware.accelerator")
    eng, _ = resolve_candidates("engine.name", ev.cands.get("engine.name", []))
    lib = ENGINE_LIB.get(str(eng.get("value")))
    if lib:
        for c in list(ev.cands.get(f"engine.libraries.{lib}", [])):
            ev.add("engine.version", c["value"], c["provenance"], c["source"], basis=c.get("basis"),
                   note=f"derived from the {lib} version")
    for c in list(ev.cands.get("model.repo_id_declared", [])):
        ev.add("model.repo_id", c["value"], c["provenance"], c["source"])
    ev.cands.pop("model.repo_id_declared", None)


# ---------------------------------------------------------------- data and outputs

def collect_data_outputs(ev: Evidence, sel: dict, discover_cols: dict, profile: dict, raw_files: list[Path]) -> None:
    roles = sel.get("roles", {})
    prov = "claude_inferred"
    if "data" in roles:
        info = discover_cols.get(roles["data"]["path"]) or {}
        add_if_absent(ev, "data.id_column", info.get("id_column"), prov, roles["data"]["path"], "column-name heuristic")
        add_if_absent(ev, "data.text_column", info.get("text_column"), prov, roles["data"]["path"], "column-name heuristic")
    if "labels" in roles:
        lp = _abs(ev.project, roles["labels"]["path"])
        try:
            rows = read_records(lp)
        except BaseException:
            rows = []
            ev.warnings.append(f"Could not read labels file {roles['labels']['path']}.")
        if rows:
            cols = list(rows[0].keys())
            idc, lc = guess_column(cols, ID_COLUMN_NAMES), guess_column(cols, LABEL_COLUMN_NAMES)
            add_if_absent(ev, "outputs.id_column", idc, prov, roles["labels"]["path"], "column-name heuristic")
            add_if_absent(ev, "outputs.label_column", lc, prov, roles["labels"]["path"], "column-name heuristic")
            if lc:
                vals = [r.get(lc) for r in rows]
                multi = any(isinstance(v, list) for v in vals) or \
                    sum(1 for v in vals if isinstance(v, str) and re.search(r"[;|]", v)) > len(vals) / 2
                add_if_absent(ev, "study.task_type", "multi_label" if multi else "single_label", prov,
                       roles["labels"]["path"], "shape of archived label values")
                flat = set()
                for v in vals:
                    if isinstance(v, list):
                        flat.update(str(x) for x in v)
                    elif v is not None:
                        flat.update(re.split(r"\s*[;|]\s*", str(v)) if multi else [str(v)])
                add_if_absent(ev, "study.label_set", sorted(flat), prov, roles["labels"]["path"],
                       "unique values observed in archived labels (unused labels would be missing)")
    rule = None
    for p in raw_files[:3]:
        text = read_text(p) or ""
        if '"predictions"' in text:
            rule = {"type": "json_predictions", "list_key": "predictions", "id_key": "id", "label_key": "label"}
            break
    if rule:
        add_if_absent(ev, "outputs.label_extraction_rule", rule, prov, ev.rel(raw_files[0]),
                      "raw outputs contain a JSON predictions list")
    if profile.get("default_extraction_rule"):
        add_if_absent(ev, "outputs.label_extraction_rule", profile["default_extraction_rule"], prov,
                      f"profile {profile['name']}", "profile default extraction rule")


def _abs(project: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else project / p


# ---------------------------------------------------------------- main collection

def collect(project: Path, args: argparse.Namespace) -> dict:
    project = project.resolve()
    st = state_dir(project)
    sel = load_yaml(st / "selection.yaml", {})
    if not sel.get("confirmed_by_researcher"):
        raise SystemExit("ERROR: file selection not confirmed. Run discover.py, ask the researcher, "
                         "then confirm.py select.")
    disc = load_yaml(st / "candidates.yaml", {})
    roles = sel.get("roles", {})

    def files(role: str) -> list[Path]:
        entry = roles.get(role)
        if not entry:
            return []
        entries = entry if isinstance(entry, list) else [entry]
        out = []
        for e in entries:
            out.extend(iter_files(_abs(project, e["path"])))
        return out

    ev = Evidence(project, args.run_mode)
    ev.add("study.run_mode", args.run_mode, "researcher_provided", "researcher input")
    ev.add("model.repo_id", args.model_id, "researcher_provided", "researcher input")
    if args.revision:
        ev.add("model.revision", args.revision, "researcher_provided", "researcher input")

    profile, pattern = match_profile(args.model_id)
    ev.info["profile"] = {"name": profile["name"], "matched_pattern": pattern}
    for fname, spec in (profile.get("fields") or {}).items():
        for k in spec.get("keys") or []:
            ev.profile_keys[k.lower()] = f"model_profile.{fname}"

    # Run window from output files (refined by run-time timestamps below).
    out_files = files("labels") + files("raw_outputs") + files("logs") + files("requests")
    mt = [dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc) for p in out_files]

    collect_runtime(ev, files("logs") + files("raw_outputs"), "meta", profile)
    req_params = collect_requests(ev, files("requests"))
    rt_times = sorted(parse_iso(t) for t, prov, _ in ev.timestamps if prov == "run_time_evidence")
    run_start = rt_times[0] if rt_times else (min(mt) if mt else None)
    run_end = rt_times[-1] if rt_times else (max(mt) if mt else None)
    ev.info["run_window"] = {"start": run_start.isoformat() if run_start else None,
                             "end": run_end.isoformat() if run_end else None,
                             "from": "run-time timestamps" if rt_times else "file modified times"}

    collect_env_package(ev, files("env_package"), run_end)
    from common import flatten, is_blank
    for pf in files("practices"):
        for key, val in flatten(load_yaml(pf, {}) or {}).items():
            if not is_blank(val):
                ev.add(key, val, "researcher_provided", ev.rel(pf), note="best-practice record")
    collect_code(ev, files("code"), files("config"), files("prompt_files"), run_end, profile)

    # Hosted API specifics.
    if args.run_mode == "hosted_api":
        if req_params:
            first = req_params[0]
            ev.add("hosted_api.parameters", first, "run_time_evidence", "archived request payloads",
                   note=None if all(values_equal(first, r) for r in req_params) else
                   "parameters differ between requests; see outputs/requests")
            for key, path in (("temperature", "decoding.temperature"), ("top_p", "decoding.top_p")):
                if all(key not in r for r in req_params):
                    ev.add(path, "not_set_provider_default", "run_time_evidence", "archived request payloads",
                           note=f"{key} absent from every request; the provider default applied")
        if ev.fingerprints:
            ev.add("hosted_api.system_fingerprints", sorted(ev.fingerprints), "run_time_evidence",
                   "archived raw responses")
        resp = [p for p in files("raw_outputs") + files("logs") if "choices" in (read_text(p) or "")]
        if resp:
            ev.add("hosted_api.raw_responses", [ev.rel(p) for p in resp], "run_time_evidence", "archived files")
        if rt_times:
            ev.add("hosted_api.call_timestamps", {"first": rt_times[0].isoformat(), "last": rt_times[-1].isoformat(),
                                                  "count": len(rt_times)}, "run_time_evidence",
                   "timestamps in archived metadata")
        elif mt:
            ev.add("hosted_api.call_timestamps", {"first": min(mt).isoformat(), "last": max(mt).isoformat()},
                   "claude_inferred", "file modified times", basis="no call timestamps saved; output file times")
        for c in list(ev.cands.get("hosted_api.base_url", [])):
            host = re.sub(r"^https?://", "", str(c["value"])).split("/")[0].lower()
            for dom, name in PROVIDERS.items():
                if dom in host:
                    ev.add("hosted_api.provider", name, "claude_inferred", c["source"],
                           basis=f"derived from base_url host {host}")
            if host.startswith(("127.0.0.1", "localhost")):
                ev.warnings.append("hosted_api.base_url is a loopback address. If the researcher served the "
                                   "model themselves, the run is self_run, not hosted_api.")
        ev.add("model.repo_id", args.model_id, "researcher_provided", "researcher input")

    # Run date.
    if rt_times:
        ev.add("study.run_date", run_start.date().isoformat() if run_start.date() == run_end.date()
               else f"{run_start.date()}..{run_end.date()}", "run_time_evidence", "timestamps in archived metadata")
    elif mt:
        ev.add("study.run_date", f"{min(mt).date()}..{max(mt).date()}", "claude_inferred", "file modified times",
               basis="no run timestamps saved; modified times of archived outputs")

    # Model openness and profile values.
    ev.add("model_profile.name", profile["name"], "claude_inferred", f"profiles/{profile['name']}.yaml",
           basis=f"repo id matches profile pattern {pattern}" if pattern else "no profile matched")
    ev.add("model.openness", profile.get("openness"), "claude_inferred", f"profiles/{profile['name']}.yaml",
           basis="model family profile; confirm against the model card")
    ev.add("model.license", profile.get("license"), "claude_inferred", f"profiles/{profile['name']}.yaml",
           basis="model family profile")
    ev.add("model.license_restrictions", profile.get("license_restrictions"), "claude_inferred",
           f"profiles/{profile['name']}.yaml", basis="model family profile")
    ev.add("model.training_data_disclosed", profile.get("training_data_disclosed"), "claude_inferred",
           f"profiles/{profile['name']}.yaml", basis="model family profile")
    for key, val in profile_hints(profile, args.model_id).items():
        ev.add(key, val, "claude_inferred", f"profiles/{profile['name']}.yaml",
               basis="profile reference value from the model card; verify")
    for fname, spec in (profile.get("fields") or {}).items():
        if "default" in spec:
            add_if_absent(ev, f"model_profile.{fname}", spec["default"], "claude_inferred",
                          f"profiles/{profile['name']}.yaml", spec.get("default_basis", "profile default"))
    if "reasoning_trace_archived" in (profile.get("fields") or {}):
        if ev.reasoning_seen:
            ev.add("model_profile.reasoning_trace_archived", True, "run_time_evidence", ev.reasoning_seen[0])
        elif files("raw_outputs"):
            ev.add("model_profile.reasoning_trace_archived", False, "claude_inferred", "raw outputs",
                   basis="no reasoning fields or analysis channel found in raw outputs")

    collect_data_outputs(ev, sel, disc.get("columns", {}), profile, files("raw_outputs"))

    # Lowest tier: model cache and current environment.
    if args.run_mode == "self_run":
        collect_model(ev, args.model_id, args, run_start, run_end, profile)
        eng, _ = resolve_candidates("engine.name", ev.cands.get("engine.name", []))
        if eng.get("value") == "ollama" or ev.cands.get("model.ollama_tag"):
            collect_ollama(ev)
    cur = current_environment(args.probe_cloud)
    add_current_env(ev, cur)
    derive(ev)
    ev.info["current_environment"] = cur

    result = {
        "generated_at": now_iso(),
        "project": str(project),
        "inputs": {"model_id": args.model_id, "revision": args.revision, "run_mode": args.run_mode,
                   "hf_cache": args.hf_cache, "model_path": args.model_path},
        "candidates": ev.cands,
        "warnings": ev.warnings,
        "info": ev.info,
    }
    save_yaml(st / "evidence.yaml", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--run-mode", choices=["self_run", "hosted_api"], required=True)
    ap.add_argument("--hf-cache")
    ap.add_argument("--model-path")
    ap.add_argument("--hub", action="store_true", help="Query the Hugging Face Hub (network).")
    ap.add_argument("--probe-cloud", action="store_true", help="Read cloud instance metadata (network).")
    ap.add_argument("--hash-large-weights", action="store_true")
    args = ap.parse_args()
    res = collect(args.project, args)
    print(f"Evidence saved: {state_dir(args.project) / 'evidence.yaml'}")
    print(f"Fields with candidates: {len(res['candidates'])}")
    print(f"Run window: {res['info']['run_window']}")
    for w in res["warnings"]:
        print("WARNING: " + w)
    from confirm import cmd_pending
    cmd_pending(args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
