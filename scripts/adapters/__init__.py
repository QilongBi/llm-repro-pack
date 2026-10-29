"""Inference engine adapters.

Each adapter lives in <name>_adapter.py and defines ADAPTER, a subclass of base.Adapter.
To add an engine, drop a new <name>_adapter.py here, or point LLMRP_ADAPTER_PATH at a
directory that contains one.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

BUILTIN = ("transformers", "vllm", "ollama", "mock")


def get_adapter(name: str):
    mod_name = f"{name.replace('.', '_').replace('-', '_')}_adapter"
    try:
        mod = importlib.import_module(f"adapters.{mod_name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"adapters.{mod_name}":
            raise
        mod = None
        for d in filter(None, os.environ.get("LLMRP_ADAPTER_PATH", "").split(os.pathsep)):
            p = Path(d) / f"{mod_name}.py"
            if p.exists():
                spec = importlib.util.spec_from_file_location(mod_name, p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                break
        if mod is None:
            raise SystemExit(f"ERROR: no adapter named '{name}'. Built-in: {', '.join(BUILTIN)}")
    return mod.ADAPTER
