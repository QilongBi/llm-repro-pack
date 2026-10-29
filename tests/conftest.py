from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import collect_env  # noqa: E402
import confirm  # noqa: E402
import discover  # noqa: E402
import package  # noqa: E402
from common import save_yaml, state_dir  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_hf(monkeypatch, tmp_path):
    empty = tmp_path / "_empty_hf"
    empty.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(empty))
    monkeypatch.setenv("HF_HOME", str(empty))


def do_discover(project: Path, **specified) -> dict:
    spec = {r: specified.get(r, []) for r in discover.ALL_ROLES}
    res = discover.discover(project, spec)
    save_yaml(state_dir(project) / "candidates.yaml", res)
    return res


def do_select(project: Path, pairs: list[str]) -> None:
    confirm.cmd_select(project, pairs)


def do_collect(project: Path, model_id: str, hf_cache: Path | None = None,
               revision: str | None = None, model_path: str | None = None) -> dict:
    args = argparse.Namespace(model_id=model_id, revision=revision,
                              hf_cache=str(hf_cache) if hf_cache else None, model_path=model_path,
                              hub=False, probe_cloud=False, hash_large_weights=False)
    return collect_env.collect(project, args)


def confirm_all(project: Path, provided: dict | None = None) -> None:
    """Simulate a researcher: resolve conflicts with the top candidate, confirm inferred values."""
    for key, value in (provided or {}).items():
        confirm.cmd_field(project, [f"{key}={value}"], "researcher_provided", None, None)
    inferred, conflicts = confirm.pending(project)
    for c in conflicts:
        import json
        confirm.cmd_field(project, [f"{c['field']}={json.dumps(c['candidates'][0]['value'])}"],
                          "researcher_confirmed", None, None)
    for key, _ in inferred:
        confirm.cmd_field(project, [key], "researcher_confirmed", None, None)


def do_package(project: Path, out: Path, data_shareable: bool = True, **kw) -> dict:
    args = argparse.Namespace(project=project, out=out, title=None, data_shareable=data_shareable,
                              data_access=kw.get("data_access"), gold_shareable=kw.get("gold_shareable", False),
                              gold_id_column=None, gold_label_column=None, render_chat_template=False,
                              max_rendered=kw.get("max_rendered", 1000), archive_weights=kw.get("archive_weights", False),
                              overwrite=True)
    return package.run(args)
