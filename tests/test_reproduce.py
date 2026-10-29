"""Reproduce mode runs the archived scripts inside the package, with the mock adapter."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from common import canonical_json, field_value, load_yaml, read_jsonl, read_records, sha256_text
from conftest import confirm_all, do_collect, do_discover, do_package, do_select
from fixtures import COMMIT, GPTOSS, OLMO, make_hf_cache, make_hosted, make_selfrun, hosted_selection, selfrun_selection


def _selfrun_pkg(tmp_path):
    p = make_selfrun(tmp_path)
    hf = make_hf_cache(tmp_path / "hf", OLMO, [COMMIT])
    do_discover(p)
    do_select(p, selfrun_selection(p))
    do_collect(p, OLMO, "self_run", hf)
    confirm_all(p)
    return do_package(p, tmp_path / "pkg")["package"]


def _mock_table(pkg, path, answer_for):
    m = load_yaml(pkg / "manifest.yaml")
    table = {}
    for r in read_jsonl(pkg / field_value(m, "prompt.rendered_inputs_file")):
        table[sha256_text(canonical_json(r["messages"]))] = answer_for(r)
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def _run(pkg, *args, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    return subprocess.run([sys.executable, str(pkg / "scripts" / "reproduce.py"), "--package", str(pkg), *args],
                          capture_output=True, text=True, env=env)


def test_reproduce_exact_match(tmp_path):
    pkg = _selfrun_pkg(tmp_path)
    labels = {r["id"]: r["label"] for r in read_records(pkg / "outputs" / "labels.csv")}
    table = _mock_table(pkg, tmp_path / "mock.json", lambda r: labels[r["item_ids"][0]])
    r = _run(pkg, "--engine", "mock", env_extra={"LLMRP_MOCK_RESPONSES": str(table)})
    assert r.returncode == 0, r.stderr + r.stdout
    metrics = json.loads((pkg / "reproduction" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["exact_match_rate"] == 1.0 and metrics["n_compared"] == 12
    report = (pkg / "reproduction" / "REPRODUCTION_REPORT.md").read_text(encoding="utf-8")
    assert "All compared labels match" in report
    # The package still verifies after a reproduction run.
    assert _run(pkg, "--verify-only").returncode == 0


def test_reproduce_partial_agreement_and_sample(tmp_path):
    pkg = _selfrun_pkg(tmp_path)
    table = _mock_table(pkg, tmp_path / "mock.json", lambda r: "positive")
    r = _run(pkg, "--engine", "mock", "--sample", "5", env_extra={"LLMRP_MOCK_RESPONSES": str(table)})
    assert r.returncode == 0, r.stderr
    metrics = json.loads((pkg / "reproduction" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_compared"] == 5
    assert metrics["exact_match_rate"] < 1.0


def test_tampered_prompt_stops_reproduction(tmp_path):
    pkg = _selfrun_pkg(tmp_path)
    (pkg / "prompts" / "prompt.yaml").write_text("system: changed\nuser_template: x\n", encoding="utf-8")
    r = _run(pkg, "--engine", "mock")
    assert r.returncode == 3
    assert "prompt" in r.stderr


def test_chat_template_mismatch_stops_run(tmp_path):
    pkg = _selfrun_pkg(tmp_path)
    r = _run(pkg, "--engine", "mock", env_extra={"LLMRP_MOCK_TEMPLATE_SHA": "0" * 64})
    assert r.returncode == 3
    assert "chat template" in r.stderr


def test_rebuild_dry_run_prints_commands(tmp_path):
    pkg = _selfrun_pkg(tmp_path)
    r = _run(pkg, "--rebuild", "--dry-run")
    assert r.returncode == 0
    assert "REBUILD:" in r.stdout and "requirements.lock" in r.stdout


def test_cross_mode_reproduction_of_hosted_package(tmp_path):
    p = make_hosted(tmp_path)
    do_discover(p)
    do_select(p, hosted_selection())
    do_collect(p, GPTOSS, "hosted_api")
    confirm_all(p)
    pkg = do_package(p, tmp_path / "pkg")["package"]
    output = (p / "runs" / "20260901T120000000000Z" / "model_output.txt").read_text(encoding="utf-8")
    table = _mock_table(pkg, tmp_path / "mock.json", lambda r: output)
    r = _run(pkg, "--engine", "mock", env_extra={"LLMRP_MOCK_RESPONSES": str(table)})
    assert r.returncode == 0, r.stderr + r.stdout
    metrics = json.loads((pkg / "reproduction" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_compared"] == 6 and metrics["exact_match_rate"] == 1.0
    report = (pkg / "reproduction" / "REPRODUCTION_REPORT.md").read_text(encoding="utf-8")
    assert "cross-mode" in report.lower() or "Cross-mode" in report
