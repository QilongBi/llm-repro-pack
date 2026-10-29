from __future__ import annotations

from common import load_yaml, resolve_candidates, state_dir
from conftest import do_collect, do_discover, do_select
from fixtures import COMMIT, COMMIT2, OLMO, make_hf_cache, make_hosted, make_selfrun, selfrun_selection, hosted_selection
import init_project


def _best(ev, key):
    return resolve_candidates(key, ev["candidates"].get(key, []))


def test_init_creates_layout_without_overwriting(tmp_path):
    p = tmp_path / "proj"
    created = init_project.init(p)
    assert {"data/", "outputs/", "logs/", "env/", "prompt.yaml", "config.yaml"} <= set(created)
    (p / "prompt.yaml").write_text("mine", encoding="utf-8")
    init_project.init(p)
    assert (p / "prompt.yaml").read_text(encoding="utf-8") == "mine"


def test_discover_lists_candidates_and_marks_specified(tmp_path):
    p = make_selfrun(tmp_path)
    res = do_discover(p, labels=["outputs/labels.csv"])
    c = res["candidates"]
    assert c["labels"][0] == {**c["labels"][0], "path": "outputs/labels.csv", "origin": "researcher_specified"}
    assert any(x["path"] == "prompt.yaml" for x in c["prompt_files"])
    assert any(x["path"] == "config.yaml" for x in c["config"])
    assert any(x["path"] == "data/items.csv" for x in c["data"])
    assert any(x["path"] == "logs/run.log" for x in c["logs"])
    assert any(x["path"] == "env/requirements.lock" for x in c["env_package"])
    assert all("mtime" in x for x in c["data"])
    assert res["columns"]["data/items.csv"]["text_column"] == "text"


def test_selection_records_origin(tmp_path):
    p = make_selfrun(tmp_path)
    do_discover(p, labels=["outputs/labels.csv"])
    do_select(p, selfrun_selection(p))
    sel = load_yaml(state_dir(p) / "selection.yaml")
    assert sel["roles"]["labels"]["origin"] == "researcher_specified"
    assert sel["roles"]["data"]["origin"] == "auto_discovered"


def test_collect_requires_confirmed_selection(tmp_path):
    import pytest
    p = make_selfrun(tmp_path)
    do_discover(p)
    with pytest.raises(SystemExit):
        do_collect(p, OLMO)


def test_evidence_hierarchy_and_provenance(tmp_path):
    p = make_selfrun(tmp_path)
    hf = make_hf_cache(tmp_path / "hf", OLMO, [COMMIT])
    do_discover(p)
    do_select(p, selfrun_selection(p))
    ev = do_collect(p, OLMO, hf)
    temp, conflict = _best(ev, "decoding.temperature")
    assert conflict is None and temp["value"] == 0.0 and temp["provenance"] == "run_time_evidence"
    ver, _ = _best(ev, "engine.version")
    assert ver["value"] == "0.6.3" and ver["provenance"] == "researcher_provided"
    acc, _ = _best(ev, "hardware.accelerator")
    assert acc["value"] == "NVIDIA A100-SXM4-80GB" and acc["provenance"] == "researcher_provided"
    assert _best(ev, "engine.cuda")[0]["value"] == "12.4"
    assert _best(ev, "engine.driver")[0]["value"] == "550.54.15"
    assert _best(ev, "hardware.accelerator_vendor")[0]["value"] == "nvidia"
    rev, conflict = _best(ev, "model.revision")
    assert conflict is None and rev["value"] == COMMIT
    w, _ = _best(ev, "model.weight_files")
    assert w["provenance"] == "claude_inferred" and w["basis"]
    assert _best(ev, "model.openness")[0]["value"] == "fully_open"
    assert _best(ev, "engine.name")[0]["value"] == "vllm"
    assert _best(ev, "study.label_set")[0]["value"] == ["positive", "negative", "neutral"]
    # The packaging machine (no GPU) must not create conflicts with run-time evidence.
    assert not any(c for c in ev["candidates"].get("hardware.accelerator", []) if c["value"] == "cpu")


def test_conflict_is_reported_not_silently_resolved(tmp_path):
    p = make_selfrun(tmp_path, conflict=True)
    do_discover(p)
    do_select(p, selfrun_selection(p))
    ev = do_collect(p, OLMO)
    best, conflict = _best(ev, "decoding.temperature")
    assert conflict is not None
    assert best["conflict"] is True
    values = {c["value"] for c in conflict["candidates"]}
    assert values == {0.0, 0.7}


def test_posthoc_environment_export_is_flagged(tmp_path):
    p = make_selfrun(tmp_path, posthoc=True)
    do_discover(p)
    do_select(p, selfrun_selection(p))
    ev = do_collect(p, OLMO)
    assert any("Post-hoc export" in w for w in ev["warnings"])
    torch = ev["candidates"]["engine.libraries.torch"][0]
    assert "post-hoc export" in torch["note"]


def test_ambiguous_hf_snapshots_are_reported(tmp_path):
    p = make_selfrun(tmp_path)
    # Remove the revision from config and logs so only the cache could tell.
    cfg = (p / "config.yaml").read_text(encoding="utf-8").replace(f'revision: "{COMMIT}"', 'revision: ""')
    (p / "config.yaml").write_text(cfg, encoding="utf-8")
    log = (p / "logs" / "run.log").read_text(encoding="utf-8").replace(f"revision={COMMIT}, ", "")
    (p / "logs" / "run.log").write_text(log, encoding="utf-8")
    hf = make_hf_cache(tmp_path / "hf", OLMO, [COMMIT, COMMIT2])
    import os
    for c in (COMMIT, COMMIT2):  # both snapshots older than the outputs
        os.utime(hf / ("models--" + OLMO.replace("/", "--")) / "snapshots" / c, (1, 1))
    do_discover(p)
    do_select(p, selfrun_selection(p))
    ev = do_collect(p, OLMO, hf)
    assert any("Ambiguous" in w for w in ev["warnings"])
    _, conflict = _best(ev, "model.revision")
    assert conflict is not None


def test_hosted_api_run_is_refused(tmp_path):
    import pytest
    p = make_hosted(tmp_path)
    do_discover(p)
    do_select(p, hosted_selection())
    with pytest.raises(SystemExit) as exc:
        do_collect(p, "openai/gpt-oss-20b")
    assert "hosted API" in str(exc.value) and "Groq" in str(exc.value)
