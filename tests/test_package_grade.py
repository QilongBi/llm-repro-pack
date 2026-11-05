from __future__ import annotations

import pytest

from common import field_value, get_path, load_yaml
from conftest import confirm_all, do_collect, do_discover, do_package, do_select
from fixtures import COMMIT, GPTOSS, OLMO, make_hf_cache, make_selfrun, selfrun_selection


def _prepare_selfrun(tmp_path, model_id=OLMO, extra_config="", quant=None):
    p = make_selfrun(tmp_path, model_id=model_id, extra_config=extra_config)
    hf = make_hf_cache(tmp_path / "hf", model_id, [COMMIT], quant=quant)
    do_discover(p, labels=["outputs/labels.csv"])
    do_select(p, selfrun_selection(p) + ["gold=gold.csv"])
    do_collect(p, model_id, hf)
    return p


def test_unconfirmed_run_is_not_packaged(tmp_path):
    p = _prepare_selfrun(tmp_path)
    res = do_package(p, tmp_path / "pkg")
    assert res["grade"]["grade"] is None and res["grade"]["eligible"] is False
    assert res["package"] is None and not (tmp_path / "pkg").exists()
    reasons = {i["field"]: i["reason"] for i in res["grade"]["critical_missing"]}
    assert reasons["model.weight_files"] == "inferred, not confirmed"
    ready = load_yaml(p / ".llmrp" / "readiness.yaml")
    assert ready["eligible"] is False


def test_olmo_confirmed_reaches_grade_a(tmp_path):
    p = _prepare_selfrun(tmp_path)
    confirm_all(p, provided={"model.training_code_commit": "allenai/OLMo@0123456789abcdef"})
    res = do_package(p, tmp_path / "pkg")
    g = res["grade"]
    assert g["grade"] == "A", g
    assert res["selfcheck"]["ok"], res["selfcheck"]["errors"]
    pkg = res["package"]
    m = load_yaml(pkg / "manifest.yaml")
    assert m["grade"]["grade"] == "A"
    assert field_value(m, "validation.accuracy") == 1.0
    assert field_value(m, "prompt.rendered_inputs_complete") is True
    assert field_value(m, "model.revision") == COMMIT
    assert get_path(m, "files.labels")[0]["origin"] == "researcher_specified"
    for f in ("README.md", "REPRODUCIBILITY.md", "checksums.sha256", "env/REBUILD.md",
              "prompts/rendered/rendered_inputs.jsonl", "scripts/reproduce.sh", "scripts/adapters/vllm_adapter.py"):
        assert (pkg / f).exists(), f
    assert not list(pkg.rglob("*.safetensors"))
    text = (pkg / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "## Grade: A" in text and "same hardware class" in text


def test_olmo_without_training_code_commit_is_b(tmp_path):
    p = _prepare_selfrun(tmp_path)
    confirm_all(p)
    res = do_package(p, tmp_path / "pkg")
    assert res["grade"]["grade"] == "B"
    assert any(i["field"] == "model.training_code_commit" for i in res["grade"]["a_only_missing"])
    assert any("training_code_commit" in s for s in res["grade"]["how_to_improve"])


def test_gpt_oss_self_run_is_capped_at_b(tmp_path):
    p = _prepare_selfrun(tmp_path, model_id=GPTOSS, quant="mxfp4",
                         extra_config="model_profile:\n  reasoning_effort: low\n")
    (p / "run.py").write_text((p / "run.py").read_text(encoding="utf-8") + "\n# vllm handles harmony\n",
                              encoding="utf-8")
    confirm_all(p)
    res = do_package(p, tmp_path / "pkg")
    m = load_yaml(res["package"] / "manifest.yaml")
    assert field_value(m, "model.openness") == "open_weight"
    assert field_value(m, "model_profile.reasoning_effort") == "low"
    assert get_path(m, "model_profile.reasoning_effort")["provenance"] == "code_declared"
    assert field_value(m, "model.quantization") == "mxfp4"
    assert field_value(m, "model.converted_or_quantized") is False
    assert res["grade"]["grade"] == "B", res["grade"]


def test_gpt_oss_missing_profile_field_is_not_packaged(tmp_path):
    p = _prepare_selfrun(tmp_path, model_id=GPTOSS)
    res = do_package(p, tmp_path / "pkg")
    missing = {i["field"] for i in res["grade"]["critical_missing"]}
    assert "model_profile.reasoning_effort" in missing
    assert res["grade"]["eligible"] is False



def test_only_outputs_is_not_packaged(tmp_path):
    p = make_selfrun(tmp_path)
    do_discover(p)
    do_select(p, ["labels=outputs/labels.csv"])
    do_collect(p, OLMO)
    confirm_all(p)
    res = do_package(p, tmp_path / "pkg", data_shareable=False)
    assert res["grade"]["eligible"] is False
    missing = {i["field"] for i in res["grade"]["critical_missing"]}
    assert {"prompt.files_sha256", "data.sha256"} <= missing


def test_package_inside_project_refused(tmp_path):
    p = _prepare_selfrun(tmp_path)
    with pytest.raises(SystemExit):
        do_package(p, p / "pkg")


def test_archive_weights_requires_permissive_license(tmp_path):
    p = make_selfrun(tmp_path, model_id="meta-llama/Llama-3.1-8B-Instruct")
    hf = make_hf_cache(tmp_path / "hf", "meta-llama/Llama-3.1-8B-Instruct", [COMMIT], license_="llama3.1")
    do_discover(p)
    do_select(p, selfrun_selection(p))
    do_collect(p, "meta-llama/Llama-3.1-8B-Instruct", hf)
    with pytest.raises(SystemExit):
        do_package(p, tmp_path / "pkg", archive_weights=True)


def test_engine_without_adapter_gets_no_wrong_command(tmp_path):
    p = _prepare_selfrun(tmp_path)
    confirm_all(p, provided={"engine.name": "mlx", "model.training_code_commit": "allenai/OLMo@0123456789abcdef"})
    res = do_package(p, tmp_path / "pkg")
    assert res["grade"]["eligible"]
    readme = (res["package"] / "README.md").read_text(encoding="utf-8")
    assert "--engine transformers" not in readme
    assert "No built-in adapter for the recorded engine 'mlx'" in readme
    rep = (res["package"] / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "no built-in reproduction adapter" in rep
    assert "MXFP4" not in rep  # OLMo package: no gpt-oss-specific wording


def test_supported_engine_gets_its_command(tmp_path):
    p = _prepare_selfrun(tmp_path)
    confirm_all(p, provided={"model.training_code_commit": "allenai/OLMo@0123456789abcdef"})
    res = do_package(p, tmp_path / "pkg")
    readme = (res["package"] / "README.md").read_text(encoding="utf-8")
    assert "python scripts/reproduce.py --package . --engine vllm" in readme
    assert "no built-in reproduction adapter" not in (res["package"] / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
