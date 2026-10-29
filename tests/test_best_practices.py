"""Checks for the Törnberg (2024) best-practice additions."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from common import canonical_json, field_value, load_yaml, read_jsonl, read_records, sha256_text
from conftest import confirm_all, do_collect, do_discover, do_package, do_select
import confirm
from fixtures import (COMMIT, GOLD, GPTOSS, OLMO, TEXTS, _w, make_hf_cache, make_hosted, make_selfrun,
                      hosted_selection, selfrun_selection)
from prompt_lint import lint
from validate import run_validation


def _status(bp, requirement):
    return next(i["status"] for i in bp["items"] if i["requirement"].startswith(requirement))


def _gold_with_coders(p):
    # coder2 disagrees on two items; the LLM labels equal GOLD.
    rows = ["id,label,coder1,coder2,lang"]
    for i, g in enumerate(GOLD):
        c2 = "neutral" if i in (0, 1) else g
        rows.append(f"t{i:02d},{g},{g},{c2},{'en' if i < 6 else 'nl'}")
    _w(p / "gold_coders.csv", "\n".join(rows) + "\n")


def _selfrun(tmp_path, extra=()):
    p = make_selfrun(tmp_path)
    _gold_with_coders(p)
    hf = make_hf_cache(tmp_path / "hf", OLMO, [COMMIT])
    do_discover(p)
    do_select(p, selfrun_selection(p) + list(extra))
    do_collect(p, OLMO, "self_run", hf)
    return p


def test_validation_metrics_icr_subsets_and_errors(tmp_path):
    p = _selfrun(tmp_path, ["gold=gold_coders.csv"])
    import argparse
    import package
    args = argparse.Namespace(project=p, out=tmp_path / "pkg", title=None, data_shareable=True, data_access=None,
                              gold_shareable=True, gold_id_column="id", gold_label_column="label",
                              gold_coder_columns="coder1,coder2", subset_column="lang",
                              render_chat_template=False, max_rendered=1000, archive_weights=False, overwrite=True)
    res = package.run(args)
    m = load_yaml(res["package"] / "manifest.yaml")
    pc = field_value(m, "validation.per_class")
    assert set(pc) == {"positive", "negative", "neutral"} and pc["positive"]["support"] == 4
    assert field_value(m, "validation.weighted_f1") == 1.0
    assert field_value(m, "validation.min_class_support") == 4
    assert set(field_value(m, "validation.small_classes")) == {"positive", "negative", "neutral"}
    assert field_value(m, "validation.human_intercoder_alpha") < 1.0
    assert set(field_value(m, "validation.subsets")["results"]) == {"en", "nl"}
    assert (res["package"] / "validation" / "errors.csv").exists()
    assert (res["package"] / "validation" / "metrics.json").exists()
    assert any("fewer than 20" in w for w in m["warnings"])
    bp = res["best_practices"]
    assert _status(bp, "Validated against human labels") == "met"
    assert _status(bp, "At least 20-30") == "not_met"
    assert _status(bp, "Compared with human inter-coder") == "met"
    assert _status(bp, "Subsets") == "met"


def test_leakage_few_shot_text_in_validation(tmp_path):
    data = [{"id": "a", "text": "Nice."}, {"id": "b", "text": "Bad."}]
    gold = tmp_path / "g.csv"
    gold.write_text("id,label\na,positive\nb,negative\n", encoding="utf-8")
    res = run_validation(gold, "id", "label", {"a": "positive", "b": "positive"}, False, None,
                         data_rows=data, data_id="id", data_text="text", few_shot_texts=["nice."],
                         dev_ids={"b"})
    assert res["leakage"]["few_shot_in_validation"] == ["a"]
    assert res["leakage"]["dev_set_in_validation"] == ["b"]
    assert res["errors"] == [{"id": "b", "gold": "negative", "llm": "positive", "text": "Bad."}]
    assert res["per_class"]["positive"]["precision"] == 0.5


def test_prompt_lint_rules():
    good = lint("You are an expert annotator.\nOptions:\n- yes\n- no\n- uncertain\nIf unsure, answer uncertain.\n"
                "Answer with one option only.", "Text: {text}", ["yes", "no", "uncertain"],
                [{"name": "yes", "definition": "d", "examples": ["e"]}], None, ["system", "user"])
    st = {c["check"]: c["status"] for c in good}
    assert all(v == "pass" for v in st.values()), st
    bad = lint("Classify.", "{text}", ["yes", "no"], [{"name": "yes"}], None, ["user"])
    st = {c["check"]: c["status"] for c in bad}
    assert st["options_enumerated"] == "warn" and st["uncertain_option"] == "warn"
    assert st["system_instruction"] == "warn" and st["codebook_definitions"] == "warn"


def test_no_gold_is_flagged_and_practices_record_is_used(tmp_path):
    p = _selfrun(tmp_path)
    _w(p / "practices.yaml", """study:
  model_choice_rationale: "Fully open model, self-hosted, strong English performance."
  coding_procedure: "Three calibration rounds with two coders."
model:
  fine_tuned: false
  training_cutoff: "2023-12"
data:
  time_period: "2024-01..2024-06"
ethics:
  consent: "Public posts; IRB approval 2024-17."
  anonymization: "Usernames removed."
  data_minimization: "Only text fields sent."
  storage_and_transfer: "University servers in the EU."
  copyright_and_tos: "Platform research API terms."
""")
    confirm.cmd_practices(p, p / "practices.yaml")
    confirm_all(p)
    res = do_package(p, tmp_path / "pkg")
    bp = res["best_practices"]
    assert bp["validated"] is False
    assert _status(bp, "Validated against human labels") == "not_met"
    assert _status(bp, "Model choice explicitly argued") == "met"
    assert _status(bp, "Fine-tuning data") == "n/a"
    assert _status(bp, "Consent") == "met"
    assert _status(bp, "Training cutoff compared") == "met"
    assert _status(bp, "All parameters explicitly specified") == "met"
    assert _status(bp, "Open-source model with publicly known") == "met"
    text = (res["package"] / "BEST_PRACTICES.md").read_text(encoding="utf-8")
    assert "No validation against human labels" in text and "Törnberg" in text
    cb = (res["package"] / "PROMPT_CODEBOOK.md").read_text(encoding="utf-8")
    assert "**positive**: approval" in cb and "temperature" in cb and "Few-shot examples" in cb
    m = load_yaml(res["package"] / "manifest.yaml")
    assert m["study"]["model_choice_rationale"]["provenance"] == "researcher_provided"


def test_hosted_checklist_flags_defaults_and_dpa(tmp_path):
    p = make_hosted(tmp_path)
    do_discover(p)
    do_select(p, hosted_selection())
    do_collect(p, GPTOSS, "hosted_api")
    res = do_package(p, tmp_path / "pkg", data_shareable=False)
    bp = res["best_practices"]
    assert _status(bp, "Model hosted on own infrastructure") == "not_met"
    assert _status(bp, "All parameters explicitly specified") == "not_met"
    assert _status(bp, "Data processing agreement") == "not_documented"
    assert _status(bp, "Open-source model with publicly known") == "partial"
    m = load_yaml(res["package"] / "manifest.yaml")
    assert field_value(m, "model.training_cutoff").startswith("2024-06")


def test_archived_stability_runs(tmp_path):
    p = make_selfrun(tmp_path)
    labels = ["neutral"] + GOLD[1:]
    _w(p / "outputs" / "labels_repeat2.csv", "id,label\n" + "".join(f"t{i:02d},{g}\n" for i, g in enumerate(labels)))
    do_discover(p)
    do_select(p, selfrun_selection(p) + ["stability_runs=outputs/labels_repeat2.csv"])
    do_collect(p, OLMO, "self_run")
    res = do_package(p, tmp_path / "pkg")
    st = field_value(load_yaml(res["package"] / "manifest.yaml"), "stability.archived_runs")
    assert st["n_runs"] == 2 and 0 < st["krippendorff_alpha"] < 1
    assert _status(res["best_practices"], "Repeat-run and prompt-paraphrase") == "met"


def test_stability_script_repeats_and_paraphrase(tmp_path):
    p = _selfrun(tmp_path)
    confirm_all(p)
    pkg = do_package(p, tmp_path / "pkg")["package"]
    m = load_yaml(pkg / "manifest.yaml")
    labels = {r["id"]: r["label"] for r in read_records(pkg / "outputs" / "labels.csv")}
    table = {sha256_text(canonical_json(r["messages"])): labels[r["item_ids"][0]]
             for r in read_jsonl(pkg / field_value(m, "prompt.rendered_inputs_file"))}
    (tmp_path / "mock.json").write_text(json.dumps(table), encoding="utf-8")
    variant = tmp_path / "variant.yaml"
    variant.write_text((pkg / "prompts" / "prompt.yaml").read_text(encoding="utf-8")
                       .replace("Classify the text.", "Please classify this text."), encoding="utf-8")
    env = dict(os.environ, LLMRP_MOCK_RESPONSES=str(tmp_path / "mock.json"), LLMRP_MOCK_DEFAULT="neutral")
    r = subprocess.run([sys.executable, str(pkg / "scripts" / "stability.py"), "--package", str(pkg), "--engine",
                        "mock", "--sample", "6", "--repeats", "2", "--paraphrase", str(variant)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    res = json.loads((pkg / "reproduction" / "stability" / "stability.json").read_text(encoding="utf-8"))
    assert res["repeats"]["krippendorff_alpha_between_runs"] == 1.0
    assert all(v == 1.0 for v in res["repeats"]["exact_match_vs_archived"].values())
    para = list(res["paraphrases"]["exact_match_vs_archived"].values())[0]
    assert para < 1.0  # the mock does not know the paraphrased prompt
    assert (pkg / "reproduction" / "stability" / "STABILITY_REPORT.md").exists()
