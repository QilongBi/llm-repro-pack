"""Best-practice checklist after Törnberg (2024), "Best Practices for Text Annotation with
Large Language Models" (arXiv:2402.05129). Deterministic rules on manifest.yaml.

Status per item: met | partial | not_met | not_documented | n/a.
The checklist is separate from the A-D reproducibility grade.
"""
from __future__ import annotations

from typing import Any

from common import field_value, get_path, is_field, is_known, is_locked

UNSET_VALUES = {"not_set_provider_default", "engine_default", "default", "unknown", None, ""}


def _known(m: dict, path: str) -> bool:
    return is_known(get_path(m, path))


def _item(point: str, req: str, status: str, evidence: str) -> dict:
    return {"point": point, "requirement": req, "status": status, "evidence": evidence}


def evaluate(m: dict) -> dict:
    mode = field_value(m, "study.run_mode")
    hosted = mode == "hosted_api"
    items: list[dict] = []
    P1, P2, P3, P4, P5, P6, P7, P8, P9 = (
        "1 Model choice", "2 Systematic coding procedure", "3 Prompt codebook", "4 Validation",
        "5 Prompt engineering", "6 LLM parameters", "7 Ethics and law", "8 Stochasticity",
        "9 Training-data contamination")

    # 1. Model choice
    openness = field_value(m, "model.openness")
    items.append(_item(P1, "Open-source model with publicly known training data",
                       "met" if openness == "fully_open" else ("partial" if openness == "open_weight" else "not_documented"),
                       f"model.openness = {openness}"))
    if hosted:
        pinned = _known(m, "hosted_api.model_name") and _known(m, "hosted_api.system_fingerprints")
        items.append(_item(P1, "Fixed, documented model version that stays available",
                           "partial" if pinned or _known(m, "hosted_api.model_name") else "not_met",
                           "hosted API: the provider controls the version; model name"
                           + (" and system fingerprint recorded" if pinned else " recorded")))
        items.append(_item(P1, "Model hosted on own infrastructure", "not_met",
                           "hosted API run; self-hosting is preferred for control, privacy and replicability"))
    else:
        items.append(_item(P1, "Fixed, documented model version that stays available",
                           "met" if is_locked(get_path(m, "model.revision")) else "not_met",
                           f"model.revision = {field_value(m, 'model.revision')}"))
        items.append(_item(P1, "Model hosted on own infrastructure", "met", "self-run"))
    items.append(_item(P1, "Model choice explicitly argued", "met" if _known(m, "study.model_choice_rationale")
                       else "not_documented", "study.model_choice_rationale"))
    items.append(_item(P1, "Robustness across several models", "met" if _known(m, "study.models_compared")
                       else "not_documented", "study.models_compared"))
    ft = field_value(m, "model.fine_tuned")
    items.append(_item(P1, "Fine-tuning data provided (if fine-tuned)",
                       "n/a" if ft is False else ("met" if ft and _known(m, "model.fine_tuning_data") else
                                                  ("not_met" if ft else "not_documented")),
                       f"model.fine_tuned = {ft}"))
    items.append(_item(P1, "Guard-rail refusals examined", "met" if _known(m, "study.guard_rail_refusals")
                       else "not_documented", "study.guard_rail_refusals"))

    # 2. Systematic coding procedure
    items.append(_item(P2, "Iterative coding procedure with human coders documented",
                       "met" if _known(m, "study.coding_procedure") else "not_documented",
                       f"study.coding_procedure; human coders: {field_value(m, 'study.human_coders') or 'unknown'}"))

    # 3. Prompt codebook
    lint = {c["check"]: c for c in (field_value(m, "prompt.lint") or [])} if isinstance(field_value(m, "prompt.lint"), list) else {}
    defs = lint.get("codebook_definitions", {}).get("status")
    exs = lint.get("codebook_examples", {}).get("status")
    amb = lint.get("ambiguous_case_rule", {}).get("status")
    human_cb = bool((m.get("files") or {}).get("codebook"))
    items.append(_item(P3, "Category definitions", "met" if defs == "pass" or human_cb else
                       ("not_met" if defs == "warn" else "not_documented"),
                       "prompt labels with definitions" + ("; human codebook archived" if human_cb else "")))
    items.append(_item(P3, "Examples for each category", "met" if exs == "pass" else
                       ("partial" if _known(m, "prompt.few_shot") or human_cb else "not_documented"),
                       "label examples / few-shot / human codebook"))
    items.append(_item(P3, "Instructions for ambiguous cases", "met" if amb == "pass" else
                       ("partial" if human_cb else "not_met"), "prompt text" + (" (human codebook not parsed)" if human_cb else "")))
    items.append(_item(P3, "Prompt and parameters documented for reproduction",
                       "met" if _known(m, "prompt.rendered_inputs_file") and _known(m, "decoding.temperature")
                       else "not_met", "PROMPT_CODEBOOK.md generated from the manifest"))

    # 4. Validation
    has_gold = _known(m, "validation.gold_sha256")
    items.append(_item(P4, "Validated against human labels", "met" if has_gold else "not_met",
                       f"accuracy {field_value(m, 'validation.accuracy')}, macro-F1 {field_value(m, 'validation.macro_f1')}"
                       if has_gold else "no gold set provided; validation is a basic requirement"))
    if has_gold:
        apf = field_value(m, "validation.after_prompt_freeze")
        items.append(_item(P4, "Validation done after the prompt was final",
                           "met" if apf is True else ("not_met" if apf is False else "not_documented"),
                           "validation.after_prompt_freeze"))
        mins = field_value(m, "validation.min_class_support")
        small = field_value(m, "validation.small_classes") or []
        items.append(_item(P4, "At least 20-30 validation items per category",
                           "met" if isinstance(mins, int) and mins >= 30 else
                           ("partial" if isinstance(mins, int) and mins >= 20 else "not_met"),
                           f"smallest class: {mins}" + (f"; under 20: {', '.join(small)}" if small else "")))
        items.append(_item(P4, "Metrics beyond accuracy (P/R/F1, weighted F1, kappa)",
                           "met" if _known(m, "validation.per_class") or _known(m, "validation.per_label")
                           else "partial", "validation.per_class, weighted_f1, cohens_kappa"))
        items.append(_item(P4, "Metric choice argued", "met" if _known(m, "validation.metric_rationale")
                           else "not_documented", "validation.metric_rationale"))
        icr = field_value(m, "validation.human_intercoder_alpha")
        items.append(_item(P4, "Compared with human inter-coder reliability",
                           "met" if icr not in (None, "unknown") else "not_documented",
                           f"human alpha {icr}, LLM+humans alpha {field_value(m, 'validation.llm_and_humans_alpha')}"))
        subs = field_value(m, "validation.subsets")
        items.append(_item(P4, "Subsets (languages, contexts, periods) validated separately",
                           "met" if subs not in (None, "unknown") else "not_documented",
                           f"subset column: {(subs or {}).get('column') if isinstance(subs, dict) else 'none'}"))
        fa = _known(m, "validation.failure_analysis")
        items.append(_item(P4, "Failures examined and explained", "met" if fa else "partial",
                           "validation/errors.csv" + (" + validation.failure_analysis" if fa else
                                                      "; write up the failure analysis")))
        lk = field_value(m, "validation.leakage") or {}
        leaked = (lk.get("few_shot_in_validation") or []) + (lk.get("dev_set_in_validation") or [])
        checked = lk.get("checked") or []
        items.append(_item(P4, "Validation data not used for prompt development",
                           "not_met" if leaked else ("met" if checked else "not_documented"),
                           f"checked: {', '.join(checked) or 'nothing (no few-shot texts or dev set)'}"
                           + (f"; overlap: {', '.join(leaked)}" if leaked else "")))

    # 5. Prompt engineering (advisory)
    for name, req in (("options_enumerated", "Answer options enumerated"),
                      ("options_one_per_line", "Options on separate lines"),
                      ("uncertain_option", "An 'uncertain / don't know' option"),
                      ("output_constraint", "Output format constrained"),
                      ("system_instruction", "Instructions in a system message"),
                      ("few_shot_as_dialogue", "Few-shot examples as user/assistant turns")):
        c = lint.get(name)
        if c:
            items.append(_item(P5, req, {"pass": "met", "warn": "not_met"}.get(c["status"], "n/a"), c["detail"]))
    if not lint:
        items.append(_item(P5, "Structured prompt (context, question, constraints)", "not_documented",
                           "no prompt available to check"))

    # 6. Parameters explicitly specified
    keys = ["temperature", "top_p", "max_tokens"] + ([] if hosted else ["top_k"])
    unset = [k for k in keys if field_value(m, f"decoding.{k}") in UNSET_VALUES]
    items.append(_item(P6, "All parameters explicitly specified, even defaults",
                       "met" if not unset else "not_met",
                       "all set" if not unset else "not explicitly set: " + ", ".join(unset)))
    t = field_value(m, "decoding.temperature")
    if isinstance(t, (int, float)):
        items.append(_item(P6, "Low temperature for annotation", "met" if t <= 0.3 else "partial", f"temperature {t}"))

    # 7. Ethics and law
    eth = [("consent", "Consent / informing data subjects"), ("anonymization", "Anonymization or pseudonymization"),
           ("data_minimization", "Data minimization"), ("storage_and_transfer", "Storage and cross-border transfer"),
           ("copyright_and_tos", "Copyright and terms of service")]
    if hosted:
        eth.insert(1, ("data_processing_agreement", "Data processing agreement with the provider"))
    for key, req in eth:
        items.append(_item(P7, req, "met" if _known(m, f"ethics.{key}") else "not_documented", f"ethics.{key}"))

    # 8. Stochasticity
    st = field_value(m, "stability.archived_runs")
    items.append(_item(P8, "Repeat-run and prompt-paraphrase stability (Krippendorff's alpha)",
                       "met" if isinstance(st, dict) else "not_documented",
                       f"alpha {st.get('krippendorff_alpha')} over {st.get('n_runs')} runs" if isinstance(st, dict)
                       else "no archived stability runs; run scripts/stability.py"))

    # 9. Training data contamination
    cutoff, period = field_value(m, "model.training_cutoff"), field_value(m, "data.time_period")
    items.append(_item(P9, "Training cutoff compared with the data period",
                       "met" if _known(m, "model.training_cutoff") and _known(m, "data.time_period") else
                       ("partial" if _known(m, "model.training_cutoff") or _known(m, "data.time_period") else "not_documented"),
                       f"cutoff {cutoff}; data period {period}"))
    if has_gold:
        pub = field_value(m, "validation.gold_publicly_available")
        items.append(_item(P9, "Validation set not a public labeled dataset",
                           "met" if pub is False else ("not_met" if pub is True else "not_documented"),
                           "validation.gold_publicly_available"))

    counts: dict[str, int] = {}
    for i in items:
        counts[i["status"]] = counts.get(i["status"], 0) + 1
    return {"source": "Törnberg (2024), arXiv:2402.05129", "items": items, "counts": counts,
            "validated": has_gold}
