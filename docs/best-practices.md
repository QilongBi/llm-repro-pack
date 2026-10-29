# Best-practice checklist (Törnberg 2024)

Source: P. Törnberg, *Best Practices for Text Annotation with Large Language Models*,
arXiv:2402.05129 (2024). Every package gets `BEST_PRACTICES.md`, built by
`scripts/best_practices.py` from `manifest.yaml` with fixed rules. It is separate from the
A/B reproducibility grade: the grade says whether the run can be re-executed, the
checklist says how well the annotation follows recommended practice.

Status values: met, partial, not met, not documented, n/a.

| Paper section | What the skill records or computes | Where |
|---|---|---|
| 1 Model choice | openness; pinned version; rationale; other models compared; fine-tuning data; guard-rail refusals | `model.*`, `study.model_choice_rationale`, `study.models_compared`, `model.fine_tuned`, `model.fine_tuning_data` (critical when fine-tuned) |
| 2 Systematic coding procedure | description of the iterative procedure, rounds, number of human coders | `study.coding_procedure`, `study.development_iterations`, `study.human_coders` |
| 3 Prompt codebook | definitions and examples per category, ambiguous-case rule, prompt + parameters in one document; the human codebook is archived | `PROMPT_CODEBOOK.md`, role `codebook`, `prompt.lint` |
| 4 Validation | accuracy, macro-F1, weighted F1, per-class precision/recall/F1, confusion matrix, kappa, alpha; class support (warns below 20); human inter-coder alpha and LLM vs each coder; per-subset metrics; misclassified items; leakage of validation items into few-shot examples or the development set; validation after prompt freeze; metric rationale; failure analysis | `validation.*`, `validation/metrics.json`, `validation/errors.csv` |
| 5 Prompt engineering | advisory lint: options enumerated and one per line, uncertain option, output constraint, system message, few-shot as user/assistant turns | `prompt.lint` |
| 6 Parameters | every parameter must be explicit; provider or engine defaults are flagged | `decoding.*` |
| 7 Ethics and law | consent, anonymization, data minimization, storage and transfer, copyright/ToS | `ethics.*` |
| 8 Stochasticity | alpha across archived repeat/paraphrase runs; `scripts/stability.py` for new repeats and researcher-written paraphrases | `stability.archived_runs`, `reproduction/stability/` |
| 9 Training data | training cutoff vs data period; whether the gold set is a public dataset | `model.training_cutoff`, `data.time_period`, `validation.gold_publicly_available` |

## What the skill does not do

The skill packages runs that are already finished. It cannot go back and redo the coding
procedure, validate a prompt before it is frozen, or change the prompt. Those steps
belong to the study design. The checklist records whether they happened, and leaves them
as *not documented* when nobody says.

Researcher answers come from `practices.yaml` (`confirm.py practices`). They enter the
manifest as `researcher_provided`. The agent never fills them in itself.
