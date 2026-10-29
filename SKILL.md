---
name: llm-repro-pack
description: Package completed, self-run LLM text-classification runs (single- or multi-label, fully open or open-weight models run on a machine the researcher controlled) into grade A or B reproducibility packages with provenance-labelled manifests, hashes, archived scripts, validation metrics and a best-practice checklist (Törnberg 2024); and reproduce such packages or test their prompt stability. Refuses runs that cannot reach grade A or B, including third-party hosted APIs. Use when a researcher wants to archive, document, validate, share or re-run an LLM annotation/classification study with open weights.
---

# llm-repro-pack

You orchestrate. The bundled scripts do all deterministic work. Never write new code
to hash, parse, grade, compare or run models; call the scripts. Never edit scripts inside
a package.

Scripts live in `scripts/` next to this file. Run them with the researcher's Python
(needs PyYAML). `S=<path to this skill>/scripts`.

Language: talk to the researcher in their language; everything written to disk is English.

## Scope: grade A and B only

| Grade | Conditions |
|---|---|
| A | Fully open model, self-run, every critical field locked, openness record complete |
| B | Open-weight model (or fully open without a complete openness record), self-run, every critical field locked |

Out of scope, refused:
- Runs through a third-party hosted API (Groq, Together, OpenRouter, ...). `collect_env.py`
  stops with an error. Tell the researcher to re-run on a machine they control with pinned
  open weights.
- Runs where a critical field stays unknown, inferred-but-unconfirmed or in conflict.
  `package.py` builds nothing (exit code 3) and lists what must be locked first.

Self-run means a machine the researcher controlled: local NVIDIA GPU, Apple Silicon, CPU,
their own cloud VM or an HPC node. Ask early whether that is the case; if not, stop.

## Mode `init`: create the recommended layout

```
python $S/init_project.py <project>
```

Creates `prompt.yaml`, `config.yaml`, `practices.yaml`, `data/`, `outputs/`, `logs/`, `env/`
without overwriting anything. Tell the researcher what each file is for (see `env/README.md`).

## Mode `package`: archive a finished run

Ask for these inputs first. Do not guess them:
project directory, model ID (and revision if known), confirmation that the run was
self-run, path to the archived labels. Optional: environment package path, gold
validation set (and its human coder columns and any subset column such as language),
prompt-development set, human codebook, archived repeat or paraphrase runs, output
directory, whether the data and the gold set may be shared.

1. **Discover.** `python $S/discover.py --project P --labels <labels> [--env-package E] [--gold G]`
   Show the researcher every candidate with path and modified time. Ask which file is the
   final prompt, config, data, labels, raw outputs, requests, logs, code and environment
   package, and which columns are the ID, text and label columns. Nothing is hashed before
   this confirmation.
2. **Confirm the selection.**
   `python $S/confirm.py select --project P role=path ...` (roles: prompt_files, config,
   data, labels, raw_outputs, requests, logs, code, env_package, gold, codebook, dev_set,
   stability_runs, practices; list roles may repeat).
   `python $S/confirm.py columns --project P data.id_column=id data.text_column=text outputs.id_column=id outputs.label_column=label`
3. **Collect evidence.**
   `python $S/collect_env.py --project P --model-id M [--revision R] [--hf-cache D] [--model-path D]`
   Add `--hub` or `--probe-cloud` only if the researcher agrees to network lookups.
   If it stops because a hosted API was used, explain the scope and stop.
4. **Resolve with the researcher.** `python $S/confirm.py pending --project P`
   - For every conflict, show all candidates with their sources and ask which is right.
     Never pick one yourself.
   - For every inferred value, ask the researcher to confirm or correct it.
     Confirm: `confirm.py field --project P <field> --as researcher_confirmed`.
     Correct: `confirm.py field --project P <field>=<value> --as researcher_provided`.
   - Report every WARNING (post-hoc environment exports, code changed after the run,
     ambiguous HF snapshots).
   - If the researcher does not know a value, leave it. It stays `unknown`. Never invent
     one to reach grade A or B.
   - You may record your own inference only with a basis:
     `confirm.py field ... --as claude_inferred --basis "<why>"`, then ask for confirmation.
5. **Best-practice record (Törnberg 2024).** Walk the researcher through
   `templates/practices.yaml` (copied into the project by `init`): model-choice rationale,
   other models compared, coding procedure, fine-tuning data, training cutoff and data
   period, whether validation happened after the prompt was final, whether the gold set is
   public, failure analysis, and the ethics/legal items (consent, anonymization, data
   minimization, storage and transfer, copyright/ToS). Write only what the researcher
   says; leave the rest empty. Load it:
   `python $S/confirm.py practices --project P --file P/practices.yaml`
6. **Build.** `python $S/package.py --project P --data-shareable yes|no [--data-access "..."] [--out DIR]`
   With a gold set: `--gold-shareable yes|no [--gold-label-column C] [--gold-coder-columns c1,c2] [--subset-column C]`.
   Add `--render-chat-template` when transformers and the tokenizer are available.
   Never pass `--archive-weights` unless the researcher asks; the script refuses when the
   license does not allow redistribution.
   Exit code 3 means "not eligible": show the listed fields (also in
   `.llmrp/readiness.yaml`), go back to step 4 with the researcher, and rebuild. If the
   fields cannot be supplied, the run is out of scope; say so.
7. **Report.** Give the grade (A or B), the reasons and, for a fully open model at grade B,
   what is missing for A (printed by package.py, written to `REPRODUCIBILITY.md`). Then
   summarize `BEST_PRACTICES.md`: say plainly when there is no validation against human
   labels, when classes have fewer than 20-30 validation items, when validation items
   leaked into the few-shot examples or the development set, and which parameters were
   left at defaults. If the self-check fails (secrets, checksums, schema), stop and tell
   the researcher. Do not edit the package by hand; fix the input and rebuild with `--overwrite`.
8. **Optional check.** Offer to re-run a small sample:
   `python <package>/scripts/reproduce.py --package <package> --engine <engine> --sample 20`
   This checks the package; it is not a pre-flight check.

## Mode `stability`: prompt-stability analysis (Törnberg 2024, section 8)

Offer it when `BEST_PRACTICES.md` shows no stability evidence and the model can be run.
Paraphrases of `prompts/prompt.yaml` must come from the researcher; you may draft them,
but the researcher approves each file before it is used. Never edit the archived prompt.

```
python <pkg>/scripts/stability.py --package <pkg> --engine <engine> --sample 50 --repeats 3 --paraphrase v1.yaml v2.yaml
```

Report Krippendorff's alpha between runs and agreement with the archived labels from
`reproduction/stability/STABILITY_REPORT.md`. The data file must be in the package or
placed under `data/` (hash-checked).

## Mode `reproduce`: re-run an existing package

1. `python <pkg>/scripts/reproduce.py --package <pkg> --verify-only`
   Stops with exit code 3 on any checksum, prompt or rendered-input mismatch. Report the
   mismatch and stop.
2. `python <pkg>/scripts/reproduce.py --package <pkg> --rebuild --dry-run` and show the
   commands. Run them (drop `--dry-run`) only with the researcher's approval.
3. Get the weights at the recorded revision (the researcher downloads them; gated models
   need license acceptance).
4. `python <pkg>/scripts/reproduce.py --package <pkg> --engine transformers|vllm|ollama [--sample N]`
   Use the archived scripts in the package, never the copies in this skill.
5. Give the researcher `reproduction/REPRODUCTION_REPORT.md`: exact match rate, Cohen's
   kappa, Krippendorff's alpha and environment differences. Bit-identical results are
   expected on the same hardware class with the same engine version, precision and batch
   size; on other hardware, only high agreement is expected.

## Rules

- Provenance labels: `run_time_evidence`, `researcher_provided`, `code_declared`,
  `claude_inferred` (always with a basis), `researcher_confirmed`, `unknown`.
- Never write or print API keys or tokens; the self-check scans every package for them.
- Never copy or upload weights by default. Reference them by repo ID, revision and SHA-256.
- Only grade A and B packages exist. See `docs/grading.md`.
- Model-specific fields come from `profiles/*.yaml`. Reference cases: `docs/reference-cases.md`.
- The best-practice checklist (`BEST_PRACTICES.md`, `docs/best-practices.md`) is separate
  from the grade. It never changes the archived run; it documents it.
- To add an engine or a model family, see `docs/extending.md`.
