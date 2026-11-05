# llm-repro-pack

A Claude skill that packages **completed** LLM text-classification runs from social-science
research into standardized, verifiable reproducibility packages, and reproduces them later.

Closed LLM APIs threaten reproducibility: models change behind stable aliases, get retired,
and run on stacks nobody outside the provider can inspect. Open models can be pinned
exactly (weights commit, tokenizer and chat template, engine version, precision, decoding
settings, hardware). Following Spirling (2023) and Palmer, Smith and Spirling (2024), this
skill only handles open and open-weight models run on a machine the researcher controlled,
and only builds packages that reach grade A or B. Each package is also checked against the
best practices for LLM text annotation in Törnberg (2024).

The skill itself only orchestrates. Fixed, bundled Python scripts do all deterministic work:
discovery, evidence collection, hashing, manifest building, grading, self-checks, re-running
and comparison. Packages carry their own copy of those scripts, so a reproduction does not
depend on which agent or person runs it.

## What you can and cannot use it for

| Use it for | Do not use it for |
|---|---|
| Runs that are **already finished**: you have the labels | Planning or running a new study; VRAM or runtime estimates |
| Single-label and multi-label text classification | Generation, summarization, extraction, numeric scoring |
| Fully open models (e.g. OLMo 2) and open-weight models (e.g. gpt-oss, Qwen, Llama) | Closed models (GPT-4, Claude, Gemini) |
| Runs on a machine you controlled: local NVIDIA GPU, Apple Silicon, CPU, your own cloud VM or HPC node | Third-party hosted APIs (Groq, Together, OpenRouter, ...). Evidence collection stops with an error |
| Any open or open-weight model; model-specific rules come from optional profiles, other models use the default profile | Chat web interfaces |
| Re-running with transformers, vLLM or Ollama. Runs made with MLX, llama.cpp or SGLang can be packaged, but re-running them needs an adapter first (`docs/extending.md`) | Re-running an MLX, llama.cpp or SGLang run without adding an adapter |
| Runs where every critical setting can be found in files or supplied by you | Runs whose critical settings are lost. No package is built (exit code 3) and the missing fields are listed |
| Re-running a package (reproduce mode) and prompt-stability analysis | Repairing or improving labels; invalid outputs stay invalid |

The skill documents the study; it cannot redo it. Steps that belong to the study design,
such as iterative calibration with human coders or validating after the prompt is frozen,
are recorded as done or not done in the best-practice checklist. They are never filled in
after the fact.

## Grades

| Grade | Conditions | What can be reproduced |
|---|---|---|
| A | Fully open model, self-run, all critical fields locked, openness record complete (training data, data version, training code commit, checkpoint) | Bit-identical on the same hardware class |
| B | Open-weight model (or fully open without a complete openness record), self-run, all critical fields locked | Bit-identical on the same hardware class |

There are no lower grades. A field is *locked* when its value is known, its provenance is
`run_time_evidence`, `researcher_provided`, `code_declared` or `researcher_confirmed`, and no
conflict between sources is open. On different hardware (e.g. Apple Silicon vs NVIDIA),
expect high agreement rather than identical output; judge it with Cohen's kappa and
Krippendorff's alpha.

## Workflow

### Before the study (optional)

`python scripts/init_project.py <project>` creates the recommended layout: `prompt.yaml`,
`config.yaml`, `practices.yaml`, `data/`, `outputs/`, `logs/`, `env/`. While you run the
model, save `pip freeze`, `nvidia-smi`, the run log and a config with every decoding
parameter written out. Then almost all technical fields can be read automatically.

### Package mode

| Step | What happens | Who acts |
|---|---|---|
| 1 Discover | `discover.py` lists candidate files with path and modified time | script; you confirm which file is final |
| 2 Confirm selection | `confirm.py select`, `confirm.py columns`: prompt, config, data, labels, raw outputs, requests, logs, code, environment files, gold set; ID, text and label columns | you |
| 3 Collect evidence | `collect_env.py` reads every source and labels each value with its provenance; conflicts are kept, never resolved silently; a hosted-API endpoint stops the run | script |
| 4 Resolve | `confirm.py pending` lists conflicts and inferred values; you confirm or correct each one; unknown values stay unknown | you, guided by Claude |
| 5 Best-practice record | fill in `practices.yaml`, load it with `confirm.py practices` | you |
| 6 Build | `package.py` grades the run. If it reaches A or B, the package is written and self-checked (schema, checksums, secret scan). If not, nothing is written and the missing fields are listed (`.llmrp/readiness.yaml`) | script |
| 7 Report | grade, caveats, best-practice checklist: missing validation, small classes, leakage, parameters left at defaults | Claude |
| 8 Optional check | re-run a small sample with the archived scripts | you decide |

### Reproduce mode (for later researchers)

1. `reproduce.py --verify-only`: checks every checksum and the prompt and rendered-input
   hashes. Stops on any mismatch.
2. `reproduce.py --rebuild --dry-run`: shows the environment commands (Docker, conda or
   pip). They run only with your approval.
3. Download the weights at the recorded revision. Weights are never included; they are
   referenced by repo ID, revision and SHA-256.
4. `reproduce.py --engine transformers|vllm|ollama [--sample N]`: runs the **archived**
   scripts and compares with the archived labels (exact match rate, Cohen's kappa,
   Krippendorff's alpha, environment differences).

### Stability mode

`stability.py --repeats K --paraphrase v1.yaml ...` re-runs a deterministic sample with
repeated identical prompts and with paraphrased prompts, and reports Krippendorff's alpha
between runs. Paraphrases are written or approved by the researcher.

## Who supplies each field

Claude does not compute values itself; it calls the scripts, which read the project folder.
The question is whether a value read from the folder counts as locked straight away.

### A. Read from the folder and locked without further confirmation

| Source | Fields | Provenance |
|---|---|---|
| Environment files (`pip freeze`, conda export, lockfile, Dockerfile, image digest, `nvidia-smi`) | library versions (torch, transformers, vllm, ...), engine version, Python, container digest, GPU model and memory, driver, CUDA, OS | `researcher_provided` |
| Run logs and metadata in outputs | engine and version, model revision, temperature, top_p, top_k, seed, max_tokens, dtype, GPU, CUDA, driver, run timestamps, whether the gpt-oss reasoning trace was saved | `run_time_evidence` |
| Archived request payloads | the exact messages sent and the request parameters; parameters absent from every request are recorded as `engine_default` | `run_time_evidence` |
| Config files and code (read at the run commit when git is available) | model ID, revision, engine, all decoding parameters, batch size, batch-invariant mode, precision, label set, task type, column names, label extraction rule, model-specific parameters (e.g. gpt-oss reasoning effort and harmony rendering) | `code_declared` |
| The prompt, data and label files you selected | prompt file hashes, data hash, item count, labels hash, fully rendered model inputs and their hash, prompt structure checks | `researcher_confirmed` |
| Gold set and repeated runs (if provided) | accuracy, macro-F1, weighted F1, per-class P/R/F1, confusion matrix, kappa, alpha, human inter-coder reliability, per-subset metrics, leakage checks, misclassified items, stability alpha | computed by the scripts |

### B. Inferred, locked only after you confirm (`claude_inferred`)

| Source | Fields |
|---|---|
| Local Hugging Face cache or model directory | revision (snapshot folder name), weight file hashes, tokenizer hash, chat template hash, weight format, quantization, whether weights were converted, license from the model card |
| Model profiles (`profiles/*.yaml`) | openness, license and restrictions, whether training data is disclosed, reference values for training data and checkpoint (OLMo 2), training cutoff (gpt-oss), model defaults (e.g. reasoning effort medium, harmony format, final-channel extraction) |
| Heuristics | ID, text and label columns; label set from observed labels; single- or multi-label; extraction rule from the shape of raw outputs |
| Current machine (only fills gaps) | OS, Python, library versions, GPU, cloud instance type |
| Git history | the commit that produced the outputs |
| Claude's own inference (rare) | always with a stated basis, then confirmed by you |

Disagreements between sources are recorded as conflicts. You decide every one.

### C. Supplied by you

**Inputs and decisions**

- project directory, model ID, revision if known, path to the labels
- confirmation that the run was self-run
- which file plays which role
- whether the data may be shared, and how others can get it if not
- whether the gold set may be shared; its label, coder and subset columns
- output directory and title
- permission for network lookups (`--hub`, `--probe-cloud`), for archiving weights, and
  for running environment rebuild commands

**Critical fields when no file contains them** (otherwise no package is built)

- model revision, batch size, precision, batch-invariant mode, seed when temperature > 0
- hardware, CUDA and driver (when `nvidia-smi` was not saved)
- engine version and OS (when there are no environment files or logs)
- model-specific parameters, e.g. gpt-oss reasoning effort
- for grade A: the training code commit (not in any profile), plus confirmation of
  training data, data version and checkpoint

**Best-practice record (`practices.yaml`, only you know these)**

| Area | Fields |
|---|---|
| Model choice | rationale, other models compared, refusals by the model |
| Coding procedure | iterative procedure, revision rounds, number of human coders |
| Fine-tuning | whether the model was fine-tuned, and the fine-tuning data |
| Training-data contamination | training cutoff (gpt-oss has a reference value), period the texts cover, language |
| Validation | done after the prompt was final, gold set public or not, metric rationale, failure analysis |
| Ethics and law | consent, anonymization, data minimization, storage and transfer, copyright and terms of service |

**Stability analysis**: paraphrased prompts are yours; Claude may draft them, but you
approve each file.

## What a package contains

```
manifest.yaml        every field: value + provenance label + source
README.md            how to reproduce
REPRODUCIBILITY.md   grade A or B, conflicts, caveats, what is missing for A
BEST_PRACTICES.md    checklist against Törnberg (2024)
PROMPT_CODEBOOK.md   categories, prompts, examples and settings in one document
env/                 environment files, hardware info, rebuild instructions
prompts/             prompt files + fully rendered model inputs (hashed)
scripts/             archived runner, engine adapters, reproduce.sh, compare, stability
outputs/             archived labels, raw outputs, requests
logs/
data/                input data, or access instructions + hash
validation/          per-class metrics, inter-coder reliability, misclassified items
checksums.sha256
```

## Best-practice checklist

`BEST_PRACTICES.md` checks each package against the nine areas of Törnberg (2024): model
choice, systematic coding procedure, prompt codebook, validation, prompt engineering,
explicit parameters, ethics and law, stochasticity, and training-data contamination. Each
item is met, partial, not met, not documented or n/a. The checklist is separate from the
grade: the grade says whether the run can be re-executed, the checklist says how well the
annotation follows recommended practice. Details: `docs/best-practices.md`.

## Install

Requires Python 3.9+ and PyYAML. Engines are optional and only needed to re-run:
transformers, vLLM or Ollama.

```
pip install -r requirements.txt
```

To use it as a Claude Code skill, copy or link this folder to `~/.claude/skills/llm-repro-pack`.

## Quick start (without an agent)

```
S=path/to/llm-repro-pack/scripts
python $S/discover.py --project study --labels outputs/labels.csv
python $S/confirm.py select --project study labels=outputs/labels.csv data=data/items.csv prompt_files=prompt.yaml config=config.yaml logs=logs env_package=env
python $S/collect_env.py --project study --model-id allenai/OLMo-2-1124-7B-Instruct
python $S/confirm.py pending --project study
python $S/confirm.py field --project study model.revision --as researcher_confirmed
python $S/confirm.py practices --project study --file study/practices.yaml
python $S/package.py --project study --data-shareable yes
python study-repro-pack/scripts/reproduce.py --package study-repro-pack --engine vllm --sample 20
python study-repro-pack/scripts/stability.py --package study-repro-pack --engine vllm --repeats 3 --paraphrase v1.yaml
```

See `SKILL.md` for the agent instructions, `docs/references.md` for citation status, `docs/grading.md` for the evidence hierarchy,
`docs/best-practices.md` for the checklist, `docs/reference-cases.md` for OLMo 2 and
gpt-oss, and `docs/extending.md` to add engines and model families.

## Tests

```
python -m pytest tests
```

The tests use synthetic projects, fake HF cache snapshots and a mock adapter. They never
download models.

## References

- Törnberg, P. (2024). Best practices for text annotation with large language models.
  *arXiv preprint* arXiv:2402.05129. https://arxiv.org/abs/2402.05129
- Spirling, A. (2023). Why open-source generative AI models are an ethical way forward for
  science. *Nature*, 616, 413.
- Palmer, A., Smith, N. A., & Spirling, A. (2024). Using proprietary language models in
  academic research requires explicit justification. *Nature Computational Science*, 4, 2–3.

## License

MIT. See `LICENSE`.
