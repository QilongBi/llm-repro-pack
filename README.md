# llm-repro-pack

A Claude skill that packages **completed** LLM text-classification runs from social-science
research into standardized, verifiable reproducibility packages, and reproduces them later.

Closed LLM APIs threaten reproducibility: models change behind stable aliases, get retired,
and run on stacks nobody outside the provider can inspect. Open models can be pinned
exactly (weights commit, tokenizer and chat template, engine version, precision, decoding
settings, hardware). Following Spirling (Nature, 2023) and Palmer, Smith & Spirling
(Nature Computational Science, 2024), this skill only handles open and open-weight models
run on a machine the researcher controlled, and only builds packages that reach grade A
or B. Runs through third-party hosted APIs, and runs whose critical settings cannot be
pinned, are refused. Each package is also checked against the best
practices for LLM text annotation in Törnberg (2024, arXiv:2402.05129): model choice,
coding procedure, prompt codebook, validation, prompt structure, explicit parameters,
ethics and law, stochasticity and training-data contamination.

The skill itself only orchestrates. Fixed, bundled Python scripts do all deterministic work:
discovery, evidence collection, hashing, manifest building, grading, self-checks, re-running
and comparison. Packages carry their own copy of those scripts, so a reproduction does not
depend on which agent or person runs it.

## What a package contains

```
manifest.yaml        every field: value + provenance label + source
README.md            how to reproduce
REPRODUCIBILITY.md   grade A or B, conflicts, caveats, what is missing for A
BEST_PRACTICES.md    checklist against Törnberg (2024)
PROMPT_CODEBOOK.md   categories, prompts, examples and settings in one document
env/                 environment files, hardware info, rebuild instructions
prompts/             prompt files + fully rendered model inputs (hashed)
scripts/             archived runner, engine adapters, reproduce.sh, compare
outputs/             archived labels, raw outputs, requests
logs/
data/                input data, or access instructions + hash
validation/          per-class metrics, inter-coder reliability, misclassified items
checksums.sha256
```

## Grades

| Grade | Conditions | What can be reproduced |
|---|---|---|
| A | Fully open model, self-run, all critical fields locked | Bit-identical on the same hardware class |
| B | Open-weight model, self-run, all critical fields locked | Bit-identical on the same hardware class |

There are no lower grades. If a critical field stays unknown, unconfirmed or in conflict,
`package.py` builds nothing and lists the fields to lock (exit code 3).

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

See `SKILL.md` for the full workflow, `docs/grading.md` for the evidence hierarchy,
`docs/best-practices.md` for the Törnberg (2024) checklist,
`docs/reference-cases.md` for OLMo 2 and gpt-oss, and `docs/extending.md` to add engines
and model families.

## Tests

```
python -m pytest tests
```

The tests use synthetic projects, fake HF cache snapshots and a mock adapter. They never
download models.

## License

MIT. See `LICENSE`.
