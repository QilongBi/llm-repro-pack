# llm-repro-pack

A Claude skill that packages **completed** LLM text-classification runs from social-science
research into standardized, verifiable reproducibility packages, and reproduces them later.

Closed LLM APIs threaten reproducibility: models change behind stable aliases, get retired,
and run on stacks nobody outside the provider can inspect. Open models can be pinned
exactly (weights commit, tokenizer and chat template, engine version, precision, decoding
settings, hardware). Following Spirling (Nature, 2023) and Palmer, Smith & Spirling
(Nature Computational Science, 2024), this skill targets open and open-weight models and
treats hosted APIs as a degraded mode.

The skill itself only orchestrates. Fixed, bundled Python scripts do all deterministic work:
discovery, evidence collection, hashing, manifest building, grading, self-checks, re-running
and comparison. Packages carry their own copy of those scripts, so a reproduction does not
depend on which agent or person runs it.

## What a package contains

```
manifest.yaml        every field: value + provenance label + source
README.md            how to reproduce
REPRODUCIBILITY.md   grade A-D, missing items, conflicts, caveats
env/                 environment files, hardware info, rebuild instructions
prompts/             prompt files + fully rendered model inputs (hashed)
scripts/             archived runner, engine adapters, reproduce.sh, compare
outputs/             archived labels, raw outputs, requests
logs/
data/                input data, or access instructions + hash
checksums.sha256
```

## Grades

| Grade | Conditions | What can be reproduced |
|---|---|---|
| A | Fully open model, self-run, all critical fields locked | Bit-identical on the same hardware class |
| B | Open-weight model, self-run, all critical fields locked | Bit-identical on the same hardware class |
| C | Some critical fields missing, or hosted API mode | Statistical reproduction (agreement metrics) |
| D | Only archived outputs available | Downstream analysis only |

## Install

Requires Python 3.9+ and PyYAML. Engines are optional and only needed to re-run:
transformers, vLLM, Ollama, or any OpenAI-compatible hosted API.

```
pip install -r requirements.txt
```

To use it as a Claude Code skill, copy or link this folder to `~/.claude/skills/llm-repro-pack`.

## Quick start (without an agent)

```
S=path/to/llm-repro-pack/scripts
python $S/discover.py --project study --labels outputs/labels.csv
python $S/confirm.py select --project study labels=outputs/labels.csv data=data/items.csv prompt_files=prompt.yaml config=config.yaml logs=logs env_package=env
python $S/collect_env.py --project study --model-id allenai/OLMo-2-1124-7B-Instruct --run-mode self_run
python $S/confirm.py pending --project study
python $S/confirm.py field --project study model.revision --as researcher_confirmed
python $S/package.py --project study --data-shareable yes
python study-repro-pack/scripts/reproduce.py --package study-repro-pack --engine vllm --sample 20
```

See `SKILL.md` for the full workflow, `docs/grading.md` for the evidence hierarchy,
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
