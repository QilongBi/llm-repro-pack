# Evidence hierarchy and grading

## Evidence hierarchy

Each field is collected from every available source. The highest-ranked source wins, and
any disagreement is recorded as a conflict for the researcher to resolve.

| Rank | Source | Provenance label |
|---|---|---|
| 0 | Researcher answers during packaging | `researcher_confirmed` / `researcher_provided` |
| 1 | Researcher-provided environment package (Docker digest, conda export, lockfile, pip freeze, hardware info) | `researcher_provided` |
| 2 | Run-time evidence: logs, metadata in outputs, archived requests and responses | `run_time_evidence` |
| 3 | Code and config, read at the git commit that produced the outputs | `code_declared` |
| 4 | Current environment, HF cache, model card, model profile defaults | `claude_inferred` (with basis) |
| 5 | Nothing | `unknown` |

Rules:

- Environment files modified more than a day after the last output are flagged
  "post-hoc export, may not match run-time environment".
- The run commit is the latest commit before the last output time. Without git, code
  files modified after the outputs are flagged as lower confidence.
- Several HF cache snapshots of the same repo are reported as ambiguous unless exactly
  one matches higher-ranked evidence or predates the outputs.
- Values are never invented. Missing evidence stays `unknown`.

## Locked fields

A field is locked when its value is known, its provenance is `run_time_evidence`,
`researcher_provided`, `code_declared` or `researcher_confirmed`, and no conflict is open.
`claude_inferred` values become locked only after the researcher confirms them.

## Grades

llm-repro-pack only builds grade A and B packages.

| Grade | Rule |
|---|---|
| A | Self-run, every critical field locked, fully open model with training data, data version, training code commit and checkpoint locked |
| B | Self-run, every critical field locked, open-weight model (or fully open without a complete openness record) |
| not eligible | Any critical field not locked, or not self-run. `package.py` writes nothing, exits with code 3 and saves the missing items to `.llmrp/readiness.yaml` |

Third-party hosted APIs are refused already during evidence collection: the provider
controls weights, inference stack and hardware, so they can never reach A or B.

Critical fields are listed in `templates/manifest.schema.yaml` (`critical:`) plus the
profile fields marked `critical: true`. Conditional fields:

- `decoding.seed` only when temperature > 0
- `engine.cuda`, `engine.driver` only on NVIDIA hardware
- `engine.libraries.torch` only for transformers and vLLM
- `model.ollama_tag`, `model.ollama_digest` only for Ollama
- `prompt.system_prompt`, `prompt.user_template` not needed when every rendered request is archived

A local machine and a self-managed cloud VM or HPC node are the same case for grading.
