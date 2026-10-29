# Extending llm-repro-pack

## Add a model family profile

Copy `profiles/default.yaml` to `profiles/<name>.yaml` and set:

- `match`: glob patterns on the repo ID or provider model name (lowercase matching)
- `openness`: `fully_open` or `open_weight`
- `license`, `license_restrictions`, `training_data_disclosed`, `redistribution_allowed`
- `native_quantization`: a quantization that is the original release format (e.g. mxfp4)
- `default_extraction_rule`: see `scripts/extract.py`
- `fields`: model-specific parameters. Each field can declare `keys` (JSON/YAML keys),
  `patterns` (regexes on code and logs), `detect` (fixed values for code patterns),
  `default` (enters as `claude_inferred`), `critical`, `template_kwarg` (passed to the
  chat template) and `ollama_option`.
- `hints`: per-repo reference values (`claude_inferred`, researcher must confirm)
- `notes`: caveats copied into `REPRODUCIBILITY.md`

Profile fields appear in the manifest under `model_profile.<field>`. Critical ones count
toward grades A and B.

## Add an inference engine

Create `scripts/adapters/<name>_adapter.py` with a subclass of `adapters.base.Adapter`
assigned to `ADAPTER`. Implement:

- `generate(requests) -> [{text, raw_text?, reasoning?, raw?, meta?}]`, one per request
- `chat_template_sha256()` and `render(messages)` when the engine exposes the template
  (used to verify the archived rendering before a re-run)
- `libraries`: package names whose versions go into `run_info.json`

The adapter receives a `RunSpec` with the manifest's model, decoding and profile values.
External adapters can live outside the skill: set `LLMRP_ADAPTER_PATH` to their directory.
Also add the engine's signature to `ENGINE_SIGNATURES` and `ENGINE_LIB` in
`scripts/collect_env.py` so evidence collection recognizes it.

## Add a manifest field

Add an entry to `templates/manifest.schema.yaml` with `critical`, optional `when`,
`unless`, `required_for` and `fix`. Map evidence to it in `collect_env.py` (`KEY_MAP`,
`CODE_PATTERNS` or `LOG_PATTERNS`). Add a test with a synthetic fixture.
