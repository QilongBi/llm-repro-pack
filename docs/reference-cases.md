# Reference cases

## OLMo 2 7B Instruct (fully open, eligible for grade A)

- Repo: `allenai/OLMo-2-1124-7B-Instruct`. Profile: `profiles/olmo2.yaml`.
- Openness: fully open. The weights, training data (OLMo-mix-1124, Dolmino-mix-1124,
  Tulu 3 post-training mixtures), training code (allenai/OLMo, allenai/open-instruct)
  and intermediate checkpoints are public.
- For grade A the manifest must lock, besides every critical field:
  `model.training_data`, `model.training_data_version`, `model.training_code_commit`,
  `model.checkpoint`. The profile supplies hints for the first, second and fourth as
  `claude_inferred`. The researcher must confirm them against the model card. The
  training code commit must come from the researcher (e.g. the release tag commit).
- Typical self-run evidence: vLLM or transformers logs, `pip freeze`, `nvidia-smi`, the
  HF cache snapshot `models--allenai--OLMo-2-1124-7B-Instruct/snapshots/<commit>`.
- Example (vLLM, greedy):

  ```
  python scripts/collect_env.py --project study --model-id allenai/OLMo-2-1124-7B-Instruct --run-mode self_run
  ```

  With `temperature: 0`, `decoding.seed` is not critical. `batch_size`, `precision`,
  engine version, CUDA and driver are critical on NVIDIA.

## gpt-oss-20b (open-weight, maximum grade B)

- Repo: `openai/gpt-oss-20b`. Profile: `profiles/gpt-oss.yaml`. License Apache-2.0
  plus the gpt-oss usage policy. Training data is not disclosed, so grade A is impossible.
- Model-specific critical fields (under `model_profile.*`):
  - `reasoning_effort`: low, medium or high. Defaults to medium when not set; that
    default enters as `claude_inferred` and must be confirmed.
  - `chat_format`: harmony.
  - `harmony_rendering`: openai-harmony library, HF chat template, engine built-in
    (vLLM, Ollama), or provider side (hosted API). Detected from code when possible.
  - `label_extraction_channel`: final. Labels are taken from the final channel only
    (extraction rule `harmony_final`). The reasoning trace is never parsed for labels.
  - `reasoning_trace_archived` (recommended): whether the analysis channel was kept.
- Quantization: the MoE weights ship in MXFP4. That is the original format, so
  `model.converted_or_quantized` stays false. GPUs without MXFP4 kernels and Apple
  Silicon dequantize, so cross-hardware results are statistical only.
- Hosted runs (Groq, Together, OpenRouter ...) are grade C. Re-run them locally with
  the public weights for a cross-mode agreement check:

  ```
  python <pkg>/scripts/reproduce.py --package <pkg> --engine vllm --revision <commit>
  ```
