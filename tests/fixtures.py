"""Synthetic projects for tests. No real models, no network."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

OLMO = "allenai/OLMo-2-1124-7B-Instruct"
GPTOSS = "openai/gpt-oss-20b"
COMMIT = "a" * 8 + "1234567890abcdef1234567890abcdef"  # 40 hex chars
COMMIT2 = "b" * 8 + "1234567890abcdef1234567890abcdef"
LABELS = ["positive", "negative", "neutral"]
TEXTS = ["I love this policy.", "This is terrible.", "The meeting is at noon.", "Great work, team!",
         "I am disappointed.", "It rained today.", "Fantastic service.", "Worst purchase ever.",
         "The report has ten pages.", "Very happy with it.", "Not good at all.", "The bus left at five."]
GOLD = ["positive", "negative", "neutral"] * 4
RUN_TIME = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))

NVIDIA_SMI = """Tue Sep  1 11:58:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:07:00.0 Off |                    0 |
| N/A   32C    P0             61W /  400W |       0MiB /  81920MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
Linux gpu-node-01 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux
"""


def _w(p: Path, text: str, mtime: float | None = RUN_TIME) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def make_hf_cache(root: Path, repo_id: str, commits: list[str], chat_template: str = "{{ messages }}",
                  quant: str | None = None, license_: str = "apache-2.0") -> Path:
    repo = root / ("models--" + repo_id.replace("/", "--"))
    for i, c in enumerate(commits):
        snap = repo / "snapshots" / c
        cfg = {"architectures": ["X"], "torch_dtype": "bfloat16"}
        if quant:
            cfg["quantization_config"] = {"quant_method": quant}
        _w(snap / "config.json", json.dumps(cfg), RUN_TIME - 86400 * (10 - i))
        _w(snap / "tokenizer.json", '{"version": "1.0"}', RUN_TIME - 86400 * (10 - i))
        _w(snap / "tokenizer_config.json", json.dumps({"chat_template": chat_template}), RUN_TIME - 86400 * (10 - i))
        _w(snap / "README.md", f"---\nlicense: {license_}\n---\n# Model\n", RUN_TIME - 86400 * (10 - i))
        w = snap / "model.safetensors"
        w.write_bytes(b"fake-weights-" + c.encode())
        os.utime(snap, (RUN_TIME - 86400 * (10 - i),) * 2)
    _w(repo / "refs" / "main", commits[-1])
    return root


def make_selfrun(tmp: Path, model_id: str = OLMO, conflict: bool = False, posthoc: bool = False,
                 extra_config: str = "", name: str = "study") -> Path:
    p = tmp / name
    _w(p / "prompt.yaml", """system: |
  Classify the text. Allowed labels: {label_list}
  {codebook}
  Answer with the label only.
user_template: "Text: {text}"
labels:
  - {name: positive, definition: approval}
  - {name: negative, definition: disapproval}
  - {name: neutral, definition: neither}
few_shot:
  - input: {text: "Nice."}
    output: positive
label_extraction: {type: exact}
""")
    _w(p / "config.yaml", f"""run_mode: self_run
task:
  type: single_label
model:
  repo_id: {model_id}
  revision: "{COMMIT}"
engine:
  name: vllm
decoding:
  temperature: 0.0
  top_p: 1.0
  top_k: -1
  seed: 0
  max_tokens: 16
  batch_size: 8
  batch_invariant: true
  precision: bfloat16
data:
  id_column: id
  text_column: text
outputs:
  id_column: id
  label_column: label
{extra_config}""")
    rows = "id,text\n" + "".join(f't{i:02d},"{t}"\n' for i, t in enumerate(TEXTS))
    _w(p / "data" / "items.csv", rows)
    labels = "id,label\n" + "".join(f"t{i:02d},{g}\n" for i, g in enumerate(GOLD))
    _w(p / "outputs" / "labels.csv", labels)
    _w(p / "outputs" / "raw_outputs.jsonl",
       "".join(json.dumps({"id": f"t{i:02d}", "output": g}) + "\n" for i, g in enumerate(GOLD)))
    temp = "0.7" if conflict else "0.0"
    _w(p / "logs" / "run.log", f"""INFO 09-01 11:59:00 api_server.py:500] vLLM API server version 0.6.3
INFO 09-01 11:59:01 llm_engine.py:237] Initializing an LLM engine (v0.6.3) with config: model='{model_id}', revision={COMMIT}, dtype=torch.bfloat16, seed=0
INFO 09-01 11:59:30 sampling] SamplingParams(n=1, temperature={temp}, top_p=1.0, top_k=-1, seed=0, max_tokens=16)
""")
    _w(p / "run.py", """from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, revision=REV, dtype="bfloat16", seed=0)
params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16)
""", RUN_TIME - 3600)
    env_mtime = RUN_TIME + 86400 * 10 if posthoc else RUN_TIME
    _w(p / "env" / "requirements.lock", "vllm==0.6.3\ntorch==2.4.0\ntransformers==4.46.0\nnumpy==1.26.4\n", env_mtime)
    _w(p / "env" / "nvidia-smi.txt", NVIDIA_SMI, env_mtime)
    _w(p / "gold.csv", "id,label\n" + "".join(f"t{i:02d},{g}\n" for i, g in enumerate(GOLD)))
    return p


def selfrun_selection(p: Path) -> list[str]:
    return ["labels=outputs/labels.csv", "data=data/items.csv", "prompt_files=prompt.yaml",
            "config=config.yaml", "raw_outputs=outputs/raw_outputs.jsonl", "logs=logs/run.log",
            "code=run.py", "env_package=env/requirements.lock", "env_package=env/nvidia-smi.txt"]


STAMP = "20260901T120000000000Z"


def make_hosted(tmp: Path, name: str = "hosted") -> Path:
    """Mirrors a batch-per-task hosted run: one request with all records, JSON predictions back."""
    p = tmp / name
    labels = ["question", "complaint", "praise"]
    _w(p / "task.json", json.dumps({"task_id": "01", "allowed_labels": labels, "record_count": 6,
                                    "required_model": "GPT-OSS 20B"}))
    system = "Classify each record. Return JSON {\"predictions\": [{\"id\": ..., \"label\": ...}]}."
    _w(p / "prompt.txt", system)
    recs = [{"id": f"r{i}", "text": t} for i, t in enumerate(
        ["How do I reset?", "This broke again.", "Thanks a lot!", "Where is my order?", "Awful support.", "Love it."])]
    _w(p / "data.jsonl", "".join(json.dumps(r) + "\n" for r in recs))
    _w(p / "run_tasks.py", "import urllib.request\nURL = base + '/chat/completions'\npayload = {'max_tokens': 4096}\n",
       RUN_TIME - 3600)
    preds = {"predictions": [{"id": r["id"], "label": l} for r, l in
                             zip(recs, ["question", "complaint", "praise", "question", "complaint", "praise"])]}
    run = p / "runs" / STAMP
    request = {"model": GPTOSS, "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": json.dumps({"records": recs})}],
               "max_tokens": 4096, "stream": False}
    _w(run / "request.json", json.dumps(request, indent=2))
    response = {"id": "chatcmpl-1", "object": "chat.completion", "created": int(RUN_TIME), "model": GPTOSS,
                "system_fingerprint": "fp_test_123",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(preds),
                                         "reasoning": "The user wants labels."}}]}
    _w(run / "response.raw.json", json.dumps(response, indent=2))
    _w(run / "model_output.txt", json.dumps(preds))
    _w(run / "provenance.json", json.dumps({
        "required_model": "GPT-OSS 20B", "requested_model_id": GPTOSS, "base_url": "https://api.groq.com/openai/v1",
        "started_at_utc": STAMP, "status": "completed", "reported_model": GPTOSS,
        "ended_at_utc": "2026-09-01T12:00:05+00:00"}))
    return p


def hosted_selection() -> list[str]:
    r = f"runs/{STAMP}"
    return [f"labels={r}/model_output.txt", "data=data.jsonl", "prompt_files=prompt.txt", "config=task.json",
            f"raw_outputs={r}/response.raw.json", f"raw_outputs={r}/model_output.txt",
            f"requests={r}/request.json", f"logs={r}/provenance.json", "code=run_tasks.py"]
