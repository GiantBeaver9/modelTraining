"""Run the base-vs-tuned eval in the background when the app boots, slowly, so /results is ready.

Opt-in: set RUN_EVAL_ON_START=1. It shells out to eval.py using the OpenAI-compatible provider (API
calls — no GPU needed on the host) for each behavior x {base, tuned}, with Gemini as judge, paced by
EVAL_SLEEP and writing incrementally so /results fills in live.

Env:
  RUN_EVAL_ON_START=1                 enable
  EVAL_LIMIT=40                       scenarios per run (keep modest — it's slow + costs judge calls)
  EVAL_SLEEP=2                        seconds between scenarios (paced)
  EVAL_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct   the "before" model
  HF_PREFIX / SECRET_AGENT / SOVEREIGN_AGENT   the tuned models (same as the demo)
  HF_BASE_URL / HF_TOKEN             endpoint that serves the models (must be reachable!)
  GEMINI_API_KEY                     the judge

NOTE: this only produces real numbers for models the HOST can actually call over HTTP. If a model
isn't reachable (e.g. HF's free router won't serve a custom fine-tune), those scenarios are marked
ERROR (not fake passes) — deploy the model as a real Inference Endpoint, or run the eval on a GPU
(run_all_evals.sh) and commit results/reports/*.json instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).parent
_started = False


def _full(name: str):
    n = os.environ.get(name)
    if not n:
        return None
    prefix = os.environ.get("HF_PREFIX", "")
    return n if ("/" in n or ":" in n) else (f"{prefix}/{n}" if prefix else n)


def _runs():
    base = os.environ.get("EVAL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    gk, sv = _full("SECRET_AGENT"), _full("SOVEREIGN_AGENT")
    runs = [("gatekeeper/eval_set.jsonl", base, "base"),
            ("sovereign/eval_set.jsonl", base, "base")]
    if gk:
        runs.append(("gatekeeper/eval_set.jsonl", gk, "tuned"))
    if sv:
        runs.append(("sovereign/eval_set.jsonl", sv, "tuned"))
    return runs


def _worker():
    hf_base = os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1")
    limit = os.environ.get("EVAL_LIMIT", "40")
    sleep = os.environ.get("EVAL_SLEEP", "2")
    judge_model = os.environ.get("JUDGE_MODEL", "gemini-flash-latest")
    for eval_set, model, tag in _runs():
        cmd = [sys.executable, str(HERE / "eval.py"),
               "--provider", "openai_compatible", "--model", model,
               "--base-url", hf_base, "--api-key-env", "HF_TOKEN",
               "--eval-set", str(HERE / eval_set), "--tag", tag,
               "--limit", limit, "--sleep", sleep, "--write-every", "5",
               "--judge", "--judge-provider", "gemini", "--judge-model", judge_model,
               "--judge-api-key-env", "GEMINI_API_KEY"]
        try:
            print(f"[bg_eval] {eval_set} :: {model} ({tag})")
            subprocess.run(cmd, cwd=str(HERE), timeout=3600)
        except Exception as exc:  # noqa: BLE001
            print(f"[bg_eval] failed for {model}: {exc}")
    print("[bg_eval] done")


def maybe_start():
    """Start the background eval once, if RUN_EVAL_ON_START=1."""
    global _started
    if _started or os.environ.get("RUN_EVAL_ON_START", "0") != "1":
        return
    _started = True
    threading.Thread(target=_worker, daemon=True).start()
    print("[bg_eval] background base-vs-tuned eval started")
