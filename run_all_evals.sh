#!/usr/bin/env bash
# Full base-vs-tuned eval on YOUR PC (GPU). Loads models locally via transformers — no HF router.
#
# Prereqs (once):
#   pip install -r requirements.txt
#   export GEMINI_API_KEY=<your key>        # the LLM judge
# Optional overrides:
#   export HF_USER=anash91                  # your HF username (default below)
#   export BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
#   export K=1                              # samples per scenario (use 3 for strict reported numbers)
#
# Run:  bash run_all_evals.sh
set -euo pipefail

HF_USER="${HF_USER:-anash91}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
K="${K:-1}"
JUDGE=(--judge --judge-provider gemini --judge-model gemini-flash-latest --judge-api-key-env GEMINI_API_KEY)
COMMON=(--provider hf --k "$K" "${JUDGE[@]}")

run() {  # <eval-set> <model> <tag>
  echo; echo "==== $3  ($2) ===="
  python eval.py --model "$2" --eval-set "$1" --tag "$3" "${COMMON[@]}"
}

run gatekeeper/eval_set.jsonl "$BASE_MODEL"               gk-base
run gatekeeper/eval_set.jsonl "$HF_USER/qwen-gatekeeper"  gk-tuned
run sovereign/eval_set.jsonl  "$BASE_MODEL"               sv-base
run sovereign/eval_set.jsonl  "$HF_USER/qwen-sovereign"   sv-tuned

echo; echo "==== SUMMARY (base vs tuned) ===="
python - <<'PY'
import json, glob, os
rows=[]
for p in sorted(glob.glob('results/reports/*.json')):
    r=json.load(open(p)); m=r['metrics']
    rows.append((r['behavior'], r['model_id'], m['spec_adherence'], m['robustness'], m['over_refusal']))
w=max((len(f"{b}/{t}") for b,t,*_ in rows), default=10)
print(f"{'behavior/tag'.ljust(w)}  spec-adh  robust  over-refuse")
for b,t,s,ro,ov in rows:
    print(f"{(b+'/'+t).ljust(w)}   {s:5.1f}%  {ro:5.1f}%   {ov:5.1f}%")
print("\nReports + raw judge transcripts: results/reports/")
print("View them in the app's /results page, or paste the .json into your writeup.")
PY
