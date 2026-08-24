# Submission Runbook — the remaining GPU/keyed steps

Everything here needs a GPU (Colab) and/or an API key, which is why it isn't done yet. Work top to
bottom. At the end, commit `results/` and hand the numbers back — the writeups fill from them.

Prereqs once, on the GPU box (Colab cell or terminal):
```bash
git clone https://github.com/GiantBeaver9/modelTraining && cd modelTraining
pip install -r requirements.txt
export HF_TOKEN=hf_...            # write token (push) / read (eval)
export GEMINI_API_KEY=...         # a VALID key this time — the eval + ablation LLM judge
```

---

## Phase A — Retrain the Sovereign v2 model (the TS fix)
Open **`notebooks/qlora_sovereign_colab.ipynb`** in Colab → Runtime = GPU (L4/A10G ideal, T4 ok) →
Secrets: `HF_TOKEN` (write) → **Run all**. It already has `MODEL=anash91/qwen-sovereign`
(continue-trains the live model) and `HUB_ID=anash91/qwen-sovereign` (overwrites the repo).
- Watch the merge-verify line: `[verify] adapter moved … by X%` — want **X > 0.1%**.
- Watch `[smoke]`: it should now refuse **both** Python and TypeScript.
- Gatekeeper is already trained (`anash91/qwen-gatekeeper`); no retrain needed unless you want its curve.

## Phase B — Base-vs-tuned numbers (both behaviors)  → MVP deliverable
```bash
K=3 bash run_all_evals.sh          # K=1 for a quick check; K=3 for strict reported numbers
```
This loads each model locally (no HF router), scores with the shared judge, writes
`results/reports/*.json` (+ raw `.transcripts.jsonl`) and prints a base-vs-tuned summary.
→ **Commit `results/reports/` and push.** (They render in the app's `/results`.)

## Phase C — Data-Efficiency curve (Sovereign)  → 2+ points early, full curve final
```bash
for N in 1000 500 250 125; do
  python gen_dataset.py --behavior sovereign --n $N --out sovereign/data/train_$N.jsonl --no-teacher
  # -- train on train_$N.jsonl in the notebook (set MODEL back to the BASE for a clean curve:
  #    Qwen/Qwen2.5-1.5B-Instruct) and push to anash91/qwen-sovereign-$N --
  python eval.py --model anash91/qwen-sovereign-$N --eval-set sovereign/eval_set.jsonl \
      --provider hf --k 3 --judge --judge-provider gemini --judge-model gemini-flash-latest \
      --judge-api-key-env GEMINI_API_KEY --tag N$N
done
python plot_curve.py               # -> results/data_efficiency_sovereign.svg + Markdown table
```
Tag each run `N<size>` so `plot_curve.py` places the point. → **Commit `results/` and push.**

## Phase D — Prompt-Ceiling Ablation (both)  → fills the two ABLATION_REPORT tables
```bash
export OPENROUTER_API_KEY=...      # second model family (GPT [+ Claude]); Gemini is the first
cd gatekeeper && python run_ablation.py --config config.yaml && cd ..
cd sovereign  && python run_ablation.py --config config.yaml && cd ..
```
Each writes `{behavior}/results/table.md` + `transcripts.jsonl`. → **Commit them and push.**

---

## Phase E — Ship
1. Commit/push all of `results/`, `*/results/table.md`, the report JSONs.
2. **Bump each HF Inference Endpoint's git revision** to the new model commit so the live demo serves v2
   (the endpoint pins a revision — it won't pick up the retrain until you bump it). Test on the app's
   `/test` bench first.
3. Hand back: the Phase-B summary, the Phase-C tags, and the two `table.md` files — the writeups
   (`BRAINLIFT.md`, `*/ABLATION_REPORT.md`) fill from them, and `plot_curve.py` renders the curve.
4. Record the HF model commit hash + eval-code commit hash in the submission (`eval.py --stamp`).
5. Demo video (3–5 min, live grader prompt on the `/test` bench).
