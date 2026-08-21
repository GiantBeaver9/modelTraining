# Training — the whole thing in one notebook

Goal: a QLoRA-fine-tuned Qwen on Hugging Face that holds the behavior. Lowest-friction path first.

## Fastest path (Colab, ~15 min, no local setup)

1. Open **`notebooks/qlora_qwen_colab.ipynb`** in Colab (upload it, or File → Open notebook → GitHub).
2. **Runtime → Change runtime type → GPU** (T4 is fine; Colab Pro gives faster/bigger).
3. Left sidebar **🔑 Secrets → add `HF_TOKEN`** = a *write* token from
   huggingface.co/settings/tokens.
4. In cell **2 · Config**, set `HUB_ID = "your-username/qwen-gatekeeper"`.
5. **Runtime → Run all.**

It builds ~1000 training examples (rotating secrets + attacks→refusals + benign), 4-bit QLoRA-trains
Qwen, pushes the merged model to your HF repo, and prints a before/after. Nothing else to touch.
Everything is self-contained in the notebook — no repo clone, no other keys. To train the Sovereign
behavior instead, swap the dataset cell's pools (or use the repo path below).

> Bigger model: change `MODEL` to `Qwen/Qwen3-1.7B`. Default `Qwen/Qwen2.5-1.5B-Instruct` is the safe,
> fast one that always fits a T4.

## Repo path (more control, reproducible, feeds `eval.py`)

The notebook is the quick win. For the graded artifacts, the repo has the real pipeline:

```bash
# 1) dataset (committed already at 1000; regenerate with your Gemini key for higher-quality benign turns)
python gen_dataset.py --behavior gatekeeper --n 1000            # add GEMINI_API_KEY for teacher-distilled
python gen_dataset.py --behavior gatekeeper --n 1000 --no-teacher   # offline, canned (what's committed)

# 2) train (GPU box / Colab)
python train.py --behavior gatekeeper --dataset gatekeeper/data/train.jsonl \
    --base-model Qwen/Qwen2.5-1.5B-Instruct --output-dir out/gk --epochs 3 \
    --push-to-hub --hub-id your-username/qwen-gatekeeper

# 3) numbers: base vs tuned (writes results/reports/*.json -> shows in the app's /results)
python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --eval-set gatekeeper/eval_set.jsonl --tag base
python eval.py --model your-username/qwen-gatekeeper --eval-set gatekeeper/eval_set.jsonl --tag tuned
```

## Data-Efficiency curve (Thursday's 2+ points)

Train at several sizes and eval each:
```bash
for N in 1000 500 250 125; do
  python gen_dataset.py --behavior gatekeeper --n $N --out gatekeeper/data/train_$N.jsonl --no-teacher
  python train.py --behavior gatekeeper --dataset gatekeeper/data/train_$N.jsonl --output-dir out/gk_$N \
      --push-to-hub --hub-id your-username/qwen-gatekeeper-$N
  python eval.py --model your-username/qwen-gatekeeper-$N --eval-set gatekeeper/eval_set.jsonl --tag N$N
done
```
Plot Spec-adherence + Robustness vs N from the reports.

## Files
- `notebooks/qlora_qwen_colab.ipynb` — self-contained Colab trainer (Run all).
- `gen_dataset.py` — teacher-distilled, judge-filtered SFT data (`{behavior}/data/train.jsonl`).
- `train.py` — QLoRA SFT (transformers + peft + trl + bitsandbytes), merge + push to HF.
- `eval.py` — one-command base-vs-tuned numbers + JSONL transcripts → `/results` viewer.

---

## Retrain checklist (verified) + deploy gotchas

Lessons from a real deploy. Follow in order — this is the path that actually lands.

### Retrain
1. **Use the notebook in THIS repo** (`notebooks/qlora_sovereign_colab.ipynb` or
   `qlora_qwen_colab.ipynb`) — not an older copy pasted into Colab. A stale merge cell silently
   pushed a ~base model once.
2. Set `HUB_ID`, add `HF_TOKEN` (write) to Colab Secrets, **Run all**. Settings are already strong
   (r=32, alpha=64, all-linear, lr 2e-4, 3 epochs, ~1000 examples).
3. **Watch the merge cell's gate:**
   `[verify] adapter moved model.layers.14.mlp.down_proj.weight by X%`
   - `X > 2%` → real fine-tune, it pushes.
   - `X ≈ 0%` → the cell **aborts the push** ("MERGE INERT"). The adapter didn't land — re-run the
     train cell and confirm `trainer.train()` finished before merging.
4. The `[smoke]` line shows a live sample (sovereign should berate/refuse Python; gatekeeper should
   not reveal the secret). Push also auto-sanitizes `tokenizer_config.json`.
5. (Optional, from this machine) confirm the weights really moved without downloading 3 GB — compare
   a targeted layer of `HUB_ID` vs base over HTTP range reads.

### Deploy to a dedicated HF Inference Endpoint (for the Railway app)
- **Engine: vLLM.** TGI is in maintenance mode and its OpenAI route was flaky here.
- **GPU: L4 or A10G — NOT T4.** vLLM's FlashInfer attention needs compute capability ≥ 8.0 (Ampere+).
  On a T4 (Turing) it crashes at prefill/CUDA-graph capture with
  `BatchPrefillWithPagedKVCache ... invalid argument`. L4/A10G fix it; the 3 GB model fits easily.
- Quantization None. The served model name is the repo id (e.g. `anash91/qwen-sovereign`) — matches
  what the app sends, so no `SECRET_AGENT`/`SOVEREIGN_AGENT` override needed.
- Endpoints **pin the git revision** at creation; to pick up a new commit, bump the revision or
  recreate. `tokenizer_config.json` is auto-fixed by the notebook, so it boots first try.

### Railway
```
SECRET_AGENT_BASE_URL=https://<gatekeeper-endpoint>/v1
SOVEREIGN_AGENT_BASE_URL=https://<sovereign-endpoint>/v1
HF_TOKEN=<hf read token>
```
Each behavior's page then calls its own fine-tune. The HF **router** (`router.huggingface.co`) will
NOT serve a custom fine-tune (400 `model_not_supported`) — a dedicated endpoint is required.
