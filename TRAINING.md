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
