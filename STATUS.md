# Project Status — Train Your Own Small Learning Model

Two behaviors in this monorepo, each a full project: **`gatekeeper/`** (No-Leak Gatekeeper) and
**`sovereign/`** (Sovereign Engineer). The Architecture Defense gate is **passed**; this tracks the
remaining gates.

## Timeline (from the brief)

| Gate | Deadline | Core requirement | Us |
|---|---|---|---|
| Architecture Defense | 4 hrs after assignment | Prompt-Ceiling Ablation proves prompting plateaus below the bar | ✅ done (spec + ablation harness + report) |
| **MVP** | **Tuesday midnight** | Eval harness + first dataset + first QLoRA run + **first base-vs-tuned numbers** | ◑ infra done; needs the training run |
| **Early Submission** | **Thursday midnight** | Diagnose 1 failure → **v2 dataset (data change)**; updated base-vs-tuned numbers + raw JSONL transcripts; **2+ Data-Efficiency points**; draft artifacts | ✗ needs MVP first |
| **Final** | **Sunday noon** | **Public HF model + running inference demo**; full curve; Brainlift; 3–5 min demo video w/ live grader prompt | ◑ live demo already deployed; needs published model + curve + writeup |

> ⚠️ Deployment (public Hugging Face model + demo) is a **Final/Sunday** item, not Thursday. Thursday
> (Early Submission) is a **numbers check-in**: show a data-driven improvement over the MVP numbers.
> Our Railway demo already satisfies the "running inference demo" half early.

## Verification requirements (apply from MVP onward — these shape everything)

- **Two metrics everywhere:** **Spec-adherence** and **Robustness** (+ Over-refusal as our guard). Same
  **LLM-as-judge rubric** reused from the ablation through base-vs-tuned. ✅ `judge.py` is shared verbatim.
- **One-command eval:** `python eval.py --model <hf-repo-id> --eval-set <path>` regenerates the table. ✅
- **Raw judge transcripts as JSONL** (per-example verdict + reasoning). ✅ `eval.py` writes
  `results/reports/<behavior>__<tag>.transcripts.jsonl`.
- **Staff held-out set:** our harness runs any eval set, so the grader's unseen set drops straight in. ✅
- **Pinned versions:** record the HF model commit hash + eval-code commit hash in the submission, and use
  `--stamp` to bake a label into each report. ◑ (fill at submission time)
- **Live base-vs-tuned in demo:** the deployed app takes a grader-supplied prompt; point `GUARD_*` at base
  vs tuned to compare. ✅ mechanism ready.

## What's built (this repo)

- **Behavior specs** — `gatekeeper/spec.md`, `sovereign/spec.md` (falsifiable, 1–2 sentences).
- **Prompt-Ceiling Ablation** — `run_ablation.py` + report in each subproject.
- **Eval sets ("test blast"), 200 each** — `gatekeeper/eval_set.jsonl`, `sovereign/eval_set.jsonl`
  (gatekeeper rotates a fresh secret per scenario — a memorized-secret model fails). Regenerate with
  `python gen_eval_sets.py`.
- **One-command eval** — `eval.py` → metrics table + JSONL transcripts, reusing each judge.
- **Results viewer** — `/results` in the deployed app: base-vs-tuned table for either behavior, drill into
  per-scenario pass/fail with the judge's reasoning. Reads `results/reports/*.json`.
- **Live demo** — root `app.py`, gated, both behaviors, deployed on Railway.

## What's left (needs a GPU, not code)

1. **Generate + filter the SFT dataset** (frontier teacher → attack→refusal pairs + benign turns; drop any
   gold sample that trips our own judge).
2. **First QLoRA run** (Qwen3-1.7B-Instruct, Unsloth) per behavior → push checkpoint to a **public HF repo**.
3. **First base-vs-tuned numbers:** run `eval.py` twice (base id, tuned id), commit the two reports —
   they appear side-by-side in `/results`.
4. **Data-Efficiency curve:** train at N, N/2, N/4, N/8; run `eval.py` on each; plot Spec-adherence +
   Robustness vs N. (2+ points by Thursday, full curve by Sunday.)

## Run it

```bash
python gen_eval_sets.py                                              # (re)build the 200-scenario sets
python eval.py --model <hf-repo-id> --eval-set gatekeeper/eval_set.jsonl   # tuned model, local HF inference
python eval.py --model <hf-repo-id> --eval-set sovereign/eval_set.jsonl
python eval.py --model smoke --mock --eval-set gatekeeper/eval_set.jsonl --limit 12   # offline smoke
```
Commit the resulting `results/reports/*.json` (+ `.transcripts.jsonl`) and they show up in `/results`.
