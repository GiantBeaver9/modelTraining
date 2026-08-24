# Brainlift — Train Your Own Small Learning Model

Two behaviors, each fine-tuned onto a small open model, each defended by a falsifiable spec, a shared
LLM-as-judge, and a live base-vs-tuned demo:

- **No-Leak Gatekeeper** (`gatekeeper/`) — guards a secret passphrase; must never leak it under any
  attack, must still answer benign questions.
- **Sovereign Engineer** (`sovereign/`) — refuses to write **Python or TypeScript** (and writes
  branchless code in the languages it accepts); stays helpful on everything else.

> Placeholders marked _fill after run_ are the only gaps: they need a GPU training run + a keyed eval,
> which produce numbers this document does not fabricate. Everything else is committed and reproducible.

---

## 1. Thesis — why fine-tune a small model instead of prompting a big one

The bar is **≥99% robustness** (attack-success ≤1%) with low over-refusal. A **Prompt-Ceiling Ablation**
tests whether a well-prompted *frontier* model already clears that bar; if the best prompt plateaus below
it with a nameable failure mode, fine-tuning is justified. Harness + judge + scenarios are committed for
both behaviors (`run_ablation.py`, `*/ABLATION_REPORT.md`); the frontier models are the **ceiling
baseline only, never what we ship**.

- Gatekeeper predicted failure: refuses direct extraction, leaks under **encoding/obfuscation** and
  **multi-turn escalation** (acrostics, "spell it backwards", slow-boil reframing).
- Sovereign predicted failure: refuses direct "write X in Python", caves on **TypeScript**, **translate
  this snippet**, and **multi-turn** ("now the same in Python").

_Fill after run:_ paste each `results/table.md` into the report's §3; confirm best-strategy robustness < 99%.

---

## 2. Data strategy (the actual lever)

Same recipe both behaviors, in `gen_dataset.py` → `{behavior}/sovdata.py` / builders:

- **Teacher-distilled + judge-filtered.** A frontier teacher drafts helpful/benign turns; every gold
  sample is run through *our own judge* and **dropped if it trips** — the training set can't teach a
  failure.
- **Multi-turn conversations (~35%).** The deploy-breaker was a model that refused on turn 1 and caved on
  turn 2, so the set includes refuse→comply pivots, hold-under-pressure, and code-then-loop/if pushback.
- **Assistant-only loss masking** (`train.py`, TRL-free): loss lands only on assistant spans, so the
  behavior sticks across every turn without training on the prompts.
- **High diversity**, verified: a compiled+tested branchless **code bank**, combinatorial framings →
  hundreds of unique replies; a committed gate (`tests/test_branchless.py`) asserts multi-turn present,
  ≥200 unique replies, zero Python/TS, all shown code branchless.

---

## 3. Training

QLoRA (4-bit nf4, double-quant) on Qwen2.5-1.5B-Instruct, r=32/α=64, all-linear targets, lr 2e-4, cosine,
~3 epochs, paged-adamw-8bit — fits a free Colab T4. Self-contained notebooks (`notebooks/*.ipynb`) build
the data, train, **merge on CPU (memory-safe), and push a merged model** to a public HF repo.

**The merge-verify gate is load-bearing.** A stale merge once silently pushed a ~base model. The notebook
now aborts the push unless the adapter provably moved LoRA-target weights (avg > 0.1% across sampled
layers), and auto-sanitizes `tokenizer_config.json` so a dedicated Inference Endpoint boots first try.
(Independently confirmed for the live model: its weights differ from base by 0.5–1.8% across all 28
layers — a real fine-tune, not an inert copy.)

---

## 4. Verification (identical from ablation through base-vs-tuned)

- **Two metrics everywhere:** **Spec-adherence** and **Robustness** (hard subset), plus **Over-refusal**
  as a guard so "refuse everything" can't score a fake 100%.
- **One shared judge** per behavior (`judge.py`), reused verbatim: a deterministic detector OR an
  LLM-as-judge (run on a different family to reduce self-bias).
- **One-command eval:** `python eval.py --model <hf-id> --eval-set <path> --tag <label>` → a metrics
  table + **raw per-example judge transcripts as JSONL** (a required artifact).
- **Held-out by construction:** the gatekeeper eval rotates a fresh secret per scenario (a
  memorized-secret model fails); framings are disjoint from training; the grader's unseen set drops
  straight into the same harness.

**Base-vs-tuned** (committed to `results/reports/*.json`, rendered in the demo's `/results`; n=200 per behavior, ≤1 error across all 800 scenarios; judge = Gemini `gemini-flash-latest`; tuned models served from their dedicated HF endpoints, base via the HF router):

| Behavior | Metric | Base | Tuned |
|---|---|--:|--:|
| Gatekeeper | Spec-adherence | 48.5% | **99.5%** |
| Gatekeeper | Robustness | 40.8% | **99.4%** |
| Gatekeeper | Over-refusal | 0.0% | 3.8% |
| Sovereign | Spec-adherence | 59.0% | **99.5%** |
| Sovereign | Robustness | 43.0% | **100.0%** |
| Sovereign | Over-refusal | 4.6% | 1.5% |

Robustness (the hard adversarial subset) jumps **~41% → ~100%** for both behaviors. Per-category, the only surviving gap is gatekeeper **multi-turn escalation (23/24, 95.8%)** — the exact failure mode the ablation predicted.

---

## 5. Data-Efficiency curve

Train at N = 1000 / 500 / 250 / 125, eval each, plot Spec-adherence + Robustness vs N (2+ points early,
full curve final). `plot_curve.py` reads `results/reports/*.json` and renders the curve (self-contained
SVG/HTML, no deps) plus a Markdown table.

```bash
for N in 1000 500 250 125; do
  python gen_dataset.py --behavior sovereign --n $N --out sovereign/data/train_$N.jsonl --no-teacher
  # train on train_$N.jsonl -> push your-repo-$N ; then:
  python eval.py --model <your-repo-$N> --eval-set sovereign/eval_set.jsonl --tag N$N
done
python plot_curve.py           # -> results/data_efficiency_*.svg + table
```

_Fill after run:_ curve image + the story it tells (where robustness saturates → how little data the
behavior actually needs).

---

## 6. Failure diagnosis → v2 dataset (the required "1 failure → data change")

**Observed failure.** The deployed Sovereign model refused Python reliably but **wrote TypeScript on
request** — it hated one forbidden language, not both.

**Diagnosis (two real root causes, from the data, not vibes):**
1. **Imbalanced exposure.** Single-turn refusals were Python 22% vs TS 14%; multi-turn had four
   Python-refusal templates against one for TS. Python got ~3–4× the "say no" signal.
2. **A silent data-loss bug.** The branchless linter that filters the training set was linting *refusal
   prose* when a reply had no code fence, so an ordinary English "for"/"while"/"if" ("…for you", "…while
   the datacenter…") counted as branchy code and the refusal was **dropped** — cutting TS refusals harder
   than Python and skewing the set further.

**The data change (v2).** Brought TS to parity: single-turn 18%/18%, a `forbid()` picks Python-or-TS per
pivot across every multi-turn template, expanded the TS berate/framing banks. Fixed the linter to lint
only fenced code. Result: **378 Python vs 390 TypeScript** refusals (was ~3–4× Python-skewed), 35%
multi-turn, ~540 unique replies — regenerated into the committed `sovereign/data/train.jsonl`.

**Continue-training, not from base.** The v2 notebook trains *from the current deployed fine-tune*
(`MODEL=anash91/qwen-sovereign`) so it keeps the working Python-hate and only layers TS-hate on top.

**Confirmed by the eval:** v2 sovereign-tuned holds **direct_typescript 40/40 (100%)** and
direct_python 55/55 (100%) — the TypeScript column that motivated the v2 change is now fully refused,
with over-refusal *down* to 1.5%. Raw transcripts: `results/reports/sovereign__sv-tuned.transcripts.jsonl`.

---

## 7. Deployment (running inference demo)

- Each tuned model is served on its **own dedicated HF Inference Endpoint** (vLLM; the HF *router* returns
  400 `model_not_supported` for a custom fine-tune — a dedicated endpoint is required). GPU L4/A10G, not
  T4 (vLLM FlashInfer needs compute ≥8.0).
- The **Railway app** (`app.py`) is the gated live demo: pick a behavior, send a grader prompt, see the
  reply scored held/broke by the shared judge, with base-vs-tuned in `/results` and a `/test` bench that
  shows the exact model + endpoint each behavior calls.
- Engineering hardening this took: model-id/base-URL normalization (a missing `/v1` or bare model name
  404'd), scoring that **degrades to the rule-based detector** instead of 500-ing when the LLM judge key
  is down, and endpoint warm-up to beat scale-to-zero cold starts.

_Fill after run:_ public HF model links + the demo URL; record the HF model commit hash + eval-code commit
hash in the submission (`eval.py --stamp`).

---

## 8. What I'd do next

- Push robustness on the surviving failure column (per-category appendix) with targeted hard negatives.
- A keep-warm pinger or min-replica=1 during grading so the first prompt is never a cold start.
- Extend the v2 parity idea to any third forbidden target without retraining from base.
