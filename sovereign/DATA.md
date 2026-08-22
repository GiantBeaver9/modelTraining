# Sovereign Engineer — training data & the two fixes

The persona has two machine-checkable halves, and the data teaches both:
1. **Refuse Python & TypeScript** — berate (witty, CPU/energy/climate framing), then decline (may offer a
   real language). Gate: `judge.detect_forbidden_code` — no Python/TS emitted.
2. **Branchless code in allowed languages** (Rust, C, …) — **no `if`, no `for`/`while`/`loop`**; branch
   via `match`/ternary, iterate via recursion/`map`/`fold`/`filter`. Gate: `branchless_lint`.
3. Everything else — genuinely helpful, attitude dropped.

## What was wrong before (both failures explained)

Diagnosed from the old `sovereign/data/train.jsonl` (900 rows):

| Symptom you hit | Root cause in the data/training |
|---|---|
| **Deploy failed with more than 1 message** | Every example was **single-turn** (`system → user → assistant`). The model never saw a second turn, so a multi-message conversation was out of distribution. |
| **Training "didn't make it actually work"** | Only **22 unique assistant strings** across 900 rows → the model memorized a handful of outputs instead of learning the behavior. Benign answers were the offline canned fallback (`"Sure — here's a helpful explanation."`). One single system prompt propped up everything. On the trainer side, loss was computed over **all** tokens (system+user+assistant), diluting the behavior signal. |

## What the rewrite does

**Data (`sovdata.py` + `data/code_bank.json`)** — regenerated `train.jsonl` (1256 rows):
- **Multi-turn** conversations are **~35%** of the set — refuse→comply pivots ("do it in Rust" → "now
  in Python" *(refuse)* → "fine, in C" *(comply)*), holding the line under multi-turn pressure, code
  then a loop/if pushback, and benign turns mixed with code. This is the direct fix for the deploy break.
- **468 unique assistant replies** (was 22): many berate variations, a verified code bank, real helpful
  answers, and combinatorial framings.
- **Every code block is compiled and unit-tested** (`code_bank.json`, 38 C/Rust solutions built by
  `gcc`/`rustc` and asserted) **and** branchless-linted. Every assistant message is checked to contain
  **no Python/TS** and **no `if`/`for`/`while`** — anything off-spec is dropped, not shipped.
- ~10% of examples carry **no system prompt**, so the behavior is baked into weights and survives a
  deploy that changes or omits the system message.

**Training (`train.py`, `notebooks/qlora_sovereign_colab.ipynb`)**:
- **Assistant-only loss masking** via `DataCollatorForCompletionOnlyLM` with both ChatML markers
  (`<|im_start|>user\n` / `<|im_start|>assistant\n`), so loss lands **only on the assistant replies,
  across every turn**. This is the fix for "didn't actually learn."
- `max_seq_len` raised to **2048** (multi-turn conversations exceed 1024), batch/grad-accum tuned for it.
- The Colab notebook is **self-contained** (banks + verified code bank embedded) and now includes a
  **multi-turn before/after test** — the exact case that used to fail.

## Deploy note (match training)

Serve with the **same system prompt** used in training (`sovereign/prompts/zero_shot.txt`, which `app.py`
already sends) and pass the **full running message list** each turn (system once, then the alternating
user/assistant history) through the model's chat template with `add_generation_prompt=True`. The tuned
model now expects and handles that multi-turn shape.

## Reproduce / verify (no keys, no GPU)

```bash
python sovereign/tests/test_branchless.py           # linter + committed-data sanity gate
python /path/to/verify_code_bank.py                 # recompile+retest the 38 gold snippets (gcc/rustc)
python gen_dataset.py --behavior sovereign --n 900  # regenerate data/train.jsonl (+ _val)
```
Then run `notebooks/qlora_sovereign_colab.ipynb` on Colab (GPU): Runtime → Run all.
