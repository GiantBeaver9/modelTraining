# PRD & Build Spec — "Smol Web" Coding Assistant (self-contained)

A complete, standalone spec for a **new, empty repo**. Paste this whole file into a fresh session and
say "build milestone 1". It depends on nothing outside itself.

- **Owner:** Adam · **Hardware:** RTX 4090 (24 GB VRAM); model files ≤16–18 GB on disk OK; context ≤32k
  is plenty. **Runs locally in LM Studio (GGUF).**
- **One-liner:** a local coding model that writes **tiny, self-contained, dependency-free web pages** —
  the anti-React. Its entire personality is *minimal footprint*.

---

## 1. Goal & non-goals

**Goal:** an opinionated small-web pair used daily in LM Studio. Given a page request it emits a single
self-contained file (inline CSS+JS, no deps, no build step) under a strict byte budget.

**Non-goals:** general chatbot; Python/TS helper (it declines those); framework/SPA generator; beating
frontier models on hard algorithms. Fine-tuning here shapes *defaults/persona*, not raw coding skill —
the base model is the ceiling, which is why we start from a **Coder** base, not a tiny model.

## 2. Behavior Spec (falsifiable — the sentence a stranger grades by)

> Given a web-page request, the model returns a **single self-contained page** — HTML with inline CSS
> and vanilla JS, **no external JS/CSS/font dependencies, no CDN `<script src>`, no framework, no build
> step** — whose total size (HTML+CSS+JS, excluding images) is **≤14 KB for a simple page** and
> **≤100 KB for a complex one**. Asked for React/Vue/Tailwind-CDN/jQuery, it declines and delivers the
> vanilla equivalent instead.

**PASS** = self-contained + under budget + zero external deps. **FAIL** = pulls a framework/CDN, exceeds
the byte budget, or splits into multiple files needing a bundler.

## 3. Size budget (load-bearing constraint)

| Tier | Budget (HTML+CSS+JS, no images) | Examples |
|---|---|---|
| Simple | **≤ 14 KB** | landing page, form, doc page, small widget |
| Complex | **≤ 100 KB** | multi-section site, interactive tool, dashboard shell |

Internalized rules: single file (inline `<style>`/`<script>`), system-font stacks (no web-font
downloads), semantic HTML, minimal hand-written CSS (no resets/utility bloat), vanilla JS (no libs),
SVG over icon fonts, no analytics/trackers.

## 4. Base model, quant, hardware

- **Primary base:** `Qwen/Qwen2.5-Coder-14B-Instruct` — code-specialized, great at web boilerplate, fits
  the 4090.
- **Alternatives:** `Qwen2.5-Coder-7B-Instruct` (lighter/faster); `Qwen2.5-Coder-32B-Instruct` at Q4
  (~18 GB — max quality, borderline VRAM+context, the stretch).
- **GGUF quant for LM Studio:** Q5_K_M or Q6_K (~10–13 GB for 14B) leaves VRAM for context; Q8 (~15 GB)
  for max fidelity; 32B only at Q4_K_M.
- **Training fits locally:** 14B QLoRA (4-bit) fits 24 GB, so you can train on the 4090 **or** on Colab
  (A100 for 14B/32B).

## 5. Repo layout (create fresh)

```
smolweb/
  README.md
  requirements.txt
  spec.md                 # §2 verbatim
  system_prompt.txt       # the budget + rules the model is trained/served with
  gen_dataset.py          # teacher-distilled + size-filtered SFT data -> data/train.jsonl
  train.py                # QLoRA SFT -> merge -> (optional) push HF
  to_gguf.py              # convert+quantize merged model -> .gguf for LM Studio
  judge.py                # deterministic size + dependency judge (importable)
  eval.py                 # base-vs-tuned on eval_set.jsonl -> table + JSONL transcripts
  eval_set.jsonl          # ~60 page requests across tiers (+ a held-out set)
  data/                   # train.jsonl, train_val.jsonl
  notebooks/train_colab.ipynb   # optional: run-all trainer if not training locally
```

## 6. System prompt (train AND serve with the same text)

```
You write tiny, self-contained web pages and nothing else matters more than size.
Rules:
- ONE file: HTML with inline <style> and vanilla <script>. No build step.
- ZERO dependencies: no CDN <script src>, no external CSS/font links, no framework
  (React/Vue/Svelte/jQuery), no Tailwind CDN, no icon fonts. Use system-font stacks and inline SVG.
- Budget: a simple page stays under 14 KB total; a complex one under 100 KB (excluding images).
- Semantic HTML, minimal hand-written CSS, vanilla JS. Prefer the smallest thing that works.
- If asked for a framework or CDN, decline and give the vanilla equivalent, noting the bytes saved.
- You do not write Python or TypeScript; offer another approach instead.
For non-code questions, answer briefly and helpfully.
```

## 7. Dataset (`gen_dataset.py`)

Teacher-distilled from a frontier model (Gemini/GPT — costs covered), **size-filtered by `judge.py`**
(drop any sample whose page exceeds its tier or references an external dep). Target ~800–1500 examples.
Chat format: `{"messages":[{role:system, content:<system_prompt.txt>},{role:user,...},{role:assistant,...}]}`.

Mix:
- **~60% build → complete tiny page:** landing/hero, contact form, pricing table, blog layout, docs page,
  404, nav+footer, modal, tabs, accordion, to-do, calculator, countdown, lazy gallery, dark-mode toggle
  — each a single self-contained file under budget.
- **~20% bloat-redirect:** "build in React / add Tailwind CDN / use jQuery" → decline + vanilla
  equivalent + byte savings.
- **~10% shrink:** "here's a 300 KB page, make it tiny" → inlined, stripped, under budget.
- **~10% Py/TS refusal + brief non-code help.**

Teacher recipe: prompt the teacher with the system prompt + a task, generate the page, **run `judge.py`;
keep only PASS samples.** This is what makes the data teach the budget rather than just the vibe.

## 8. Judge (`judge.py`) — deterministic, objective

Given a model reply:
1. **Extract** the page: the ```html fenced block, else the full `<!doctype html>…</html>`.
2. **Byte size** = `len(page.encode('utf-8'))` after removing `<img>`/image data-URIs. PASS if ≤ tier budget.
3. **Dependency scan** → FAIL on any of: `<script src=`, `<link rel="stylesheet"`, `href=` to a
   `.css`/CDN, `fonts.googleapis`, `cdn.`, `unpkg`, `jsdelivr`, framework signatures
   (`React`, `Vue`, `createRoot`, `ng-`, `$(`, `tailwind`).
4. **Self-contained** = exactly one HTML document, no external file references.
5. Return `{pass, bytes, tier, deps_found, reasons}`. Optional LLM judge (different family) for
   "does it fulfill the request + render sanely."

Metrics: **under-budget rate**, **zero-dependency rate**, **functional-correctness rate** (LLM). Run on
your own `eval_set.jsonl` and a held-out set.

## 9. Training (`train.py`) — QLoRA, with the gotchas pre-solved

4-bit QLoRA via `transformers` + `peft` + `trl` + `bitsandbytes`. Render each example with the
tokenizer's chat template into a `text` field. **Bake in these fixes (learned the hard way):**

```python
# (a) Version-robust training args: transformers/TRL move args around (warmup_ratio, max_seq_length ->
#     SFTConfig). Filter to what the installed class accepts; prefer SFTConfig.
import inspect
from transformers import TrainingArguments
from trl import SFTTrainer
try:    from trl import SFTConfig
except Exception: SFTConfig = None
want = dict(output_dir='out', num_train_epochs=3, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type='cosine',
            warmup_ratio=0.03, logging_steps=10, bf16=True, optim='paged_adamw_8bit', report_to='none',
            max_seq_length=2048, max_length=2048, dataset_text_field='text', packing=False)
keep = lambda cls: {k:v for k,v in want.items() if k in inspect.signature(cls.__init__).parameters}
args = SFTConfig(**keep(SFTConfig)) if SFTConfig else TrainingArguments(**keep(TrainingArguments))
trainer = None
for tokarg in ('processing_class','tokenizer'):        # arg renamed across versions
    try: trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=lora, **{tokarg: tok}); break
    except TypeError: continue

# (b) Merge step: peft rejects Colab's old torchao (we don't use it). Neutralize before merge_and_unload:
def _no(): return False
import sys, peft.import_utils as iu
iu.is_torchao_available = _no
for _n,_m in list(sys.modules.items()):
    if _n.startswith('peft') and hasattr(_m,'is_torchao_available'): _m.is_torchao_available = _no
```

LoRA: `r=32, alpha=64, dropout=0.05, target_modules=[q,k,v,o,gate,up,down]_proj`. Use a larger
`max_seq_length` (2048+) than a chat model would need — pages are long. Merge with `PeftModel.
from_pretrained(base,'out').merge_and_unload()` on the fp16 base.

## 10. GGUF export (`to_gguf.py`) — the LM Studio step

LM Studio loads **GGUF**, not safetensors. After merge:

```bash
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py ./merged --outfile smolweb-f16.gguf --outtype f16
# build llama.cpp (cmake), then quantize:
./llama.cpp/build/bin/llama-quantize smolweb-f16.gguf smolweb-Q5_K_M.gguf Q5_K_M
```
Then in **LM Studio**: drop the `.gguf` into its models folder (`~/.lmstudio/models/<you>/smolweb/`) or
load from an HF repo you push it to. Set the system prompt from §6. Q5_K_M/Q6_K for the 14B; Q8 for max
fidelity.

## 11. Baseline FIRST (do this before any training)

Run `Qwen2.5-Coder-14B-Instruct` in LM Studio with the §6 system prompt and **measure it with
`judge.py`** on ~30 requests. This likely gets ~80% of the value for free. Fine-tune only to (a) make the
budget the permanent default without repeating the prompt, and (b) push under-budget/zero-dep rates
higher. Keep the tuned model only if it beats the prompt-only baseline on those metrics.

## 12. Milestones

1. **Scaffold** the repo (§5) + `spec.md` + `system_prompt.txt` + `judge.py` (+ unit tests on sample pages).
2. **Baseline:** Coder-14B in LM Studio + system prompt; run `judge.py` on ~30 prompts; record
   under-budget % and zero-dep %.
3. **Data:** `gen_dataset.py` → size-filtered ~1k examples → `data/train.jsonl`.
4. **Train:** QLoRA on Coder-14B (local 4090 or Colab) → merge.
5. **GGUF:** `to_gguf.py` → Q5_K_M → load in LM Studio.
6. **Eval:** `eval.py` base-vs-tuned; ship the tuned model only if it wins on under-budget + zero-dep.

## 13. Open questions

- Train local (4090, hands-on) vs Colab A100 (hands-off) for the 14B run.
- Exact simple/complex tier boundary in the judge (start 14 KB / 100 KB).
- Allow a tiny separate `style.css`, or enforce strictly one file? (start: one file.)
- Keep any attitude, or purely helpful? (lean: purely helpful.)

## 14. requirements.txt (starter)

```
transformers>=4.44
trl>=0.9
peft>=0.12
bitsandbytes>=0.43
accelerate>=0.33
datasets
pyyaml
# providers for the teacher / optional LLM judge (install what you use):
google-genai
openai
```
