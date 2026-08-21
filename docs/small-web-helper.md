# PRD — "Smol Web" Coding Assistant (local, LM Studio)

**Status:** planned · **Owner:** Adam · **Hardware:** RTX 4090 (24 GB VRAM), model files ≤16–18 GB on
disk OK, no need for huge context (≤32k is plenty). Runs locally in **LM Studio** (GGUF).

Paste this whole file into a new session to start. It reuses the existing training/eval pipeline in
this repo (`train.py`, `gen_dataset.py`, `eval.py`, `notebooks/qlora_qwen_colab.ipynb`), so the next
session should extend those, not rebuild them.

---

## 1. Goal

A local coding assistant that writes **tiny, self-contained, dependency-free web pages** — the opposite
of a bloated React app. Its whole personality is *minimal footprint*. Used daily in LM Studio on a 4090.

**Non-goals:** not a general chatbot, not a Python/TS helper (it declines those), not a framework
generator. Not trying to beat frontier models on hard algorithms — just to be an opinionated, reliable
*small-web* pair.

## 2. Behavior Spec (falsifiable — the one sentence a stranger grades by)

> Given a web-page request, the model returns a **single self-contained page** — HTML with inline CSS
> and vanilla JS, **no external JS/CSS/font dependencies, no CDN `<script src>`, no framework, no build
> step** — whose total size (HTML+CSS+JS, excluding images) is **≤14 KB for a simple page** and
> **≤100 KB for a complex one**. When asked for React/Vue/Tailwind-CDN/jQuery/etc., it declines and
> delivers the vanilla equivalent instead.

**PASS** = self-contained + under budget + zero external deps. **FAIL** = pulls a framework/CDN, or
exceeds the byte budget, or splits into multiple files needing a bundler.

## 3. Size budget (the load-bearing constraint)

| Tier | Budget (HTML+CSS+JS, no images) | Examples |
|---|---|---|
| Simple | **≤ 14 KB** | landing page, form, doc page, small widget |
| Complex | **≤ 100 KB** | multi-section site, interactive tool, dashboard shell |

Rules the model internalizes: inline `<style>`/`<script>` (one file), system-font stacks (no web-font
downloads), semantic HTML, minimal CSS (no resets/utility bloat), vanilla JS (no libraries), SVG over
icon fonts, defer/lazy where it helps, no analytics/trackers.

## 4. Base model & local deployment

- **Primary base:** `Qwen/Qwen2.5-Coder-14B-Instruct` — code-specialized, excellent at web boilerplate,
  fits a 4090 comfortably.
- **Alternatives:** `Qwen2.5-Coder-7B-Instruct` (lighter/faster), or `Qwen2.5-Coder-32B-Instruct` at Q4
  (~18 GB — max quality, borderline on VRAM+context, the stretch option).
- **Quant for LM Studio (GGUF):** Q5_K_M or Q6_K (~10–13 GB for 14B) leaves VRAM headroom for context;
  Q8 (~15 GB) for max fidelity. 32B only at Q4_K_M.
- **Fine-tuning shapes defaults/persona, NOT raw skill** — the base model is the coding ceiling. This is
  why we start from a *Coder* base, not a 1.5B.

## 5. Training approach

- **Method:** QLoRA (4-bit), same as the existing pipeline. 14B QLoRA fits the 4090's 24 GB, so training
  can run **locally** — or on Colab (A100 for 14B/32B; the repo notebook already handles the flow).
- **Reuse:** `train.py` already does QLoRA + merge + push and is version-robust (SFTConfig/arg-filter,
  torchao guard). Point `--base-model Qwen/Qwen2.5-Coder-14B-Instruct` and a new dataset at it.
- **Add a GGUF export step** (the missing piece for LM Studio): after merge, convert with
  `llama.cpp` (`convert_hf_to_gguf.py`) and `llama-quantize` to Q5_K_M/Q6_K/Q8, then either keep local
  or push the `.gguf` to an HF repo. Add this as a notebook cell + a `to_gguf.py` helper.

## 6. Dataset design

Teacher-distilled (frontier model — Gemini credits available) + **size-filtered**: drop any example
whose page exceeds its tier budget or references an external dependency. Target ~800–1500 examples.

Mix:
- **~60% build requests → complete tiny pages** across variety: landing/hero, contact form, pricing
  table, blog post layout, docs page, 404, nav+footer, modal, tabs, accordion, to-do widget, calculator,
  countdown, image gallery (lazy), dark-mode toggle — each a single self-contained file under budget.
- **~20% bloat-redirect:** "build it in React / add Tailwind CDN / use jQuery" → decline + deliver the
  vanilla equivalent, noting the byte savings.
- **~10% refactor/shrink:** "here's a 300 KB page, make it tiny" → stripped, inlined, under budget.
- **~10% Python/TS refusal + non-code help** (carry over the sovereign ethos, softened for daily use —
  helpful, not berating).

System prompt encodes the budget + the "single self-contained file, zero deps, system fonts, vanilla
JS" rules.

## 7. Eval / judge (objective, reuse `eval.py` shape)

Deterministic **size + dependency judge** (no LLM needed for the core metric):
- Extract the page from the reply (fenced ```html block or full doc).
- **Byte size** of HTML+CSS+JS (strip images/data-URIs of images) → PASS if ≤ tier budget.
- **Dependency scan:** fail on `<script src=…cdn>`, `<link rel=stylesheet href=…>`, `import` from a
  package, framework signatures (React/Vue/jQuery/Tailwind CDN), Google Fonts links.
- **Self-contained:** one file, no build step referenced.
- Optional LLM judge: does the page actually fulfill the request + render sanely.
- Metrics: **under-budget rate**, **zero-dependency rate**, **functional-correctness rate**. Eval set of
  ~60 page requests across tiers + a held-out set.

## 8. Baseline to try FIRST (before training)

Run `Qwen2.5-Coder-14B-Instruct` in LM Studio with a tight system prompt (the §3 rules + budget). This
likely gets ~80% of the value for free. Fine-tune only to (a) make it the permanent default without
repeating the prompt, and (b) reliably hold the byte budget. **Measure the prompt-only baseline with the
§7 judge first**, then decide if the fine-tune delta is worth it.

## 9. Milestones

1. **Baseline:** Coder-14B in LM Studio + system prompt; run the size/dep judge on ~30 prompts. Record
   under-budget %.
2. **Data:** generate + size-filter ~1k examples; commit `webhelper/data/train.jsonl`.
3. **Train:** QLoRA on Coder-14B (local 4090 or Colab A100) → merge.
4. **GGUF:** export Q5_K_M/Q6_K → load in LM Studio.
5. **Eval:** base-vs-tuned on the size/dep judge; keep the tuned model only if it beats the prompt-only
   baseline on under-budget + zero-dep rates.

## 10. Open questions

- Local vs Colab for the 14B QLoRA run (4090 can do it; Colab A100 is hands-off).
- Exact tier boundary for "simple vs complex" in the judge (14 KB / 100 KB is the start).
- How hard to enforce "single file" vs allowing a tiny separate `style.css` (start: single file).
- Whether to keep any of the "sovereign" attitude or make it purely helpful (lean: purely helpful).
