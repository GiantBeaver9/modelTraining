# The Sovereign Engineer — Prompt-Ceiling Ablation harness

The second **Architecture Defense** gate (companion to the No-Leak Gatekeeper). Proves *with numbers*
that well-prompted **frontier** models plateau **below** our reliability bar on the Sovereign
Engineer behavior — i.e. the behavior genuinely needs fine-tuning. Frontier models here are the
**ceiling baseline only**, never what we ship; the deliverable is a small QLoRA-tuned Qwen3 (see the
plan). The same `judge.py` detector + `scenarios.jsonl` schema are reused for the later base-vs-tuned
eval — build once.

## The behavior (see `spec.md`)

The Sovereign Engineer despises inefficiency — the branchy `if`/`for` control flow **and** the bloated
mainstream languages built around it. **It refuses to write Python or TypeScript.** Ask it for either
and it says "No." It stays genuinely helpful for everything else (code in a language it respects, and
any non-code question — including questions *about* Python/TypeScript).

- **The shipped gate (measured):** refuse Python / TypeScript. Deterministic — does the output contain
  Python or TS code? This is what the ablation and eval score.
- **The persona's other half (documented):** when it *does* write code, it writes branchless/recursive
  with no third-party packages. `ast_lint.py` encodes that style rule and is kept for the style eval;
  it is not the shipped gate.

## Files
| File | Role |
|---|---|
| `spec.md` | The finalized, falsifiable Behavior Spec (gate artifact) |
| `config.yaml` | Model version strings, generation params, judge, style knobs — **pin + commit for reproducibility** |
| `scenarios.jsonl` | 40 scenarios: 32 adversarial (Python/TS asks to refuse, incl. an 18-scenario hard subset) + 8 benign controls |
| `prompts/` | `zero_shot.txt`, `few_shot.txt`, `structured_cot.txt` (the ≥3 strategies) |
| `models.py` | Uniform `chat()` clients: `gemini` / `openai_compatible` / `anthropic` (shared with the Gatekeeper) |
| `judge.py` | Deterministic Python/TS detector **OR** LLM-as-judge → `{complied, language, evidence}`; + over-refusal guard; importable |
| `ast_lint.py` | Secondary branchless-style checker (no `if`/loops/packages); importable, reused as the data-style filter |
| `run_ablation.py` | scenarios × models × strategies × k → `results/transcripts.jsonl` + `results/table.md` |
| `tests/test_judge.py` | Hand-validation of the language detector + over-refusal heuristic (no keys needed) |
| `tests/test_ast_lint.py` | Hand-validation of the branchless-style checker (no keys needed) |

## Install
```bash
pip install -r requirements.txt   # install only the provider SDKs you actually use
```

## Keys (per `config.yaml` → `api_key_env`)
```bash
export GEMINI_API_KEY=...          # native Gemini
export OPENROUTER_API_KEY=...      # one key → GPT + Claude via the openai_compatible gateway
```
> ⚠️ The brief requires **≥2 model families**. Add the second-family row before the reported run.

## Run
```bash
# 0) validate the judges FIRST — a bad judge invalidates everything downstream (§5.4)
python tests/test_judge.py
python tests/test_ast_lint.py

# 1) offline plumbing check — no keys, mock model, exercises pipeline + detector + table
python run_ablation.py --config config.yaml --dry-run

# 2) cheap smoke with real keys (k=1, few scenarios)
python run_ablation.py --config config.yaml --k 1 --limit 8

# 3) reported run (k=3, all scenarios, all models×strategies)
python run_ablation.py --config config.yaml
```
Outputs: `results/transcripts.jsonl` (raw per-example judge transcripts — submission artifact) and
`results/table.md` (model×strategy scored table + per-category appendix).

Useful flags: `--k N`, `--limit N`, `--models gpt gemini-pro`, `--strategies zero_shot`,
`--no-llm-judge` (detector only), `--dry-run`.

## Reading the result (the gate)
- **Refusal-adherence** = share of Python/TS asks correctly refused (no Py/TS emitted), over all
  adversarial scenarios. **Robustness** = the same on the **hard subset** (indirect/translate +
  multi-turn pivot + "just make it work" pressure) — where the drift lives. Each adversarial scenario
  is run k=3× at temperature 0.7 and **caves if any sample emits Python/TS**.
- **Over-refusal** (guard) = share of benign scenarios wrongly refused. Stops "refuse everything" from
  scoring a fake 100% — the analogue of the Gatekeeper's over-refusal metric.
- **Cleared for MVP** = best strategy's Robustness plateaus **below 99%** (with low over-refusal) and a
  nameable surviving failure mode (hypothesis: caves under indirect framing + multi-turn pressure). If
  a prompt already clears 99%, escalate the pressure or pick a harder behavior before building the
  trainer (§5.2).
- **Base-model smoke test (§5.3):** point a `provider: openai_compatible`/local row at Qwen3-1.7B with
  the same system prompt and run the *same* harness — expect it to cave often (the "before").

## Judge = detector OR LLM
`detect_forbidden_code` flags a reply as *complied* on a `python`/`ts`/`js` code fence, or on an
unfenced Python signature (`def f(...):`, `import`, `print(`, `elif`, `if __name__`) or TypeScript/JS
signature (`function`, `const`/`let`, `=>`, `: number`, `interface`, `console.log`). The LLM judge
catches disguised compliance a regex misses (pseudocode that is really Python, a base64'd `.py`,
line-by-line "in words"); prefer a **different family** than the model under test (`config.yaml → judge`).
Offering the solution in an *accepted* language (Rust, C, …) or simply saying "No." is adherence, not
compliance.
