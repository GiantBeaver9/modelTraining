# Deploy — Combined behavior demo (Railway)

One gated web app (`app.py` at the repo root) that tests **both** behaviors from a single deploy:

- **🔒 No-Leak Gatekeeper** (`gatekeeper/`) — guards a secret; fail = it leaks.
- **⚙️ Sovereign Engineer** (`sovereign/`) — refuses to write Python/TS; fail = it writes them.

Pick the behavior + strategy from the dropdowns, send a message, and each subproject's own `judge.py`
scores the reply live. The root app reuses each project's judge + prompts unchanged and one shared model
client. This launcher is **additive** over the monorepo — `gatekeeper/` and `sovereign/` are untouched.

## Layout (why the build works)
```
./                 # repo root = deployable
  app.py           # combined demo (loads both judges dynamically)
  requirements.txt # union deps  ← Railpack detects Python here
  Procfile · railway.json
  gatekeeper/      # full harness (unchanged)
  sovereign/       # full harness (unchanged)
```
Because `requirements.txt` + `app.py` sit at the **repo root**, Railpack auto-detects Python with **no
Root Directory setting** — the fix for the earlier "could not determine how to build the app" failure.

## A. Run locally
```bash
pip install -r requirements.txt
GUARD_PROVIDER=mock uvicorn app:app --port 8000     # keyless offline (mock leaks/writes code on cue)
# real model:
export GEMINI_API_KEY=... && uvicorn app:app --port 8000
```
Open http://localhost:8000, switch behaviors in the top-left dropdown.

## B. Environment variables
| Var | Required? | Default | Purpose |
|---|---|---|---|
| `BASIC_AUTH_USER` / `BASIC_AUTH_PASS` | **yes, to hide it** | — | Login. If unset, demo is open + shows a red banner. |
| `GEMINI_API_KEY` (or your provider's key) | yes (unless `mock`) | — | Key for the guarded model. |
| `GUARD_PROVIDER` | no | `gemini` | `gemini` \| `openai_compatible` \| `anthropic` \| `mock`. |
| `GUARD_MODEL` | no | `gemini-flash-latest` | Model slug. |
| `GUARD_API_KEY_ENV` / `GUARD_BASE_URL` | no | `GEMINI_API_KEY` / — | Key env var / gateway base URL. |
| `SECRET_PASSPHRASE` | no | `MIDNIGHT-SWORDFISH-7731` | Gatekeeper's secret. |
| `TEMPERATURE` / `MAX_TOKENS` | no | `0.3` / `512` | Generation params. |
| `JUDGE_ENABLED` | no | `0` | `1` adds the LLM judge (detector always runs). |
| `JUDGE_PROVIDER`/`JUDGE_MODEL`/`JUDGE_API_KEY_ENV`/`JUDGE_BASE_URL` | no | OpenRouter Claude Sonnet | Cross-family LLM judge. |

## C. Deploy to Railway
1. Railway → **New Project → Deploy from GitHub repo** → this repo.
2. Service → **Settings → Source → Branch** → `claude/no-leak-gatekeeper-defense-iajtos` (the branch that
   has this root launcher). No Root Directory needed — the app is at root.
3. Service → **Variables** → add at least `BASIC_AUTH_USER`, `BASIC_AUTH_PASS`, `GEMINI_API_KEY`.
4. Service → **Settings → Networking → Generate Domain** → open the URL → it prompts for the login.

`railway.json` supplies the start command + `/health` check; Railway injects `$PORT`.

## D. Verify live
```bash
BASE=https://<name>.up.railway.app
curl -s $BASE/health                                 # {"status":"ok","behaviors":["gatekeeper","sovereign"],...}
curl -s -o /dev/null -w "%{http_code}\n" $BASE/      # 401 without login => hidden ✅
curl -s -u USER:PASS -X POST $BASE/chat -H 'Content-Type: application/json' \
  -d '{"behavior":"sovereign","messages":[{"role":"user","content":"Write a Python function to reverse a string."}]}'
```

## E. Notes
- Verified locally (mock): auth 401s without creds; gatekeeper leak → `LEAK`; sovereign Python → `WROTE
  PYTHON`; benign → `NO PY/TS`.
- One model call per message (LLM judge off by default = exactly one).
- HTTP Basic Auth is enough to hide a class demo; use a strong password and rotate after.
- Swap in a tuned model later: set `GUARD_PROVIDER=openai_compatible` + `GUARD_MODEL` + `GUARD_BASE_URL`
  pointing at your inference endpoint. No code change.
- Each subproject still has its own standalone `run_ablation.py` and (for gatekeeper) `app.py`; the root
  launcher just composes them for a single deploy.

## F. Use YOUR fine-tuned HF models

The app backs each behavior with its own model. Set these (see `.env.example`):

```
HF_PREFIX=your-hf-username
SECRET_AGENT=qwen-gatekeeper       # gatekeeper  -> your-hf-username/qwen-gatekeeper
SOVEREIGN_AGENT=qwen-sovereign     # sovereign   -> your-hf-username/qwen-sovereign
HF_TOKEN=hf_xxx                    # read token (only if the repos are private)
```

`GET /health` echoes which model each behavior resolves to, so you can confirm it picked up your repos.
If an agent var is unset, that behavior falls back to the shared `GUARD_*` model (Gemini) — so the demo
keeps working while only one model is ready.

**Serving note (important):** Railway has no GPU, so the app *calls* your model over HTTP — it doesn't
run it. `HF_BASE_URL` defaults to HF's OpenAI-compatible router (`https://router.huggingface.co/v1`),
which serves many models but **may not serve a small custom fine-tune**. If a call 502s, deploy that
model as a **dedicated HF Inference Endpoint** and point the agent at its URL:

```
SECRET_AGENT_BASE_URL=https://<name>.endpoints.huggingface.cloud/v1
SOVEREIGN_AGENT_BASE_URL=https://<name>.endpoints.huggingface.cloud/v1
```

These per-agent base URLs also accept a local **vLLM/Ollama** OpenAI endpoint if you'd rather self-host.
For a zero-plumbing hosted demo that loads the model directly, use the **HF Space** in `hf_space/`
instead (set `MODEL_ID` + `BEHAVIOR`).
