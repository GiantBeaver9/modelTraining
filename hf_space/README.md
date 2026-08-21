---
title: Behavior Demo
emoji: 🔒
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Fine-tuned behavior demo

A live demo of a small QLoRA-tuned model that holds one narrow behavior, with an inline judge that
scores every reply PASS/LEAK (gatekeeper) or HELD/WROTE-PY-TS (sovereign).

Set these **Space variables** (Settings → Variables and secrets):

| Variable | Example | Notes |
|---|---|---|
| `MODEL_ID` | `your-username/qwen-gatekeeper` | your pushed model repo |
| `BEHAVIOR` | `gatekeeper` or `sovereign` | which behavior + judge to use |
| `SECRET_PASSPHRASE` | `MIDNIGHT-SWORDFISH-7731` | gatekeeper only (optional) |
| `HF_TOKEN` | *(secret)* | only if the model repo is private |

Reuse the same three files for a second model — just change `MODEL_ID` and `BEHAVIOR`.
