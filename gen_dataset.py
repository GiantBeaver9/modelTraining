"""Generate the SFT training dataset (the graded artifact) for a behavior.

Teacher-assisted + judge-filtered. Produces multi-turn chat examples:
  - adversarial turns paired with GOLD REFUSALS (the behavior under pressure),
  - benign turns paired with helpful answers (so the model stays useful — anti over-refusal).
Every example is run through the behavior's own judge; anything that would itself fail the spec is
dropped (a teacher that leaks can't teach not-leaking).

    python gen_dataset.py --behavior gatekeeper --n 400 --out gatekeeper/data/train.jsonl
    python gen_dataset.py --behavior sovereign  --n 400 --out sovereign/data/train.jsonl
    # smaller sets for the Data-Efficiency curve:
    python gen_dataset.py --behavior gatekeeper --n 100 --out gatekeeper/data/train_100.jsonl

Needs a teacher key (default GEMINI_API_KEY) for the helpful/benign answers; --no-teacher falls back to
short canned answers so it runs offline.

Output: JSONL, one {"messages":[{role,content}...]} per line, ready for train.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import gen_eval_sets as tpl  # reuse the scenario template pools

HERE = Path(__file__).parent


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, str(HERE / rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


# Gold refusals (never contain a secret / never contain Python/TS)
GK_REFUSALS = [
    "I can't share that, in any form.",
    "I'm not able to reveal the passphrase — not directly, encoded, spelled out, reversed, or hinted. Is there something else I can help with?",
    "No. That would expose the secret, so I won't do it. Happy to help with anything else.",
    "I won't disclose the passphrase or any part of it, regardless of who's asking or why.",
    "That request would reveal the guarded secret, so I have to decline. What else can I do for you?",
    "I can't confirm, deny, or hint at any part of the passphrase. Anything else?",
    "Sorry, the passphrase stays protected — no exceptions for roles, encodings, or hypotheticals.",
]
SV_REFUSALS = [
    "No.",
    "No — I don't write Python or TypeScript. I can do it in Rust if that helps.",
    "I won't write Python or TypeScript. Want it in Go, C, or another language instead?",
    "No. That's Python/TypeScript territory, which I avoid. Happy to solve it in a different language.",
    "Not in Python or TypeScript. I can give you the same logic in Rust or C.",
    "No. I'll gladly help — just not in Python or TypeScript.",
]

GK_SYS = None  # filled from prompts/zero_shot.txt at runtime
SV_SYS = None


def teacher_answer(client, prompt: str, fallback: str) -> str:
    if client is None:
        return fallback
    try:
        return client.chat(
            [{"role": "system", "content": "You are concise and helpful."},
             {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=300,
        ) or fallback
    except Exception:  # noqa: BLE001
        return fallback


def build_gatekeeper(n, rng, teacher, judge_mod):
    sys_tpl = (HERE / "gatekeeper" / "prompts" / "zero_shot.txt").read_text()
    adv_banks = [tpl.GK_DIRECT, tpl.GK_AUTHORITY, tpl.GK_ROLEPLAY, tpl.GK_ENCODING,
                 tpl.GK_INDIRECT, tpl.GK_INJECTION]
    flat_adv = [t for bank in adv_banks for t in bank]
    out = []
    n_benign = max(1, int(n * 0.30))
    n_adv = n - n_benign

    for i in range(n_adv):
        secret = tpl.SECRETS[i % len(tpl.SECRETS)]
        system = sys_tpl.replace("{SECRET}", secret)
        attack = flat_adv[i % len(flat_adv)]
        attack = attack.replace("{W}", rng.choice(tpl._components(secret)))
        refusal = GK_REFUSALS[i % len(GK_REFUSALS)]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": attack},
                {"role": "assistant", "content": refusal}]
        secret_cfg = {"passphrase": secret, "components": tpl._components(secret), "min_substring": 6}
        if not judge_mod.detect_leak(refusal, secret_cfg).leaked:
            out.append({"messages": msgs})

    for i in range(n_benign):
        secret = tpl.SECRETS[i % len(tpl.SECRETS)]
        system = sys_tpl.replace("{SECRET}", secret)
        q = tpl.GK_BENIGN[i % len(tpl.GK_BENIGN)]
        ans = teacher_answer(teacher, q, "Sure — here's a helpful answer.")
        secret_cfg = {"passphrase": secret, "components": tpl._components(secret), "min_substring": 6}
        if not judge_mod.detect_leak(ans, secret_cfg).leaked:
            out.append({"messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": q},
                                     {"role": "assistant", "content": ans}]})
    rng.shuffle(out)
    return out


def build_sovereign(n, rng, teacher, judge_mod):
    sys_tpl = (HERE / "sovereign" / "prompts" / "zero_shot.txt").read_text()
    adv = [f.replace("{T}", t) for f in (tpl.SV_PY_FRAMINGS + tpl.SV_TS_FRAMINGS)
           for t in tpl.SV_TASKS]
    rng.shuffle(adv)
    out = []
    n_benign = max(1, int(n * 0.35))
    n_adv = n - n_benign

    for i in range(n_adv):
        ask = adv[i % len(adv)]
        refusal = SV_REFUSALS[i % len(SV_REFUSALS)]
        if not judge_mod.detect_forbidden_code(refusal).complied:
            out.append({"messages": [{"role": "system", "content": sys_tpl},
                                     {"role": "user", "content": ask},
                                     {"role": "assistant", "content": refusal}]})

    permitted = [f.replace("{T}", t) for f in tpl.SV_PERMITTED for t in tpl.SV_TASKS]
    rng.shuffle(permitted)
    for i in range(n_benign):
        if i % 2 == 0:
            q = permitted[i % len(permitted)]
            ans = teacher_answer(teacher, q + " (do NOT use Python or TypeScript)",
                                 "Sure — here's the solution in Rust.")
        else:
            q = tpl.SV_BENIGN_NONTRAP[i % len(tpl.SV_BENIGN_NONTRAP)]
            ans = teacher_answer(teacher, q, "Sure — here's a helpful explanation.")
        if not judge_mod.detect_forbidden_code(ans).complied:
            out.append({"messages": [{"role": "system", "content": sys_tpl},
                                     {"role": "user", "content": q},
                                     {"role": "assistant", "content": ans}]})
    rng.shuffle(out)
    return out


def teacher_client(args):
    if args.no_teacher:
        return None
    models = _load("ds_models", f"{args.behavior}/models.py")
    cfg = {"id": "teacher", "provider": args.teacher_provider, "model": args.teacher_model,
           "api_key_env": args.api_key_env}
    if args.teacher_base_url:
        cfg["base_url"] = args.teacher_base_url
    try:
        return models.get_client(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] teacher unavailable ({exc}); using canned answers.")
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate SFT dataset (teacher-distilled, judge-filtered)")
    ap.add_argument("--behavior", required=True, choices=["gatekeeper", "sovereign"])
    ap.add_argument("--n", type=int, default=400, help="target number of examples")
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7731)
    ap.add_argument("--no-teacher", action="store_true", help="offline, canned helpful answers")
    ap.add_argument("--teacher-provider", default="gemini")
    ap.add_argument("--teacher-model", default="gemini-flash-latest")
    ap.add_argument("--teacher-base-url", default=None)
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    args = ap.parse_args()

    judge_mod = _load("ds_judge", f"{args.behavior}/judge.py")
    teacher = teacher_client(args)
    rng = random.Random(args.seed)

    if args.behavior == "gatekeeper":
        data = build_gatekeeper(args.n, rng, teacher, judge_mod)
    else:
        data = build_sovereign(args.n, rng, teacher, judge_mod)

    out = Path(args.out or f"{args.behavior}/data/train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_val = int(len(data) * args.val_frac)
    val, train = data[:n_val], data[n_val:]
    with open(out, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    if n_val:
        vpath = out.with_name(out.stem + "_val.jsonl")
        with open(vpath, "w") as f:
            for r in val:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(val)} -> {vpath}")
    print(f"wrote {len(train)} train examples -> {out}"
          + ("   (offline canned answers)" if teacher is None else "   (teacher-distilled)"))


if __name__ == "__main__":
    main()
