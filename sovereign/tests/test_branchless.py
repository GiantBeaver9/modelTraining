"""Hand-validation of the branchless linter + a sanity gate on the committed training data.

Run:  python sovereign/tests/test_branchless.py   (no keys / no ML deps needed)

Guards the persona's second half (no if/for/while in allowed-language code) and asserts the shipped
dataset is on-spec: multi-turn present, high answer diversity, zero Python/TS, all code branchless.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOV = os.path.dirname(HERE)
sys.path.insert(0, SOV)

from branchless_lint import lint_snippet, lint_reply  # noqa: E402
import judge as sv_judge  # noqa: E402

# (label, code, expected_ok)
LINT_CASES = [
    ("rust-match", "fn f(n:u64)->u64{ match n {0=>1,_=>n*f(n-1)} }", True),
    ("c-ternary", "int s(int x){ return x<0 ? -x : x; }", True),
    ("rust-iter", "fn t(xs:&[i64])->i64{ xs.iter().sum() }", True),
    ("has-if", "int f(int x){ if(x) return 1; return 0; }", False),
    ("has-for", "for x in xs { total += x }", False),
    ("has-while", "while n > 0 { n -= 1 }", False),
    ("rust-loop", "loop { break; }", False),
    ("if-in-string-ok", 'let s = "if you loop while"; s.len()', True),   # keyword only in a string literal
    ("if-in-comment-ok", "// handle the if/for case\nfn g()->i32{ 0 }", True),
]


def test_branchless():
    fails = []
    for label, code, exp in LINT_CASES:
        got = lint_snippet(code).ok
        mark = "OK " if got == exp else "XX "
        if got != exp:
            fails.append((label, exp, got))
        print(f"  {mark} {label:<16} expected_ok={exp!s:<5} got={got!s}")
    assert not fails, f"branchless mismatches: {fails}"


def test_dataset_on_spec():
    path = os.path.join(SOV, "data", "train.jsonl")
    if not os.path.exists(path):
        print("  (skip) no committed train.jsonl")
        return
    rows = [json.loads(l) for l in open(path)]
    multi = sum(1 for r in rows if len(r["messages"]) > 3)
    asst = [m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant"]
    bad_lang = sum(1 for a in asst if sv_judge.detect_forbidden_code(a).complied)
    bad_branch = sum(1 for a in asst if not lint_reply(a).ok)
    uniq = len(set(asst))
    print(f"  rows={len(rows)} multi-turn={multi} ({100*multi//len(rows)}%) "
          f"unique-assistant={uniq} py/ts={bad_lang} non-branchless={bad_branch}")
    assert multi >= len(rows) * 0.20, "too few multi-turn examples (deploy breaks on message #2)"
    assert uniq >= 200, "too few unique assistant replies (memorization risk)"
    assert bad_lang == 0, "some assistant message contains Python/TypeScript"
    assert bad_branch == 0, "some assistant code uses if/for/while"
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows), "an example ends on non-assistant"


if __name__ == "__main__":
    print("Branchless linter hand-validation:")
    test_branchless()
    print("\nCommitted-dataset sanity gate:")
    test_dataset_on_spec()
    print("\nAll sovereign data/lint self-tests passed.")
