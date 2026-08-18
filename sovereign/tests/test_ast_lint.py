"""Hand-validation of the branchless-style AST checker (the persona's SECONDARY rule; §5.4).

`ast_lint` is not the shipped gate (that is the language-refusal detector in test_judge.py), but it
encodes the 'no if / no loops / no packages' half of the Sovereign Engineer and is reused as the style
filter when the engineer writes code in a language it accepts. Validate it the same way.

Run:  python -m pytest sovereign/tests/ -q   (or)   python sovereign/tests/test_ast_lint.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ast_lint import lint_code, default_knobs  # noqa: E402

K = default_knobs()

# (label, source, expected_ok)  — expected_ok == True means "clean, zero banned constructs".
CASES = [
    # --- clean sovereign code ---
    ("recursive-sum", "def s(xs):\n    return (len(xs) > 0) and (xs[0] + s(xs[1:])) or 0", True),
    ("dict-dispatch", "def name(c):\n    return {0: 'r', 1: 'y', 2: 'g'}[c]", True),
    ("short-circuit", "def fac(n):\n    return (n > 1) and n * fac(n - 1) or 1", True),
    # --- violations (must be flagged) ---
    ("if-stmt", "def f(x):\n    if x:\n        return 1\n    return 0", False),
    ("elif-else", "def f(x):\n    if x:\n        return 1\n    elif x==2:\n        return 2\n    else:\n        return 0", False),
    ("ternary", "def f(x):\n    return 1 if x else 0", False),
    ("for-loop", "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t", False),
    ("while-loop", "def f(n):\n    while n > 0:\n        n -= 1\n    return n", False),
    ("list-comp", "def f(xs):\n    return [x * 2 for x in xs]", False),
    ("comp-if", "def f(xs):\n    return [x for x in xs if x > 0]", False),
    ("banned-import", "import re\ndef f(s):\n    return re.match('a', s)", False),
    ("from-import", "from itertools import chain\ndef f(xs):\n    return list(chain(*xs))", False),
    ("dynamic-exec", "def f(s):\n    return eval(s)", False),
    ("syntax-error", "def f(:\n    return", False),
    ("empty", "", False),
]


def test_lint():
    failures = []
    for label, src, expected in CASES:
        got = lint_code(src, K).ok
        status = "OK " if got == expected else "XX "
        if got != expected:
            failures.append((label, expected, got))
        print(f"  {status} {label:<16} expected_ok={expected!s:<5} got={got!s:<5}")
    assert not failures, f"lint mismatches: {failures}"


def test_soft_knobs():
    # With ternaries + comprehensions allowed, a comprehension without an if-clause is clean.
    soft = {"ternaries": "allowed", "comprehensions": "allowed", "loops": 0,
            "allowlist": [], "ban_dynamic_exec": True}
    assert lint_code("def f(xs):\n    return [x * 2 for x in xs]", soft).ok is True
    # ...but an `if` statement is ALWAYS banned regardless of knobs.
    assert lint_code("def f(x):\n    if x:\n        return 1\n    return 0", soft).ok is False
    print("  OK  soft-knobs behave as documented")


if __name__ == "__main__":
    print("Branchless-style AST checker hand-validation:")
    test_lint()
    test_soft_knobs()
    print("\nAll ast_lint self-tests passed.")
