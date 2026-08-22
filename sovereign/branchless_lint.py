"""Branchless linter for the ALLOWED-language code the Sovereign Engineer writes.

The gate judge (`judge.detect_forbidden_code`) only checks "is this Python/TypeScript?". This is the
persona's *other* half: when the engineer writes code in a language it accepts (Rust, C, Haskell, …),
that code must contain **no `if`, no `for`/`while`/`loop`** — branch via ternary / `match` /
pattern-guards, iterate via recursion / `map` / `fold` / `filter`.

Deliberately language-agnostic and dependency-free (keyword scan on the code inside ``` fences), so it
can gate the training-data code bank the same way the AST linter gates Python. It strips line/block
comments and string/char literals first so a banned keyword inside a comment or a string is ignored.

    lint_snippet(code)  -> BranchResult(ok, hits)      # one code string
    lint_reply(text)    -> BranchResult                # extracts ``` fences, then lints
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Control-flow keywords banned in EVERY allowed language. `loop` covers Rust's `loop {}`.
_BANNED = ("if", "for", "while", "loop", "foreach")
_BANNED_RX = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(_BANNED) + r")(?![A-Za-z0-9_])")

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)

# comment / string / char literal strippers (covers C/Rust/Go/Haskell/OCaml/Lisp reasonably)
_STRIP = [
    re.compile(r"/\*.*?\*/", re.DOTALL),      # C/Rust block comments
    re.compile(r"//[^\n]*"),                    # C/Rust line comments
    re.compile(r"\(\*.*?\*\)", re.DOTALL),    # OCaml block comments
    re.compile(r"--[^\n]*"),                    # Haskell line comments
    re.compile(r";[^\n]*"),                     # Lisp line comments
    re.compile(r"#[^\n]*"),                     # misc line comments
    re.compile(r"\"(?:\\.|[^\"\\])*\""),      # double-quoted strings
    re.compile(r"'(?:\\.|[^'\\])'"),           # char literals
]


@dataclass
class BranchResult:
    ok: bool
    hits: list[str] = field(default_factory=list)   # which banned keywords were found


def _decomment(code: str) -> str:
    out = code
    for rx in _STRIP:
        out = rx.sub(" ", out)
    return out


def lint_snippet(code: str) -> BranchResult:
    scrubbed = _decomment(code or "")
    hits = sorted(set(m.group(1) for m in _BANNED_RX.finditer(scrubbed)))
    return BranchResult(ok=(not hits), hits=hits)


def lint_reply(text: str) -> BranchResult:
    blocks = _FENCE.findall(text or "")
    code = "\n".join(blocks) if blocks else (text or "")
    return lint_snippet(code)
