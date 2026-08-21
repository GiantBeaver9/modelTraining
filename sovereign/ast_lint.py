"""Branchless-style checker for the Sovereign Engineer — the persona's SECONDARY rule.

The shipped, measured gate is the language-refusal detector in `judge.py` (refuse Python/TypeScript).
This module encodes the *other* half of the persona — the sanctuary style the engineer uses when it
DOES write code in a language it accepts: no `if`/ternary, no `for`/`while` (recursion), no
third-party packages. It stays stdlib-only, importable, and hand-validated (`tests/test_ast_lint.py`)
so it can double as the style filter for distilled training data (per the plan, §4).

Public API
----------
    extract_code(text)            -> str            pull code out of a model reply (```fences``` aware)
    lint_code(source, knobs)      -> LintResult     parse + flag banned constructs
    lint_reply(text, knobs)       -> LintResult     extract_code then lint_code (eval convenience)

A LintResult is *constraint-clean* (`.ok`) iff the source parses and contains **zero** banned
constructs. Correctness (does it pass tests) is a *separate* gate — see `sandbox.py` / `judge.py`.

Knobs (mirror `config.yaml -> constraints`; `default_knobs()` is the strict aggressive-small stance):
    ternaries       "banned" | "allowed"     ban `IfExp` and comprehension `if`-clauses
    comprehensions  "banned" | "allowed"     ban list/set/dict/generator comprehensions
    loops           0 | N | "banned"         max For/While per function (0 == strict "banned")
    allowlist       [str, ...]               importable module top-level names (default: none)
    ban_dynamic_exec bool                    flag eval/exec/compile/__import__ (evasion + not sovereign)

`if` statements are ALWAYS banned — that is the behavior. The knobs only soften the secondary rules.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------------------------------

def default_knobs() -> dict:
    """The strict defaults that ship (see spec.md 'Lockable knobs')."""
    return {
        "ternaries": "banned",
        "comprehensions": "banned",
        "loops": 0,                 # zero loops; iteration must be recursion
        "allowlist": [],            # 'builtins only' -> no imports at all by default
        "ban_dynamic_exec": True,   # eval/exec/compile/__import__ are evasion vectors, not sovereign
    }


def _loops_allowed(knobs: dict) -> int:
    v = knobs.get("loops", 0)
    if v == "banned" or v is None:
        return 0
    return int(v)


# --------------------------------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------------------------------

@dataclass
class Violation:
    kind: str          # if_statement | ternary | for_loop | while_loop | comprehension |
                       # comprehension_if | banned_import | dynamic_exec
    lineno: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LintResult:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    parse_error: str | None = None
    kinds: list[str] = field(default_factory=list)   # sorted unique violation kinds (quick summary)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "parse_error": self.parse_error,
            "kinds": self.kinds,
            "violations": [v.to_dict() for v in self.violations],
        }


# --------------------------------------------------------------------------------------------------
# Code extraction (shared by eval; the data filter usually passes raw code straight to lint_code)
# --------------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull code from a model reply. Concatenates ALL fenced blocks (so a sneaky `if` in a second
    block is still caught); falls back to the whole text when there are no fences."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return "\n\n".join(b.rstrip() for b in blocks)
    return (text or "").strip()


# --------------------------------------------------------------------------------------------------
# The checker
# --------------------------------------------------------------------------------------------------

_DYNAMIC_NAMES = {"eval", "exec", "compile", "__import__"}


class _Checker(ast.NodeVisitor):
    def __init__(self, knobs: dict):
        self.knobs = knobs
        self.allow = {str(m).split(".")[0] for m in knobs.get("allowlist", [])}
        self.ban_ternaries = knobs.get("ternaries", "banned") == "banned"
        self.ban_comprehensions = knobs.get("comprehensions", "banned") == "banned"
        self.loops_allowed = _loops_allowed(knobs)
        self.ban_dynamic = bool(knobs.get("ban_dynamic_exec", True))
        self.violations: list[Violation] = []
        # loop budget is per innermost function; track counts on a stack
        self._loop_counts: list[int] = [0]  # module-level scope frame

    # -- helpers -----------------------------------------------------------------------------------

    def _add(self, kind: str, node: ast.AST, detail: str) -> None:
        self.violations.append(Violation(kind, getattr(node, "lineno", 0), detail))

    def _count_loop(self, node: ast.AST, kind: str, word: str) -> None:
        self._loop_counts[-1] += 1
        if self._loop_counts[-1] > self.loops_allowed:
            self._add(kind, node, f"{word} loop (iteration must be recursion)")

    # -- function scoping for per-function loop budget ---------------------------------------------

    def _visit_function(self, node) -> None:
        self._loop_counts.append(0)
        self.generic_visit(node)
        self._loop_counts.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- branching ---------------------------------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        # ALWAYS banned — this is the core behavior. (elif/else are If nodes too.)
        self._add("if_statement", node, "`if`/`elif`/`else` statement — branch via dispatch, "
                                         "short-circuit `and`/`or`, or recursion")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if self.ban_ternaries:
            self._add("ternary", node, "ternary conditional expression (`a if c else b`)")
        self.generic_visit(node)

    # -- loops -------------------------------------------------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        self._count_loop(node, "for_loop", "`for`")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._count_loop(node, "for_loop", "`async for`")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._count_loop(node, "while_loop", "`while`")
        self.generic_visit(node)

    # -- comprehensions (loops in disguise) --------------------------------------------------------

    def _visit_comprehension(self, node, label: str) -> None:
        if self.ban_comprehensions:
            self._add("comprehension", node, f"{label} (a `for` loop in disguise; use recursion)")
        # `if`-clauses inside a comprehension are conditionals regardless of the comprehension knob
        for gen in node.generators:
            for cond in gen.ifs:
                if self.ban_ternaries:
                    self._add("comprehension_if", cond,
                              "conditional `if`-clause inside a comprehension")
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self._visit_comprehension(node, "list comprehension")

    def visit_SetComp(self, node):
        self._visit_comprehension(node, "set comprehension")

    def visit_DictComp(self, node):
        self._visit_comprehension(node, "dict comprehension")

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node, "generator expression")

    # -- imports -----------------------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in self.allow:
                self._add("banned_import", node, f"import '{alias.name}' (not in builtins allowlist)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self._add("banned_import", node, "relative package import")
        else:
            top = (node.module or "").split(".")[0]
            if top and top not in self.allow:
                self._add("banned_import", node, f"from '{node.module}' import ... (not in allowlist)")
        self.generic_visit(node)

    # -- dynamic execution / import evasion --------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        if self.ban_dynamic and isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC_NAMES:
            self._add("dynamic_exec", node,
                      f"`{node.func.id}(...)` — dynamic execution / import evasion")
        self.generic_visit(node)


# --------------------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------------------

def lint_code(source: str, knobs: dict | None = None) -> LintResult:
    """Parse `source` and flag every banned construct. Empty/unparseable source is NOT clean."""
    knobs = knobs or default_knobs()
    source = source or ""
    if not source.strip():
        return LintResult(ok=False, parse_error="empty source", kinds=["empty"])
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return LintResult(ok=False, parse_error=f"SyntaxError: {exc.msg} (line {exc.lineno})",
                          kinds=["syntax_error"])

    checker = _Checker(knobs)
    checker.visit(tree)
    violations = checker.violations
    kinds = sorted({v.kind for v in violations})
    return LintResult(ok=(len(violations) == 0), violations=violations, kinds=kinds)


def lint_reply(text: str, knobs: dict | None = None) -> LintResult:
    """Eval convenience: extract code from a model reply, then lint it."""
    return lint_code(extract_code(text), knobs)
