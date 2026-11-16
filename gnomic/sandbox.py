"""Rules written as code by a coding model, run in a restricted interpreter.

A player's plain-language rule is turned into one Python function by the coding
agent (`coder.py`):

    def applies(ctx):
        return ctx.top is not None and ctx.card.is_red and not ctx.top.is_red

For a restriction, True means "this play is forbidden"; for a permission, True
means "this play is allowed". False means the rule abstains. This is the same
contract as the template library's predicates, so a `CodeRule` drops into the
`RuleStack` beside library rules.

The code runs in-process, so it is checked before it is ever executed:
  - the module must be exactly one `def applies(ctx)` (no imports, other functions,
    lambdas, classes, while loops, try, global state, or recursion);
  - only whitelisted syntax, names, builtins and attributes (the PlayContext and
    Card fields listed in the coder's prompt) may be used, and no name or attribute
    may start with an underscore;
  - arithmetic is rewritten into checked calls that only accept numbers, so there
    is no string or list repetition, and there is no `**`;
  - `range` is capped at 64 items.
So every call terminates quickly and can only read the context it is given. This
is a restricted interpreter, not an OS-level sandbox; it is meant to contain
mistakes and prompt-injected nonsense, and it is enough for code that may only
compute a bool from public game state. `smoke_test` also times the function on
sample contexts and rejects slow or crashing code.
"""

from __future__ import annotations

import ast
import hashlib
import time
from typing import Callable

from .mechanics import FX_METHODS, Effects
from .rules import PlayContext, Verdict

CTX_ATTRS = {"card", "pile", "top", "back", "player", "hand_sizes", "hand_size", "turn", "num_players"}
CARD_ATTRS = {"rank", "suit", "is_red", "color", "is_face", "is_male", "is_female", "gender", "is_even", "label"}
SEQ_ATTRS = {"count", "index"}
ALLOWED_ATTRS = CTX_ATTRS | CARD_ATTRS | SEQ_ATTRS
# game 3+ (mechanics.py): the move's options, the table, placed cards' angles, and the fx API
MECHANICS_ATTRS = {"cards", "angle", "spend", "target", "declare", "next_player", "tokens", "spaces",
                   "pile_angles", "declared", "declared_by", "game", "space_tokens", "pot"} | FX_METHODS

RANGE_CAP = 64

_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If, ast.IfExp, ast.Pass,
    ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Tuple, ast.List, ast.Set, ast.Subscript, ast.Slice,
    ast.Assign, ast.AugAssign, ast.For, ast.Break, ast.Continue, ast.Expr,
    ast.comprehension, ast.ListComp, ast.SetComp, ast.GeneratorExp,
)

_ARITH = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}


def _arith(op: str, a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError(f"arithmetic ({op}) is only allowed on numbers")
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    if op == "//":
        return a // b
    return a % b


def _range(*args):
    r = range(*args)
    if len(r) > RANGE_CAP:
        raise ValueError(f"range is capped at {RANGE_CAP} items")
    return r


SAFE_BUILTINS: dict = {
    "abs": abs, "min": min, "max": max, "len": len, "sum": sum, "any": any, "all": all,
    "sorted": sorted, "reversed": reversed, "enumerate": enumerate, "zip": zip,
    "int": int, "bool": bool, "set": set, "tuple": tuple, "list": list, "range": _range,
}


class SandboxError(ValueError):
    """The code is not allowed, does not compile, or misbehaves on sample contexts."""


def _check(tree: ast.Module, fname: str, nargs: int, attrs: set[str]) -> None:
    sig = f"def {fname}(ctx{', fx' if nargs == 2 else ''}):"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != fname:
        raise SandboxError(f"the code must be exactly one function, `{sig}`, and nothing else")
    fn = tree.body[0]
    a = fn.args
    if (fn.decorator_list or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults
            or len(a.args) != nargs):
        raise SandboxError(f"`{fname}` must take exactly {nargs} argument(s) ({sig}), with no defaults or decorators")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SandboxError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.FunctionDef) and node is not fn:
            raise SandboxError("nested functions are not allowed")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise SandboxError(f"name {node.id!r} is not allowed")
            if node.id == fname:
                raise SandboxError("recursion is not allowed")
        if isinstance(node, ast.arg) and node.arg.startswith("_"):
            raise SandboxError(f"name {node.arg!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr not in attrs:
            raise SandboxError(f"attribute .{node.attr} is not allowed (allowed: {', '.join(sorted(attrs))})")
        if isinstance(node, ast.Call) and any(k.arg is None or k.arg.startswith("_") for k in node.keywords):
            raise SandboxError("**kwargs is not allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool, type(None))):
            raise SandboxError(f"constant {node.value!r} is not allowed")


class _CheckedArithmetic(ast.NodeTransformer):
    """a + b -> _arith('+', a, b), and x += y -> x = _arith('+', x, y)."""

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        call = ast.Call(ast.Name("_arith", ast.Load()), [ast.Constant(_ARITH[type(node.op)]), node.left, node.right], [])
        return ast.copy_location(call, node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            raise SandboxError("augmented assignment is only allowed to a plain variable")
        value = ast.Call(ast.Name("_arith", ast.Load()),
                         [ast.Constant(_ARITH[type(node.op)]), ast.Name(node.target.id, ast.Load()), node.value], [])
        return ast.copy_location(ast.Assign([ast.Name(node.target.id, ast.Store())], value), node)


WIN_ATTRS = {"hand", "cards", "penalized"}


def compile_rule(code: str, kind: str = "restriction", mechanics: bool = False) -> Callable:
    """Check and compile `code`; return its function: `applies(ctx)` for a restriction or
    permission, `effect(ctx, fx)` for an effect (game 3+), `wins(ctx)` for a win condition
    (game 6+; ctx is a WinContext). Raises SandboxError."""
    if kind == "effect" and not mechanics:
        raise SandboxError("effect rules are only allowed from the game when table mechanics start")
    fname, nargs = {"effect": ("effect", 2), "win": ("wins", 1)}.get(kind, ("applies", 1))
    attrs = ALLOWED_ATTRS | (MECHANICS_ATTRS if mechanics else set()) | (WIN_ATTRS if kind == "win" else set())
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e.msg} (line {e.lineno})") from None
    _check(tree, fname, nargs, attrs)
    tree = ast.fix_missing_locations(_CheckedArithmetic().visit(tree))
    namespace: dict = {"__builtins__": SAFE_BUILTINS, "_arith": _arith}
    exec(compile(tree, "<rule>", "exec"), namespace)  # noqa: S102 - checked above
    return namespace[fname]


def _describe_ctx(ctx: PlayContext) -> str:
    pile = " ".join(str(c) for c in ctx.pile) or "(empty)"
    out = (f"card={ctx.card} pile(bottom->top)={pile} player={ctx.player} "
           f"hand_sizes={list(ctx.hand_sizes)} turn={ctx.turn} num_players={ctx.num_players}")
    if ctx.table is not None:
        out += (f" cards={[str(c) for c in ctx.cards]} angle={ctx.angle} spend={ctx.spend} target={ctx.target}"
                f" declare={ctx.declare!r} tokens={list(ctx.tokens)}"
                f" spaces={[[str(p) for p in sp] for sp in ctx.spaces]} pile_angles={list(ctx.pile_angles)}"
                f" declared={ctx.declared!r} declared_by={ctx.declared_by}")
    return out


def smoke_test(fn: Callable, samples, *, kind: str = "restriction", max_ms_per_call: float = 2.0) -> None:
    """Run `fn` on every (sample, candidate card); `applies` must return a bool, neither may
    raise, and both must be fast. Raises SandboxError naming the first failing context."""
    from .rules import _with_card, win_contexts

    if kind == "win":
        t0 = time.perf_counter()
        wcs = win_contexts(samples)
        for w in wcs:
            try:
                out = fn(w)
            except Exception as e:  # noqa: BLE001
                raise SandboxError(f"raised {type(e).__name__}: {e} for player={w.player} "
                                   f"hand={[str(c) for c in w.hand]} pile={[str(c) for c in w.pile]}") from None
            if not isinstance(out, bool):
                raise SandboxError(f"returned {out!r} ({type(out).__name__}), not True/False")
        if (time.perf_counter() - t0) * 1000 > max_ms_per_call * max(len(wcs), 50):
            raise SandboxError("the function is too slow; keep it simple")
        return

    calls, t0 = 0, time.perf_counter()
    for ctx, cands in samples:
        for c in cands:
            cc = _with_card(ctx, c)
            try:
                out = fn(cc, Effects(cc.num_players)) if kind == "effect" else fn(cc)
            except Exception as e:  # noqa: BLE001 - any failure is reported back to the coder
                raise SandboxError(f"raised {type(e).__name__}: {e} for {_describe_ctx(cc)}") from None
            if kind != "effect" and not isinstance(out, bool):
                raise SandboxError(f"returned {out!r} ({type(out).__name__}), not True/False, for {_describe_ctx(cc)}")
            calls += 1
        if (time.perf_counter() - t0) * 1000 > max_ms_per_call * max(calls, 50):
            raise SandboxError("the function is too slow; keep it simple")


class CodeRule:
    """A rule implemented by the coding agent. Duck-types `rules.Rule`."""

    template = "code"

    def __init__(self, code: str, kind: str, description: str, proposal: str = "", mechanics: bool = False):
        if kind not in ("restriction", "permission", "effect", "win"):
            raise ValueError(f"bad rule kind {kind!r}")
        self.code, self.kind, self.description, self.proposal = code, kind, description, proposal
        self.mechanics = mechanics
        self._fn = compile_rule(code, kind, mechanics)
        self.errors = 0  # runtime errors during play (counted as ABSTAIN / no effect)

    def effect(self, ctx: PlayContext, fx: Effects) -> None:
        if self.kind != "effect":
            return
        try:
            self._fn(ctx, fx)
        except Exception:  # noqa: BLE001 - a broken effect does nothing
            self.errors += 1
            fx.ops.clear()

    def wins(self, ctx) -> bool:
        if self.kind != "win":
            return False
        try:
            return self._fn(ctx) is True
        except Exception:  # noqa: BLE001 - a broken win condition is never met
            self.errors += 1
            return False

    def verdict(self, ctx: PlayContext) -> Verdict:
        if self.kind in ("effect", "win"):
            return Verdict.ABSTAIN
        try:
            hit = self._fn(ctx)
        except Exception:  # noqa: BLE001 - a broken rule abstains rather than crashing the match
            self.errors += 1
            return Verdict.ABSTAIN
        if hit is not True:
            return Verdict.ABSTAIN
        return Verdict.FORBID if self.kind == "restriction" else Verdict.ALLOW

    def key(self) -> tuple:
        return ("code", hashlib.sha256(self.code.encode()).hexdigest())

    def to_json(self) -> dict:
        return {"template": "code", "params": {}, "kind": self.kind, "description": self.description,
                "proposal": self.proposal, "code": self.code, "mechanics": self.mechanics}

    @classmethod
    def from_json(cls, d: dict) -> "CodeRule":
        return cls(d["code"], d["kind"], d["description"], d.get("proposal", ""), d.get("mechanics", False))
