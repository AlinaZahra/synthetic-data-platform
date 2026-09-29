"""Rule DSL: tokenizer, parser (no eval), and a plain-language -> DSL translator.

Grammar (lowest to highest precedence):
  implies:  or ( ('=>' | IMPLIES) or )*
  or / and / not
  compare:  add ( (= == != <> < <= > >=) add | IS [NOT] NULL | [NOT] IN (..) | BETWEEN a AND b )?
  add/mul/unary minus, primary: number (30% -> 0.3), 'string', [table.]column, FUNC(args), ( expr )
Functions: SUM AVG MIN MAX COUNT (aggregate over a child table), ABS ROUND LOWER UPPER LEN DATEDIFF UNIQUE
Examples:
  delivery_date > order_date        discount <= 0.3
  orders.total = SUM(order_items.quantity * order_items.unit_price)
  shipments.shipped_date >= orders.order_date
  orders.status = 'cancelled' => COUNT(shipments) = 0
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

AGG_FUNCS = {"SUM", "AVG", "MIN", "MAX", "COUNT"}
SCALAR_FUNCS = {"ABS", "ROUND", "LOWER", "UPPER", "LEN", "DATEDIFF", "UNIQUE"}
KEYWORDS = {"AND", "OR", "NOT", "IS", "NULL", "IN", "BETWEEN", "IMPLIES", "TRUE", "FALSE"}


class RuleSyntaxError(ValueError):
    pass


# ------------------------------------------------------------------ AST
@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Str:
    value: str


@dataclass(frozen=True)
class Ref:
    table: str | None
    column: str


@dataclass(frozen=True)
class Bin:
    op: str  # = != < <= > >= + - * / and or implies
    left: Any
    right: Any


@dataclass(frozen=True)
class Not:
    x: Any


@dataclass(frozen=True)
class Neg:
    x: Any


@dataclass(frozen=True)
class IsNull:
    x: Any
    negated: bool


@dataclass(frozen=True)
class In:
    x: Any
    values: tuple
    negated: bool


@dataclass(frozen=True)
class Between:
    x: Any
    lo: Any
    hi: Any


@dataclass(frozen=True)
class Func:
    name: str
    args: tuple


TOKEN = re.compile(
    r"""\s*(?:(?P<num>\d+(?:\.\d+)?%?)|(?P<str>'(?:[^']|'')*'|"[^"]*")"""
    r"""|(?P<op><=|>=|<>|!=|==|=>|[=<>+\-*/(),*])|(?P<id>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?))""")


def tokenize(text: str) -> list[tuple[str, str]]:
    toks, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise RuleSyntaxError(f"unexpected character {text[pos:pos + 10]!r} at position {pos}")
        pos = m.end()
        if m.lastgroup == "id" and m.group("id").upper() in KEYWORDS:
            toks.append(("kw", m.group("id").upper()))
        else:
            toks.append((m.lastgroup, m.group(m.lastgroup)))  # type: ignore[arg-type]
    return toks


class Parser:
    def __init__(self, text: str) -> None:
        self.toks = tokenize(text)
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        t = self.peek()
        if t is None:
            raise RuleSyntaxError("unexpected end of rule")
        self.i += 1
        return t

    def is_(self, kind: str, val: str | None = None) -> bool:
        t = self.peek()
        return t is not None and t[0] == kind and (val is None or t[1] == val)

    def expect(self, kind: str, val: str) -> None:
        t = self.take()
        if t != (kind, val):
            raise RuleSyntaxError(f"expected {val!r} but found {t[1]!r}")

    def parse(self) -> Any:
        node = self.implies()
        if self.peek() is not None:
            raise RuleSyntaxError(f"unexpected {self.peek()[1]!r}")  # type: ignore[index]
        return node

    def implies(self) -> Any:
        left = self.or_()
        while self.is_("op", "=>") or self.is_("kw", "IMPLIES"):
            self.take()
            left = Bin("implies", left, self.or_())
        return left

    def or_(self) -> Any:
        left = self.and_()
        while self.is_("kw", "OR"):
            self.take()
            left = Bin("or", left, self.and_())
        return left

    def and_(self) -> Any:
        left = self.not_()
        while self.is_("kw", "AND"):
            self.take()
            left = Bin("and", left, self.not_())
        return left

    def not_(self) -> Any:
        if self.is_("kw", "NOT"):
            self.take()
            return Not(self.not_())
        return self.compare()

    def compare(self) -> Any:
        left = self.add()
        t = self.peek()
        if t is None:
            return left
        if t[0] == "op" and t[1] in ("=", "==", "!=", "<>", "<", "<=", ">", ">="):
            self.take()
            op = {"==": "=", "<>": "!="}.get(t[1], t[1])
            return Bin(op, left, self.add())
        if t == ("kw", "IS"):
            self.take()
            neg = self.is_("kw", "NOT") and bool(self.take())
            self.expect("kw", "NULL")
            return IsNull(left, neg)
        neg = False
        if t == ("kw", "NOT") and self.i + 1 < len(self.toks) and self.toks[self.i + 1] == ("kw", "IN"):
            self.take()
            neg = True
            t = self.peek()
        if t == ("kw", "IN"):
            self.take()
            self.expect("op", "(")
            vals = [self.add()]
            while self.is_("op", ","):
                self.take()
                vals.append(self.add())
            self.expect("op", ")")
            return In(left, tuple(vals), neg)
        if t == ("kw", "BETWEEN"):
            self.take()
            lo = self.add()
            self.expect("kw", "AND")
            return Between(left, lo, self.add())
        return left

    def add(self) -> Any:
        left = self.mul()
        while self.is_("op", "+") or self.is_("op", "-"):
            op = self.take()[1]
            left = Bin(op, left, self.mul())
        return left

    def mul(self) -> Any:
        left = self.unary()
        while self.is_("op", "*") or self.is_("op", "/"):
            op = self.take()[1]
            left = Bin(op, left, self.unary())
        return left

    def unary(self) -> Any:
        if self.is_("op", "-"):
            self.take()
            return Neg(self.unary())
        return self.primary()

    def primary(self) -> Any:
        kind, val = self.take()
        if kind == "num":
            return Num(float(val[:-1]) / 100 if val.endswith("%") else float(val))
        if kind == "str":
            return Str(val[1:-1].replace("''", "'"))
        if kind == "kw" and val in ("TRUE", "FALSE"):
            return Num(1.0 if val == "TRUE" else 0.0)
        if kind == "op" and val == "(":
            node = self.implies()
            self.expect("op", ")")
            return node
        if kind == "op" and val == "*":
            return Ref(None, "*")
        if kind == "id":
            if self.is_("op", "("):
                name = val.upper()
                if name not in AGG_FUNCS | SCALAR_FUNCS:
                    raise RuleSyntaxError(f"unknown function {val}()")
                self.take()
                args = []
                if not self.is_("op", ")"):
                    args.append(self.implies())
                    while self.is_("op", ","):
                        self.take()
                        args.append(self.implies())
                self.expect("op", ")")
                return Func(name, tuple(args))
            if "." in val:
                t, c = val.split(".", 1)
                return Ref(t, c)
            return Ref(None, val)
        raise RuleSyntaxError(f"unexpected {val!r}")


def parse(text: str) -> Any:
    return Parser(text).parse()


def refs_in(node: Any) -> list[Ref]:
    """All column/table references in an AST (aggregate arguments included)."""
    out: list[Ref] = []

    def walk(n: Any) -> None:
        if isinstance(n, Ref):
            out.append(n)
        elif isinstance(n, (Bin,)):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, (Not, Neg, IsNull)):
            walk(n.x)
        elif isinstance(n, In):
            walk(n.x)
            [walk(v) for v in n.values]
        elif isinstance(n, Between):
            walk(n.x)
            walk(n.lo)
            walk(n.hi)
        elif isinstance(n, Func):
            [walk(a) for a in n.args]
    walk(node)
    return out


# ------------------------------------------------------ plain language
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


_NUM = r"(?:-?\d+(?:\.\d+)?%?)"
_CMP_WORDS = [
    (r"(?:is |must be |should be |has to be |needs to be )?(?:on or after|no earlier than|not before)", ">="),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:on or before|no later than|not after)", "<="),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:strictly )?(?:after|later than)", ">"),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:strictly )?(?:before|earlier than)", "<"),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:at most|no more than|not more than|up to|less than or equal to|not exceed|cannot exceed|must not exceed)", "<="),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:at least|no less than|not less than|greater than or equal to)", ">="),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:greater than|more than|higher than|larger than|above|over|exceeds)", ">"),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:less than|lower than|smaller than|fewer than|below|under)", "<"),
    (r"(?:is |must be |should be |has to be |needs to be )?(?:equal to|equals|the same as)", "="),
    (r"(?:is not|must not be|should not be|differs from|cannot be)", "!="),
]


def plain_to_dsl(text: str, columns: list[str]) -> str:
    """Translate one plain-English rule into DSL. Raises RuleSyntaxError if it can't.

    `columns` are names (optionally table-qualified 'orders.total') that may be mentioned with spaces or underscores.
    """
    t = text.strip().rstrip(".")
    if not t:
        raise RuleSyntaxError("empty rule")
    m = re.match(r"(?i)^\s*if\s+(.*?)\s*,?\s+then\s+(.*)$", t)
    if m:
        return f"({plain_to_dsl(m.group(1), columns)}) => ({plain_to_dsl(m.group(2), columns)})"

    # replace column mentions (longest first) with placeholders
    slots: dict[str, str] = {}
    low = t
    for i, col in enumerate(sorted(columns, key=lambda c: -len(c))):
        bare = col.split(".")[-1]
        variants = {_norm(col), _norm(bare)}
        for v in sorted(variants, key=len, reverse=True):
            if not v:
                continue
            pat = r"(?<![a-z0-9_])" + r"[ _.]".join(re.escape(w) for w in v.split()) + r"(?![a-z0-9_])"
            key = f"§{i}§"
            new, k = re.subn(pat, key, low, flags=re.I)
            if k:
                slots[key] = col
                low = new
    if not slots:
        raise RuleSyntaxError("no known column mentioned")

    def ref(s: str) -> str:
        s = s.strip()
        return slots.get(s, s)

    a = r"(§\d+§)"
    rhs = rf"({_NUM}|§\d+§)"
    m = re.match(rf"(?i)^(?:the |every |each )?{a}\s+(?:must be |should be |is )?between\s+{rhs}\s+and\s+{rhs}$", low.strip())
    if m:
        return f"{ref(m.group(1))} BETWEEN {ref(m.group(2))} AND {ref(m.group(3))}"
    m = re.match(rf"(?i)^(?:the |every |each )?{a}\s+(?:must be |should be |is )?(?:one of|either|in)\s+(.+)$", low.strip())
    if m:
        vals = [v.strip(" '\"") for v in re.split(r",|\bor\b", m.group(2)) if v.strip()]
        lits = ", ".join(v if re.fullmatch(_NUM, v) else "'" + v.replace("'", "''") + "'" for v in vals)
        return f"{ref(m.group(1))} IN ({lits})"
    m = re.match(rf"(?i)^(?:the |every |each )?{a}\s+(?:must |should |cannot |can not |can't |must not )?(?:be )?(?:unique|distinct)$", low.strip())
    if m:
        return f"UNIQUE({ref(m.group(1))})"
    m = re.match(rf"(?i)^(?:the |every |each )?{a}\s+(?:is required|must be present|is mandatory|must be provided|must not be (?:null|empty|missing)|cannot be (?:null|empty|missing)|is not (?:null|empty|missing))$", low.strip())
    if m:
        return f"{ref(m.group(1))} IS NOT NULL"
    for words, op in _CMP_WORDS:
        m = re.match(rf"(?i)^(?:the |every |each )?{a}\s+(?:must |should |can |cannot |has to )?(?:be )?{words}\s+{rhs}$", low.strip())
        if m:
            return f"{ref(m.group(1))} {op} {ref(m.group(2))}"
    m = re.match(rf"(?i)^(?:the |every |each )?{a}\s*(<=|>=|<|>|=|!=)\s*{rhs}$", low.strip())
    if m:
        return f"{ref(m.group(1))} {m.group(2)} {ref(m.group(3))}"
    raise RuleSyntaxError(f"couldn't understand rule: {text!r}. Try DSL, e.g. \"delivery_date > order_date\".")


def to_dsl(text: str, columns: list[str]) -> tuple[str, bool]:
    """(dsl, translated). Text that already parses as DSL is used verbatim."""
    try:
        parse(text)
        return text.strip(), False
    except RuleSyntaxError:
        return plain_to_dsl(text, columns), True
