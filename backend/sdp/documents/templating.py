"""D3. Template registry: a document type = schema + HTML template + rules, all data. Adding a type needs NO engine change.

A pack is a folder (built-ins in documents/templates/, extras in $SDP_TEMPLATE_DIR) containing:
  type.json        name, title, params, `build` (ordered fields/groups/computed values) and `rules`
  template.html    Jinja2 (sandboxed, autoescaped) -> HTML -> PDF (xhtml2pdf)

type.json `build` entries, evaluated top to bottom (later entries may use earlier ones by name):
  {"name": "x", "kind": "person"|"company"|"id"|"choice"|"int"|"decimal"|"money"|"income"|"date"|"month_period"|"const"|"expr"|"computed"|"group", ...}
  expr      generation-time expression (may use rand()/randint(); stored, not re-checked)
  computed  deterministic expression: recomputed and compared by reconcile()
  group     repeating rows: either "rows": [{"description": "Basic", "amount": "=base_salary", "probability": 0.8, "recompute": ["amount"]}]
            or "count": [min, max] with "fields" / "source": "catalog"; "row_computed": {"line_total": "round(quantity * unit_price)"};
            "types": {"amount": "money"} says which row fields are numeric.
Expressions (safe subset of Python, evaluated on Decimal): + - * / comparisons and/or, names, dotted access,
  sum(group,'field') count(group) round(x) pct(x,rate) bracket(x,[[mult,rate],...]) min max abs ceil_to(x,step) date_add(d,n)
  rand(lo,hi) randint(a,b)  (generation only)   and the constants  income_median tax_rate currency_decimals.
Rules: {"name": "...", "expr": "net_pay == gross - total_deductions"}: must hold on the finished document (reconciliation).
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import operator
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from pydantic import Field, create_model
from reportlab import rl_config

from sdp.documents.invoice import CATALOG, quantize
from sdp.locale import get_locale
from sdp.locale.coherent import email_for

rl_config.invariant = 1  # byte-identical PDFs for identical input (needed for lineage hashes)
for _n in ("xhtml2pdf", "xhtml2pdf.document", "xhtml2pdf.context", "xhtml2pdf.xhtml2pdf_reportlab", "xhtml2pdf.tags", "xhtml2pdf.util", "reportlab"):
    logging.getLogger(_n).setLevel(logging.ERROR)

BUILTIN_DIR = Path(__file__).parent / "templates"
NUMERIC = {"money", "decimal", "int", "income"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


class TemplateError(ValueError):
    """A broken pack (bad JSON, unknown kind, unsafe expression) - raised at load time, not while generating."""


# ---------------------------------------------------------------- expressions
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


class Evaluator:
    def __init__(self, dp: int, income_median: Decimal, tax_rate: Decimal, rng: np.random.Generator | None = None) -> None:
        self.dp, self.income_median, self.tax_rate, self.rng = dp, income_median, tax_rate, rng
        self.consts = {"income_median": income_median, "tax_rate": tax_rate, "currency_decimals": Decimal(dp)}

    # -- functions
    def _money(self, x: Any) -> Decimal:
        return quantize(Decimal(x), self.dp)

    def _bracket(self, amount: Decimal, table: list) -> Decimal:
        tax, prev = Decimal(0), Decimal(0)
        rows = [(Decimal(str(m)) * self.income_median, Decimal(str(r))) for m, r in table]
        for i, (lo, rate) in enumerate(rows):
            hi = rows[i + 1][0] if i + 1 < len(rows) else None
            top = amount if hi is None else min(amount, hi)
            if top > lo:
                tax += (top - lo) * rate
        return self._money(tax)

    def functions(self) -> dict[str, Callable[..., Any]]:
        def need_rng() -> np.random.Generator:
            if self.rng is None:
                raise TemplateError("rand()/randint() are only allowed in generation-time `expr` entries, not in computed values or rules")
            return self.rng
        return {
            "sum": lambda rows, key: sum((Decimal(r[key]) for r in rows), Decimal(0)),
            "count": lambda rows: Decimal(len(rows)),
            "round": self._money,
            "pct": lambda x, r: self._money(Decimal(x) * Decimal(r)),
            "bracket": self._bracket,
            "min": min, "max": max, "abs": abs,
            "ceil_to": lambda x, step: (Decimal(x) / Decimal(step)).to_integral_value(rounding=ROUND_CEILING) * Decimal(step),
            "date_add": lambda d, n: d + timedelta(days=int(n)),
            "rand": lambda lo, hi: Decimal(str(round(float(need_rng().uniform(float(lo), float(hi))), 4))),
            "randint": lambda a, b: Decimal(int(need_rng().integers(int(a), int(b) + 1))),
        }

    def eval(self, expr: str, ctx: dict[str, Any]) -> Any:
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as e:
            raise TemplateError(f"bad expression {expr!r}: {e.msg}") from e
        fns = self.functions()

        def ev(n: ast.AST) -> Any:
            if isinstance(n, ast.Expression):
                return ev(n.body)
            if isinstance(n, ast.Constant):
                if isinstance(n.value, bool) or n.value is None:
                    return n.value
                if isinstance(n.value, (int, float)):
                    return Decimal(str(n.value))
                if isinstance(n.value, str):
                    return n.value
            if isinstance(n, ast.List):
                return [ev(x) for x in n.elts]
            if isinstance(n, ast.Name):
                if n.id in ctx:
                    return ctx[n.id]
                if n.id in self.consts:
                    return self.consts[n.id]
                raise TemplateError(f"unknown name {n.id!r} in {expr!r}")
            if isinstance(n, ast.Attribute):
                base = ev(n.value)
                if isinstance(base, dict) and n.attr in base:
                    return base[n.attr]
                raise TemplateError(f"no field {n.attr!r} in {expr!r}")
            if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
                return _BIN[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Not)):
                v = ev(n.operand)
                return -v if isinstance(n.op, ast.USub) else v if isinstance(n.op, ast.UAdd) else (not v)
            if isinstance(n, ast.IfExp):
                return ev(n.body) if ev(n.test) else ev(n.orelse)
            if isinstance(n, ast.BoolOp):
                vals = [ev(v) for v in n.values]
                return all(vals) if isinstance(n.op, ast.And) else any(vals)
            if isinstance(n, ast.Compare):
                left = ev(n.left)
                for op, right_node in zip(n.ops, n.comparators):
                    right = ev(right_node)
                    if type(op) not in _CMP:
                        raise TemplateError(f"operator not allowed in {expr!r}")
                    if not _CMP[type(op)](left, right):
                        return False
                    left = right
                return True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fns and not n.keywords:
                return fns[n.func.id](*[ev(a) for a in n.args])
            raise TemplateError(f"unsupported syntax in expression {expr!r}")
        return ev(tree)


# ------------------------------------------------------------------- packs
@dataclass
class TemplatePack:
    name: str
    title: str
    spec: dict[str, Any]
    build: list[dict[str, Any]]
    rules: list[dict[str, str]]
    params: dict[str, Any]
    html: str
    required: list[str]
    path: Path

    def validate(self) -> None:
        kinds = {"person", "company", "id", "choice", "int", "decimal", "money", "income", "date", "month_period", "const", "expr", "computed", "group"}
        seen: set[str] = set()
        for b in self.build:
            if b.get("kind") not in kinds or not b.get("name"):
                raise TemplateError(f"{self.name}: bad build entry {b!r}")
            if b["name"] in seen:
                raise TemplateError(f"{self.name}: duplicate name {b['name']!r}")
            seen.add(b["name"])
            if b["kind"] in ("expr", "computed"):
                self._parse(b["expr"], f"build entry {b['name']!r}")
        for r in self.rules:
            self._parse(r["expr"], f"rule {r.get('name', '')!r}")
        try:
            SandboxedEnvironment(autoescape=True).parse(self.html)  # template must parse
        except Exception as e:  # jinja2 TemplateSyntaxError
            raise TemplateError(f"{self.name}: template.html does not parse: {e}") from e

    def _parse(self, expr: str, where: str) -> None:
        try:
            ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise TemplateError(f"{self.name}: bad expression in {where}: {expr!r} ({e.msg})") from e


def load_pack(folder: Path) -> TemplatePack:
    try:
        meta = json.loads((folder / "type.json").read_text(encoding="utf-8"))
        html = (folder / meta.get("template", "template.html")).read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as e:
        raise TemplateError(f"{folder.name}: {e}") from e
    for k in ("name", "title", "build"):
        if k not in meta:
            raise TemplateError(f"{folder.name}: type.json missing {k!r}")
    pack = TemplatePack(meta["name"], meta["title"], meta.get("spec", {}), meta["build"], meta.get("rules", []), meta.get("params", {}), html,
                        meta.get("required", []), folder)
    pack.validate()
    return pack


def discover(extra: list[Path] | None = None) -> dict[str, TemplatePack]:
    dirs = [BUILTIN_DIR] + ([Path(os.environ["SDP_TEMPLATE_DIR"])] if os.environ.get("SDP_TEMPLATE_DIR") else []) + list(extra or [])
    out: dict[str, TemplatePack] = {}
    for d in dirs:
        if d.exists():
            for f in sorted(d.iterdir()):
                if (f / "type.json").exists():
                    p = load_pack(f)
                    out[p.name] = p
    return out


# ------------------------------------------------------------- generation
def _s(v: Any) -> Any:
    """JSON-friendly: Decimal -> plain string, date -> ISO, containers recursively."""
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _s(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_s(x) for x in v]
    return v


def _typed(kind: str, v: Any) -> Any:
    if v is None:
        return None
    if kind in NUMERIC:
        return Decimal(str(v))
    if kind == "date":
        return v if isinstance(v, date) else date.fromisoformat(v)
    return v


class Builder:
    def __init__(self, pack: TemplatePack, locale: str, region: str | None, seed: int, params: dict[str, Any], overrides: dict[str, Any]) -> None:
        self.pack, self.params, self.overrides = pack, params, overrides
        self.locale = get_locale(locale)
        self.rng = np.random.default_rng(seed)
        self.dp = self.locale.data["currency"]["decimals"]
        _, rate = self.locale.tax_rate(region)
        self.ev = Evaluator(self.dp, Decimal(self.locale.data["scale"]["income_median"]), rate, self.rng)
        self.factor = Decimal(self.locale.data["scale"]["price_factor"])

    # -- one value
    def value(self, spec: dict[str, Any], ctx: dict[str, Any]) -> Any:
        k, r, p = spec["kind"], self.rng, self.locale
        if k == "const":
            return spec["value"]
        if k == "choice":
            vals, w = spec["values"], spec.get("weights")
            pick = vals[int(r.choice(len(vals), p=None if not w else np.array(w) / sum(w)))]
            return _typed(spec["type"], pick) if spec.get("type") else pick
        if k == "int":
            return Decimal(int(r.integers(spec["min"], spec["max"] + 1)))
        if k == "decimal":
            return Decimal(str(round(float(r.uniform(spec["min"], spec["max"])), spec.get("places", 2))))
        if k == "money":
            base = Decimal(str(round(float(r.uniform(spec["min"], spec["max"])), 2)))
            return quantize(base * self.factor, self.dp)
        if k == "income":
            m = Decimal(str(round(float(r.uniform(spec.get("lo", 0.6), spec.get("hi", 2.5))), 3)))
            return (self.ev.income_median * m / 10).to_integral_value(rounding=ROUND_HALF_UP) * 10
        if k == "id":
            return spec.get("prefix", "") + "".join(str(int(x)) for x in r.integers(0, 10, spec.get("digits", 6)))
        if k == "date":
            a, b = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
            return a + timedelta(days=int(r.integers(0, (b - a).days + 1)))
        if k == "month_period":
            m = int(r.integers(1, 13))
            y = spec.get("year", 2025)
            start = date(y, m, 1)
            end = (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)
            pay = end
            while pay.weekday() >= 5:
                pay -= timedelta(days=1)
            return {"start": start, "end": end, "label": f"{MONTHS[m - 1]} {y}", "pay_date": pay}
        if k == "person":
            first_last = p.person_name(r, latin=True)
            addr, city = p.address(r, latin=True)
            f, l = first_last.split()[0], first_last.split()[-1]
            return {"name": first_last, "address": addr, "city": city, "national_id": p.national_id(r), "phone": p.phone(r), "email": email_for(p, f, l, r)}
        if k == "company":
            last = p.person_name(r, latin=True).split()[-1]
            suffixes = spec.get("suffixes", ["Trading Co.", "Ltd", "Industries", "Partners"])
            addr, city = p.address(r, latin=True)
            return {"name": f"{last} {suffixes[int(r.integers(0, len(suffixes)))]}", "address": addr, "city": city,
                    "tax_id": "TAX-" + "".join(str(int(x)) for x in r.integers(0, 10, 8)), "phone": p.phone(r)}
        if k == "expr":
            return self.ev.eval(spec["expr"], ctx)
        raise TemplateError(f"cannot generate kind {k!r}")

    def group_rows(self, spec: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        n_lines = self.params.get("n_lines") if spec.get("param") == "n_lines" else None
        if "rows" in spec:  # fixed rows, each optionally present with a probability
            for tpl in spec["rows"]:
                if self.rng.random() >= tpl.get("probability", 1.0):
                    continue
                row: dict[str, Any] = {}
                for key, v in tpl.items():
                    if key in ("probability", "recompute"):
                        continue
                    row[key] = self.ev.eval(v[1:], {**ctx, **row}) if isinstance(v, str) and v.startswith("=") else v
                rows.append(row)
        else:
            lo, hi = spec.get("count", [1, 3])
            n = int(n_lines) if n_lines else int(self.rng.integers(lo, hi + 1))
            for _ in range(n):
                row = {}
                if spec.get("source") == "catalog":
                    item = CATALOG[int(self.rng.integers(0, len(CATALOG)))]
                    base = Decimal(str(round(float(self.rng.uniform(item["min"], item["max"])), 2)))
                    row.update(description=item["description"], category=item["category"], unit_price=quantize(base * self.factor, self.dp))
                for key, fs in spec.get("fields", {}).items():
                    row[key] = self.value(fs, {**ctx, **row})
                rows.append(row)
        for key, expr in spec.get("row_computed", {}).items():
            for row in rows:
                row[key] = self.ev.eval(expr, {**ctx, **row})
        return rows

    def build(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        for b in self.pack.build:
            name, kind = b["name"], b["kind"]
            if kind == "group":
                ctx[name] = self.group_rows(b, ctx)
            elif kind == "computed":
                ctx[name] = self.ev.eval(b["expr"], ctx)
            else:
                ctx[name] = self.value(b, ctx)
            if name in self.overrides:
                ctx[name] = _typed(b.get("type", kind), self.overrides[name])
        for path, v in self.overrides.items():  # dotted overrides into dict fields, e.g. employee.name
            if "." in path:
                head, tail = path.split(".", 1)
                ctx[head][tail] = v
        return ctx


def _group_types(b: dict[str, Any]) -> dict[str, str]:
    types = dict(b.get("types", {}))
    for k in b.get("row_computed", {}):
        types.setdefault(k, "money")
    for k, fs in b.get("fields", {}).items():
        types.setdefault(k, fs["kind"] if fs["kind"] in NUMERIC | {"date"} else "str")
    if b.get("source") == "catalog":
        types.setdefault("unit_price", "money")
    return types


def _typed_ctx(pack: TemplatePack, doc: dict[str, Any]) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    for b in pack.build:
        v = doc.get(b["name"])
        k = b["kind"]
        t = b.get("type", k)
        if k == "group":
            types = _group_types(b)
            ctx[b["name"]] = [{key: _typed(types.get(key, "str"), val) for key, val in row.items()} for row in v]
        elif k == "computed":
            ctx[b["name"]] = _typed("decimal", v) if isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v) else v
        elif k == "month_period":
            ctx[b["name"]] = {**v, "start": date.fromisoformat(v["start"]), "end": date.fromisoformat(v["end"]), "pay_date": date.fromisoformat(v["pay_date"])}
        else:
            ctx[b["name"]] = _typed(t, v)
    return ctx


def reconcile(pack: TemplatePack, doc: dict[str, Any]) -> list[str]:
    """Independent re-check from the JSON: recompute `computed` values and flagged row fields, then evaluate every rule."""
    pack_locale = get_locale(doc["locale"])
    _, rate = pack_locale.tax_rate(doc.get("region"))
    ev = Evaluator(pack_locale.data["currency"]["decimals"], Decimal(pack_locale.data["scale"]["income_median"]), rate, None)
    errs: list[str] = []
    try:
        ctx = _typed_ctx(pack, doc)
    except (KeyError, ValueError, TypeError) as e:
        return [f"document does not match its schema: {e}"]
    try:
        # forward recompute in build order using stored values for everything that is not recomputed
        for b in pack.build:
            name = b["name"]
            if b["kind"] == "computed":
                expect = ev.eval(b["expr"], ctx)
                if ctx[name] != expect:
                    errs.append(f"{name}: stored {format(ctx[name], 'f')} != recomputed {format(expect, 'f') if isinstance(expect, Decimal) else expect}")
                ctx[name] = expect
            elif b["kind"] == "group":
                for row in ctx[name]:
                    for key in b.get("row_computed", {}):
                        expect = ev.eval(b["row_computed"][key], {**ctx, **row})
                        if row[key] != expect:
                            errs.append(f"{name}[{row.get('description', '?')}].{key}: stored {row[key]} != recomputed {expect}")
                    for tpl in b.get("rows", []):
                        if tpl.get("description") == row.get("description"):
                            for key in tpl.get("recompute", []):
                                expect = ev.eval(tpl[key][1:], {**ctx, **row})
                                if row[key] != expect:
                                    errs.append(f"{name}[{row['description']}].{key}: stored {row[key]} != recomputed {expect}")
        for r in pack.rules:
            try:
                if not ev.eval(r["expr"], ctx):
                    errs.append(f"rule failed: {r.get('name', r['expr'])}")
            except (TemplateError, KeyError, TypeError) as e:
                errs.append(f"rule error ({r.get('name', r['expr'])}): {e}")
    except (TemplateError, KeyError, TypeError, ArithmeticError, ValueError, AttributeError) as e:
        errs.append(f"document is incomplete or inconsistent: {type(e).__name__}: {e}")
    return errs


# --------------------------------------------------------------- rendering
def render_html(pack: TemplatePack, doc: dict[str, Any]) -> str:
    pk = get_locale(doc["locale"])
    env = SandboxedEnvironment(autoescape=True, undefined=StrictUndefined)
    def money(v: Any) -> str:
        text = pk.format_currency(Decimal(str(v))).replace(" ", " ")
        try:
            text.encode("cp1252")
            return text
        except UnicodeEncodeError:  # e.g. the rupee sign is not in the standard PDF font: use the ISO code, as invoices do
            return f"{pk.data['currency']['code']} {pk.format_number(Decimal(str(v))).replace(chr(0x202f), chr(0xa0))}"
    env.filters["money"] = money
    env.filters["date"] = lambda v: pk.format_date(date.fromisoformat(v)) if isinstance(v, str) else pk.format_date(v)
    env.filters["num"] = lambda v: pk.format_number(Decimal(str(v)))
    env.filters["pct"] = lambda v: format((Decimal(str(v)) * 100).normalize(), "f") + "%"
    return env.from_string(pack.html).render(doc=doc, currency=doc["currency"], tax_label=pk.data["tax"]["label"])


def html_to_pdf(html: str) -> bytes:
    from xhtml2pdf import pisa
    buf = io.BytesIO()
    res = pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    if res.err:
        raise TemplateError(f"HTML to PDF conversion failed ({res.err} errors)")
    return buf.getvalue()


def unrenderable_chars(html: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<style.*?</style>", "", html, flags=re.S))
    bad = set()
    for ch in text:
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            bad.add(ch)
    return sorted(bad)


def make_doc_type(pack: TemplatePack):
    """Turn a pack into the (spec class, generate, reconcile, render, html, required) the registry needs. Pure data in, callables out."""
    from pydantic import model_validator

    from sdp.documents.types import _Base

    names = {b["name"]: b for b in pack.build}
    fields: dict[str, Any] = {"overrides": (dict[str, Any], Field(default_factory=dict))}
    for pname, pdef in pack.params.items():
        fields[pname] = (int, Field(pdef.get("default", 3), ge=pdef.get("min", 1), le=pdef.get("max", 100)))
    Base = create_model(f"{pack.name}_params", __base__=_Base, **fields)

    class TemplateSpec(Base):  # type: ignore[valid-type, misc]
        @model_validator(mode="after")
        def _check_overrides(self):
            for path, v in self.overrides.items():  # unknown / group / oversized overrides are rejected at validation time
                head = path.split(".")[0]
                if head not in names or names[head]["kind"] == "group":
                    raise ValueError(f"cannot override {path!r}; overridable: {sorted(n for n, b in names.items() if b['kind'] != 'group')}")
                if isinstance(v, (int, float, str)) and len(str(v)) > 40:
                    raise ValueError(f"override value for {path!r} is too long")
            return self
    TemplateSpec.__name__ = f"{pack.name}Spec"

    def generate(spec) -> dict[str, Any]:
        pk = get_locale(spec.locale)
        params = {p: getattr(spec, p) for p in pack.params}
        ctx = Builder(pack, spec.locale, spec.region, spec.seed, params, spec.overrides).build()
        return {"doc_type": pack.name, "locale": pk.code, "region": spec.region, "currency": pk.data["currency"]["code"],
                "tax_label": pk.data["tax"]["label"],
                "presentation": {"native": False, "native_digits": False, "direction": "ltr", "script": "latin",
                                 "notes": ["template documents render in Latin script"] if spec.native else []},
                "document_id": hashlib.sha256((pack.name + spec.model_dump_json()).encode()).hexdigest()[:12].upper(), **_s(ctx)}

    def render(doc: dict[str, Any], font: str = "Helvetica", strict: bool = False):
        from sdp.documents.render import RenderResult, count_pdf_pages
        html = render_html(pack, doc)
        pdf = html_to_pdf(html)
        return RenderResult(pdf=pdf, pages=count_pdf_pages(pdf), fonts_used=["Helvetica"], unrenderable=unrenderable_chars(html), boxes=[])

    required = tuple(["document_id", *pack.required])
    return TemplateSpec, generate, (lambda doc: reconcile(pack, doc)), render, (lambda doc: render_html(pack, doc)), required
