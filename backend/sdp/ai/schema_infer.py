"""A1. Schema understanding from a small sample (20-50 rows).

Pipeline: deterministic heuristics -> (optional) Claude refinement -> every suggestion verified against the sample -> validated
`SchemaProposal` JSON -> user confirms or edits (`apply_edits`). The LLM can never assert something the data contradicts:
a proposed type must match the sample values, a proposed constraint must hold on the sample (else it is kept but flagged).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sdp.ai import llm

SEMANTIC_TYPES = [
    "person_name", "first_name", "last_name", "email", "phone", "national_id", "identifier", "currency_amount", "currency_code",
    "date", "datetime", "percentage", "boolean", "category", "integer", "decimal", "free_text", "address", "city", "country",
    "postal_code", "url", "ip_address", "uuid", "other",
]
SemType = Literal[tuple(SEMANTIC_TYPES)]  # type: ignore[valid-type]
MIN_ROWS, LLM_ROWS = 5, 50
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%d %b %Y", "%b %d, %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
CONSTRAINT_TYPES = ("not_null", "unique", "range", "in_set", "regex", "order")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_PHONE = re.compile(r"^\+?[\d\s().-]{7,20}$")
_URL = re.compile(r"^https?://\S+$")
_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MONEY = re.compile(r"^\s*(?:[$€£¥₹]|Rs\.?|PKR|USD|EUR|GBP|INR)?\s*-?\d[\d,]*(?:\.\d+)?\s*(?:[$€£¥₹]|PKR|USD|EUR|GBP|INR)?\s*$")
_ID_NAME = re.compile(r"(^|_)(id|uuid|code|key|no|number|ref)$|^id(_|$)")

NAME_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"e-?mail"), "email"), (re.compile(r"phone|mobile|tel\b|cell|fax"), "phone"),
    (re.compile(r"first.?name|given.?name|forename"), "first_name"), (re.compile(r"last.?name|surname|family.?name"), "last_name"),
    (re.compile(r"(^|_)(full_?)?name$|customer_name|patient_name|person"), "person_name"),
    (re.compile(r"national.?id|cnic|ssn|passport|aadhaar|nin\b|tax.?id"), "national_id"),
    (re.compile(r"currency|ccy"), "currency_code"), (re.compile(r"amount|price|balance|salary|income|total|cost|fee|revenue|payment|charge"), "currency_amount"),
    (re.compile(r"percent|pct|rate$|ratio"), "percentage"),
    (re.compile(r"zip|postal|postcode|pin.?code"), "postal_code"), (re.compile(r"address|street"), "address"), (re.compile(r"(^|_)city|town"), "city"),
    (re.compile(r"country|nation"), "country"), (re.compile(r"url|website|link"), "url"), (re.compile(r"(^|_)ip(_|$)"), "ip_address"),
    (re.compile(r"comment|review|description|notes?|message|text|body|summary|feedback"), "free_text"),
]


class ColumnProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    semantic_type: SemType = "other"
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    dtype: str = "str"
    nullable: bool = True
    unique: bool = False
    date_format: str | None = None
    currency: str | None = None
    allowed_values: list[str] | None = None
    min: float | None = None
    max: float | None = None
    source: Literal["heuristic", "llm", "user"] = "heuristic"
    reason: str = ""


class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    source: Literal["heuristic", "llm", "user"] = "heuristic"


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["not_null", "unique", "range", "in_set", "regex", "order"]
    table: str
    columns: list[str]
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    source: Literal["heuristic", "llm", "user"] = "heuristic"
    holds_on_sample: bool = True


class SchemaProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tables: dict[str, list[ColumnProposal]]
    relationships: list[Relationship] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    used_llm: bool = False
    confirmed: bool = False
    proposal_hash: str = ""

    def column(self, table: str, name: str) -> ColumnProposal:
        for c in self.tables[table]:
            if c.name == name:
                return c
        raise KeyError(f"{table}.{name}")

    def seal(self) -> "SchemaProposal":
        body = self.model_dump(exclude={"proposal_hash", "confirmed"})
        self.proposal_hash = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return self


class SchemaError(ValueError):
    pass


# ------------------------------------------------------------------ heuristics
def _strings(s: pd.Series) -> pd.Series:
    return s.dropna().astype(str).str.strip().loc[lambda x: x != ""]


def _frac(s: pd.Series, pat: re.Pattern[str]) -> float:
    v = _strings(s)
    return float(v.map(lambda x: bool(pat.match(x))).mean()) if len(v) else 0.0


def detect_date_format(s: pd.Series) -> str | None:
    """The strptime format that parses >= 90% of the non-null values, preferring the earliest in DATE_FORMATS."""
    v = _strings(s)
    if v.empty or _frac(s, re.compile(r"^-?\d+(\.\d+)?$")) > 0.5:
        return None
    for fmt in DATE_FORMATS:
        ok = pd.to_datetime(v, format=fmt, errors="coerce").notna().mean()
        if ok >= 0.9:
            return fmt
    return None


def _numeric(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(s):
        return None
    if pd.api.types.is_numeric_dtype(s):
        return s.dropna().astype(float)
    n = pd.to_numeric(_strings(s).str.replace(r"[$€£¥₹,\s]|Rs\.?|PKR|USD|EUR|GBP|INR", "", regex=True), errors="coerce")
    return n.dropna().astype(float) if len(n) and n.notna().mean() >= 0.95 else None


def _heuristic_column(name: str, s: pd.Series, n: int) -> ColumnProposal:
    lname = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    nn = s.dropna()
    nunique = int(nn.nunique())
    p = ColumnProposal(name=name, dtype=str(s.dtype), nullable=bool(s.isna().any()), unique=bool(len(nn) and nunique == len(nn) and n >= MIN_ROWS))
    hint = next((t for pat, t in NAME_HINTS if pat.search(lname)), None)

    def done(t: str, conf: float, why: str, **kw: Any) -> ColumnProposal:
        return p.model_copy(update={"semantic_type": t, "confidence": conf, "reason": why, **kw})

    if pd.api.types.is_bool_dtype(s) or set(map(str, nn.unique())) <= {"true", "false", "True", "False", "yes", "no", "Y", "N", "0", "1"} and 0 < nunique <= 2 and not pd.api.types.is_float_dtype(s):
        return done("boolean", 0.85, "two-valued flag")
    if pd.api.types.is_datetime64_any_dtype(s):
        return done("date", 0.95, "datetime dtype", date_format="%Y-%m-%d")
    if _frac(s, _UUID) > 0.9:
        return done("uuid", 0.98, "matches UUID pattern")
    if _frac(s, _EMAIL) > 0.9:
        return done("email", 0.98, "matches e-mail pattern")
    if _frac(s, _URL) > 0.9:
        return done("url", 0.95, "matches URL pattern")
    if _frac(s, _IP) > 0.9:
        return done("ip_address", 0.9, "matches IPv4 pattern")
    fmt = detect_date_format(s)
    if fmt:
        has_time = "%H" in fmt
        return done("datetime" if has_time else "date", 0.93, f"parses with {fmt}", date_format=fmt)
    num = _numeric(s)
    if num is not None and len(num):
        lo, hi = float(num.min()), float(num.max())
        is_int = bool((num % 1 == 0).all())
        raw_money = s.dtype == object or str(s.dtype) == "str"
        rng = {"min": lo, "max": hi}
        if hint == "currency_amount" or (raw_money and _frac(s, _MONEY) > 0.9 and re.search(r"[$€£¥₹]|Rs|PKR|USD|EUR|GBP|INR", " ".join(_strings(s).head(10)))):
            sym = re.search(r"[$€£¥₹]|Rs\.?|PKR|USD|EUR|GBP|INR", " ".join(_strings(s).head(20)))
            return done("currency_amount", 0.85, "amount-like name/values", currency=sym.group(0) if sym else None, **rng)
        if hint == "percentage" or (0 <= lo and hi <= 100 and not is_int and "pct" in lname):
            return done("percentage", 0.8, "percentage-like name", **rng)
        if _ID_NAME.search(lname) or (p.unique and is_int and lname.endswith("id")):
            return done("identifier", 0.9 if p.unique else 0.7, "id-like name" + (" and unique values" if p.unique else ""))
        if hint == "postal_code":
            return done("postal_code", 0.75, "postal-code-like name")
        if hint == "phone" and _frac(s, _PHONE) > 0.8:
            return done("phone", 0.9, "phone-like name and values")
        if is_int and nunique <= 2 and n >= MIN_ROWS:
            return done("boolean", 0.7, "integer with two values", **rng)
        return done("integer" if is_int else "decimal", 0.8, "numeric values", **rng)
    if hint == "email" and _frac(s, _EMAIL) > 0.5:
        return done("email", 0.9, "e-mail-like name")
    if _frac(s, _PHONE) > 0.9 and (hint == "phone" or _strings(s).str.len().median() >= 9):
        return done("phone", 0.9 if hint == "phone" else 0.7, "matches phone pattern")
    if hint in ("person_name", "first_name", "last_name", "national_id", "address", "city", "country", "postal_code", "url", "phone", "email", "currency_code"):
        conf = 0.85 if hint != "national_id" else 0.8
        return done(hint, conf, f"column name suggests {hint}")
    if _ID_NAME.search(lname):
        return done("identifier", 0.75, "id-like name")
    vals = _strings(s)
    if len(vals) and nunique <= max(10, n // 4) and vals.str.len().max() <= 40:
        vc = sorted(map(str, nn.unique()))
        return done("category", 0.8, f"{nunique} distinct short values", allowed_values=vc[:30])
    if len(vals) and vals.str.split().str.len().median() >= 4:
        return done("free_text", 0.8, "multi-word text")
    return done("other", 0.3, "no strong signal")


def _heuristic_constraints(table: str, df: pd.DataFrame, cols: list[ColumnProposal]) -> list[Constraint]:
    out: list[Constraint] = []
    for c in cols:
        s = df[c.name]
        if not s.isna().any() and len(s) >= MIN_ROWS:
            out.append(Constraint(type="not_null", table=table, columns=[c.name], description=f"{c.name} is never empty in the sample"))
        if c.unique or c.semantic_type in ("identifier", "uuid") and s.dropna().is_unique:
            out.append(Constraint(type="unique", table=table, columns=[c.name], description=f"{c.name} values are distinct"))
        if c.semantic_type in ("integer", "decimal", "currency_amount", "percentage") and c.min is not None and c.max is not None:
            params: dict[str, Any] = {"max": c.max}
            if c.min >= 0:
                params["min"] = 0.0
            else:
                params["min"] = c.min
            out.append(Constraint(type="range", table=table, columns=[c.name], params=params, description=f"{c.name} between {params['min']:g} and {c.max:g}"))
        if c.semantic_type == "category" and c.allowed_values:
            out.append(Constraint(type="in_set", table=table, columns=[c.name], params={"values": c.allowed_values}, description=f"{c.name} is one of {len(c.allowed_values)} values"))
    dates = [c for c in cols if c.semantic_type in ("date", "datetime") and c.date_format]
    for a in dates:
        for b in dates:
            if a.name < b.name and (_is_start(a.name) or _is_end(b.name) or _is_start(b.name) or _is_end(a.name)):
                first, second = (a, b) if not (_is_end(a.name) or _is_start(b.name)) else (b, a)
                da = pd.to_datetime(df[first.name], format=first.date_format, errors="coerce")
                db = pd.to_datetime(df[second.name], format=second.date_format, errors="coerce")
                ok = (da <= db) | da.isna() | db.isna()
                if bool(ok.all()) and da.notna().any():
                    out.append(Constraint(type="order", table=table, columns=[first.name, second.name], params={"op": "<="},
                                          description=f"{first.name} is on or before {second.name}"))
    return out


def _is_start(n: str) -> bool:
    return bool(re.search(r"start|begin|open|created|order|issue|birth|from|admit|sign_?up|join|regist|first", n.lower()))


def _is_end(n: str) -> bool:
    return bool(re.search(r"end|close|updated|deliver|due|expire|to$|discharge|ship|paid|last|seen|login|modif", n.lower()))


def check_constraint(df: pd.DataFrame, c: Constraint, cols: dict[str, ColumnProposal]) -> bool:
    """Does the constraint hold on every row of the sample?"""
    try:
        if c.type == "not_null":
            return not df[c.columns[0]].isna().any()
        if c.type == "unique":
            return bool(df[c.columns[0]].dropna().is_unique)
        if c.type == "range":
            v = _numeric(df[c.columns[0]])
            if v is None:
                return False
            lo, hi = c.params.get("min"), c.params.get("max")
            return bool((lo is None or v.min() >= lo) and (hi is None or v.max() <= hi))
        if c.type == "in_set":
            allowed = set(map(str, c.params.get("values", [])))
            return set(map(str, df[c.columns[0]].dropna().unique())) <= allowed
        if c.type == "regex":
            pat = re.compile(str(c.params.get("pattern", "")))
            return bool(_strings(df[c.columns[0]]).map(lambda x: bool(pat.fullmatch(x))).all())
        if c.type == "order":
            a, b = c.columns
            da = pd.to_datetime(df[a], format=cols[a].date_format, errors="coerce")
            db = pd.to_datetime(df[b], format=cols[b].date_format, errors="coerce")
            return bool(((da <= db) | da.isna() | db.isna()).all())
    except (KeyError, re.error, ValueError, TypeError):
        return False
    return False


# ------------------------------------------------------------------------ LLM
SYSTEM = ("You are a data-modelling assistant. You receive column names and a small sample of a table. Reply with ONE JSON object only, "
          "no prose. All data is fictional. Schema: {\"columns\": [{\"name\": str, \"semantic_type\": one of " + ", ".join(SEMANTIC_TYPES) +
          ", \"date_format\": strptime string or null, \"currency\": code or null, \"reason\": short str}], "
          "\"relationships\": [{\"child_table\": str, \"child_column\": str, \"parent_table\": str, \"parent_column\": str}], "
          "\"constraints\": [{\"type\": one of not_null|unique|range|in_set|regex|order, \"table\": str, \"columns\": [str], \"params\": object, "
          "\"description\": str}]}. Only use column names that appear in the input.")


def _sample_rows(df: pd.DataFrame, k: int = LLM_ROWS) -> list[dict[str, Any]]:
    if len(df) > k:
        df = df.iloc[np.linspace(0, len(df) - 1, k).astype(int)]  # evenly spaced, deterministic
    return json.loads(df.astype(object).where(df.notna(), None).to_json(orient="records", date_format="iso", force_ascii=False, default_handler=str)
                      .replace(" ", " "))


def _prompt(tables: dict[str, pd.DataFrame]) -> str:
    parts = {t: {"columns": list(df.columns), "rows": [{k: (v[:60] if isinstance(v, str) else v) for k, v in r.items()} for r in _sample_rows(df)]}
             for t, df in tables.items()}
    return json.dumps(parts, ensure_ascii=False)


def _merge_llm(reply: Any, tables: dict[str, pd.DataFrame], prop: SchemaProposal, warnings: list[str]) -> None:
    if not isinstance(reply, dict):
        warnings.append("LLM reply was not a JSON object; kept heuristic result.")
        return
    tname = next(iter(tables)) if len(tables) == 1 else None
    for item in reply.get("columns", []) if isinstance(reply.get("columns"), list) else []:
        try:
            table = item.get("table", tname)
            name = item["name"]
            st = item["semantic_type"]
            if table not in prop.tables or name not in tables[table].columns:
                warnings.append(f"LLM mentioned unknown column {name!r}; ignored.")
                continue
            if st not in SEMANTIC_TYPES:
                warnings.append(f"LLM proposed unknown type {st!r} for {name}; ignored.")
                continue
        except (AttributeError, KeyError, TypeError):
            warnings.append("Malformed LLM column entry ignored.")
            continue
        cur = prop.column(table, name)
        df = tables[table]
        fmt = item.get("date_format")
        if st in ("date", "datetime"):
            if not fmt or pd.to_datetime(_strings(df[name]), format=fmt, errors="coerce").notna().mean() < 0.9:
                warnings.append(f"LLM proposed {st} for {name} but its format does not parse the sample; ignored.")
                continue
        elif st == "email" and _frac(df[name], _EMAIL) < 0.8:
            warnings.append(f"LLM proposed email for {name} but values are not e-mail-shaped; ignored.")
            continue
        elif st in ("integer", "decimal", "currency_amount", "percentage") and _numeric(df[name]) is None:
            warnings.append(f"LLM proposed {st} for {name} but values are not numeric; ignored.")
            continue
        if st != cur.semantic_type and (cur.confidence < 0.9 or st == cur.semantic_type):
            updates: dict[str, Any] = {"semantic_type": st, "source": "llm", "confidence": max(cur.confidence, 0.85), "reason": str(item.get("reason", ""))[:200]}
            if st in ("date", "datetime"):
                updates["date_format"] = fmt
            if item.get("currency") and isinstance(item["currency"], str):
                updates["currency"] = item["currency"][:8]
            prop.tables[table] = [c.model_copy(update=updates) if c.name == name else c for c in prop.tables[table]]
    cols_map = {t: {c.name: c for c in cs} for t, cs in prop.tables.items()}
    for r in reply.get("relationships", []) if isinstance(reply.get("relationships"), list) else []:
        try:
            rel = Relationship(**{**r, "source": "llm", "confidence": 0.6})
        except (TypeError, ValueError):
            warnings.append("Malformed LLM relationship ignored.")
            continue
        if rel.child_table in cols_map and rel.parent_table in cols_map and rel.child_column in cols_map[rel.child_table] and rel.parent_column in cols_map[rel.parent_table]:
            child, parent = tables[rel.child_table][rel.child_column].dropna().astype(str), tables[rel.parent_table][rel.parent_column].dropna().astype(str)
            overlap = float(child.isin(set(parent)).mean()) if len(child) else 0.0
            if overlap < 0.8 and rel.child_table != rel.parent_table:
                warnings.append(f"LLM relationship {rel.child_table}.{rel.child_column} -> {rel.parent_table}.{rel.parent_column} does not hold in the sample ({overlap:.0%}); ignored.")
                continue
            key = (rel.child_table, rel.child_column, rel.parent_table, rel.parent_column)
            if key not in {(x.child_table, x.child_column, x.parent_table, x.parent_column) for x in prop.relationships}:
                prop.relationships.append(rel.model_copy(update={"confidence": max(0.6, min(0.95, overlap))}))
        else:
            warnings.append("LLM relationship references unknown columns; ignored.")
    for c in reply.get("constraints", []) if isinstance(reply.get("constraints"), list) else []:
        try:
            con = Constraint(**{**c, "source": "llm"})
        except (TypeError, ValueError):
            warnings.append("Malformed LLM constraint ignored.")
            continue
        if con.table not in tables or any(x not in tables[con.table].columns for x in con.columns):
            warnings.append("LLM constraint references unknown columns; ignored.")
            continue
        if con.type == "order" and (len(con.columns) != 2 or any(cols_map[con.table][x].date_format is None for x in con.columns)):
            warnings.append("LLM order constraint needs two date columns; ignored.")
            continue
        con.holds_on_sample = check_constraint(tables[con.table], con, cols_map[con.table])
        if not con.holds_on_sample:
            warnings.append(f"LLM constraint '{con.description or con.type}' is violated by the sample; kept but flagged.")
        if not any(x.type == con.type and x.table == con.table and x.columns == con.columns and x.params == con.params for x in prop.constraints):
            prop.constraints.append(con)


# ------------------------------------------------------------------- public API
def infer_schema(tables: pd.DataFrame | dict[str, pd.DataFrame], client: llm.LLMClient | None | bool = True, name: str = "table") -> SchemaProposal:
    """Propose semantic types, relationships and constraints. `client=True` uses the configured LLM if any; None/False = offline only."""
    if isinstance(tables, pd.DataFrame):
        tables = {name: tables}
    if not tables:
        raise SchemaError("no tables supplied")
    for t, df in tables.items():
        if len(df) < MIN_ROWS:
            raise SchemaError(f"table {t!r} has {len(df)} rows; at least {MIN_ROWS} are needed (20-50 recommended)")
        if df.columns.duplicated().any():
            raise SchemaError(f"table {t!r} has duplicate column names")
    warnings: list[str] = []
    props = {t: [_heuristic_column(str(c), df[c], len(df)) for c in df.columns] for t, df in tables.items()}
    for t, df in tables.items():
        if len(df) < 20:
            warnings.append(f"{t}: only {len(df)} rows; suggestions are less reliable below 20.")
    prop = SchemaProposal(tables=props)
    # relationships from the existing relational inference (multi-table) or a self-reference guess (single table)
    if len(tables) > 1:
        from sdp.relational.inference import infer_graph
        try:
            g = infer_graph(tables)
            for fk in g.foreign_keys:
                if len(fk.child_columns) == 1:
                    prop.relationships.append(Relationship(child_table=fk.child_table, child_column=fk.child_columns[0], parent_table=fk.parent_table,
                                                           parent_column=fk.parent_columns[0], confidence=float(fk.confidence)))
        except Exception as e:  # noqa: BLE001
            warnings.append(f"relationship inference skipped: {type(e).__name__}")
    else:
        t, df = next(iter(tables.items()))
        pk = next((c.name for c in props[t] if c.semantic_type == "identifier" and c.unique), None)
        for c in props[t]:
            if pk and c.name != pk and c.semantic_type in ("identifier", "integer") and re.search(r"(manager|parent|supervisor|referr)", c.name.lower()):
                v = df[c.name].dropna()
                if len(v) and v.isin(df[pk]).mean() >= 0.8:
                    prop.relationships.append(Relationship(child_table=t, child_column=c.name, parent_table=t, parent_column=pk, confidence=0.7))
    for t, df in tables.items():
        cmap = {c.name: c for c in prop.tables[t]}
        prop.constraints += _heuristic_constraints(t, df, prop.tables[t])
        for con in prop.constraints:
            if con.table == t:
                con.holds_on_sample = check_constraint(df, con, cmap)
    c = llm.get_client() if client is True else (client or None)
    if c is not None:
        try:
            text, _ = llm.cached_complete(c, SYSTEM, _prompt(tables), max_tokens=2048)
            _merge_llm(llm.extract_json(text), tables, prop, warnings)
            prop.used_llm = True
        except llm.LLMError as e:
            warnings.append(f"LLM unavailable ({e}); showing local heuristics only.")
    prop.warnings = warnings
    return SchemaProposal.model_validate(prop.model_dump()).seal()


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["set_type", "set_format", "remove_constraint", "add_constraint", "remove_relationship", "add_relationship", "confirm"]
    table: str | None = None
    column: str | None = None
    value: Any = None


def apply_edits(proposal: SchemaProposal, edits: list[dict[str, Any] | Edit]) -> SchemaProposal:
    """Apply user edits (validated) and return a new sealed proposal. `confirm` marks it as reviewed."""
    p = proposal.model_copy(deep=True)
    for raw in edits:
        e = raw if isinstance(raw, Edit) else Edit.model_validate(raw)
        if e.action == "confirm":
            p.confirmed = True
            continue
        if e.action in ("set_type", "set_format"):
            if e.table not in p.tables or not any(c.name == e.column for c in p.tables[e.table]):
                raise SchemaError(f"unknown column {e.table}.{e.column}")
            if e.action == "set_type":
                if e.value not in SEMANTIC_TYPES:
                    raise SchemaError(f"unknown semantic type {e.value!r}")
                upd: dict[str, Any] = {"semantic_type": e.value, "source": "user", "confidence": 1.0}
            else:
                try:
                    pd.Timestamp("2024-01-31").strftime(str(e.value))
                except ValueError as ex:
                    raise SchemaError(f"bad date format: {ex}") from ex
                upd = {"date_format": e.value, "source": "user"}
            p.tables[e.table] = [c.model_copy(update=upd) if c.name == e.column else c for c in p.tables[e.table]]
        elif e.action == "remove_constraint":
            idx = int(e.value)
            if not 0 <= idx < len(p.constraints):
                raise SchemaError(f"no constraint #{idx}")
            p.constraints.pop(idx)
        elif e.action == "add_constraint":
            con = Constraint.model_validate({**(e.value or {}), "source": "user"})
            if con.table not in p.tables or any(not any(c.name == x for c in p.tables[con.table]) for x in con.columns):
                raise SchemaError("constraint references unknown columns")
            p.constraints.append(con)
        elif e.action == "remove_relationship":
            idx = int(e.value)
            if not 0 <= idx < len(p.relationships):
                raise SchemaError(f"no relationship #{idx}")
            p.relationships.pop(idx)
        elif e.action == "add_relationship":
            rel = Relationship.model_validate({**(e.value or {}), "source": "user", "confidence": 1.0})
            for t, c in ((rel.child_table, rel.child_column), (rel.parent_table, rel.parent_column)):
                if t not in p.tables or not any(x.name == c for x in p.tables[t]):
                    raise SchemaError(f"unknown column {t}.{c}")
            p.relationships.append(rel)
    return SchemaProposal.model_validate(p.model_dump()).seal()


def to_rules(proposal: SchemaProposal) -> dict[str, list[str]]:
    """Confirmed constraints as rule-engine DSL strings per table (only those that hold on the sample)."""
    out: dict[str, list[str]] = {}
    for c in proposal.constraints:
        if not c.holds_on_sample:
            continue
        col = c.columns[0]
        rule = None
        if c.type == "range":
            lo, hi = c.params.get("min"), c.params.get("max")
            parts = ([f"{col} >= {lo:g}"] if lo is not None else []) + ([f"{col} <= {hi:g}"] if hi is not None else [])
            out.setdefault(c.table, []).extend(parts)
        elif c.type == "order":
            rule = f"{c.columns[1]} >= {c.columns[0]}"
        if rule:
            out.setdefault(c.table, []).append(rule)
    return out
