"""A6. Conversational editing: "double customers from Lahore", "add more outliers to balance", "fraud 5%".

A chat message becomes a list of typed ops (`interpret`, rule-based and multilingual; an optional LLM only proposes ops that are then
validated the same way). Ops are applied INCREMENTALLY to the current tables: only the affected rows/columns/tables change, everything else
is byte-identical, and the result reports exactly which parts were regenerated plus a before/after summary and a config diff.

State is `(base DatasetConfig, ops history)`. Every op has a seed derived from the config seed and its position, so replaying the state
reproduces the tables exactly; the server stays stateless (the client sends the state back with the next message).
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Annotated, Any, Literal, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from sdp.ai import llm
from sdp.locale.coherent import strip_diacritics
from sdp.nl.builder import generate_dataset
from sdp.nl.multilingual import detect_language, msg, to_canonical
from sdp.nl.parser import DatasetConfig, FlagSpec, domains

MAX_ROWS = 200_000


class ChatEditError(ValueError):
    pass


# ---------------------------------------------------------------------- ops
class SetRows(BaseModel):
    op: Literal["set_rows"] = "set_rows"
    rows: int = Field(ge=1, le=MAX_ROWS)


class ScaleRows(BaseModel):
    op: Literal["scale_rows"] = "scale_rows"
    factor: float = Field(gt=0, le=50)


class ScaleSegment(BaseModel):
    op: Literal["scale_segment"] = "scale_segment"
    column: str
    value: str
    factor: float = Field(ge=0, le=50)


class SetFlagRate(BaseModel):
    op: Literal["set_flag_rate"] = "set_flag_rate"
    rate: float = Field(ge=0, le=1)


class SetOutlierRate(BaseModel):
    op: Literal["set_outlier_rate"] = "set_outlier_rate"
    column: str
    rate: float = Field(ge=0, le=0.5)


class SetHistoryMonths(BaseModel):
    op: Literal["set_history_months"] = "set_history_months"
    months: int = Field(ge=1, le=120)


Op = Annotated[Union[SetRows, ScaleRows, ScaleSegment, SetFlagRate, SetOutlierRate, SetHistoryMonths], Field(discriminator="op")]
_OP = TypeAdapter(Op)


class EditState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: DatasetConfig
    ops: list[dict[str, Any]] = Field(default_factory=list)

    def key(self) -> str:
        return hashlib.sha256(json.dumps([self.config.model_dump(mode="json"), self.ops], sort_keys=True, default=str).encode()).hexdigest()


def _seed(cfg: DatasetConfig, i: int) -> int:
    return int(cfg.seed) * 100_003 + 7_919 * (i + 1)


# ---------------------------------------------------------------- helpers
def _names(cfg: DatasetConfig) -> tuple[dict[str, Any], str, str | None, str]:
    dom = domains()[cfg.domain]
    return dom, dom["table"], (dom["history"]["table"] if "history" in dom else None), dom["id_column"]


def _match(col: pd.Series, value: str) -> pd.Series:
    key = strip_diacritics(str(value)).casefold().strip()
    return col.astype(str).map(lambda x: strip_diacritics(x).casefold().strip()) == key


def _drop_customers(tables: dict[str, pd.DataFrame], cfg: DatasetConfig, drop: np.ndarray) -> dict[str, pd.DataFrame]:
    dom, main, hist, idc = _names(cfg)
    out = dict(tables)
    ids = tables[main][idc].to_numpy()[drop]
    out[main] = tables[main].loc[~np.isin(tables[main][idc], ids)].reset_index(drop=True)
    if hist:
        out[hist] = tables[hist].loc[~np.isin(tables[hist][idc], ids)].reset_index(drop=True)
    return out


def _append_rows(tables: dict[str, pd.DataFrame], cfg: DatasetConfig, k: int, seed: int, column: str | None = None, value: str | None = None) -> dict[str, pd.DataFrame]:
    """Generate `k` more customers (optionally only those where `column` == `value`) with their history and append them with fresh ids."""
    dom, main, hist, idc = _names(cfg)
    if k <= 0:
        return tables
    got_m: list[pd.DataFrame] = []
    got_h: list[pd.DataFrame] = []
    need, attempt = k, 0
    next_id = int(tables[main][idc].max()) + 1 if len(tables[main]) else 1
    while need > 0 and attempt < 25:
        batch = int(min(MAX_ROWS, max(50, need * (4 if column else 1) + 20)))
        ds = generate_dataset(cfg.model_copy(update={"rows": batch, "seed": seed + attempt}))
        m = ds.tables[main]
        if column is not None:
            m = m[_match(m[column], value or "")]
        m = m.head(need).copy()
        if len(m):
            remap = {int(old): next_id + i for i, old in enumerate(m[idc])}   # ids are per batch, so remap before batches are combined
            next_id += len(m)
            m[idc] = [remap[int(x)] for x in m[idc]]
            got_m.append(m)
            if hist:
                h = ds.tables[hist][ds.tables[hist][idc].isin(list(remap))].copy()
                h[idc] = [remap[int(x)] for x in h[idc]]
                got_h.append(h)
            need -= len(m)
        attempt += 1
    if need > 0:
        raise ChatEditError(f"could not generate {k} rows with {column} = {value!r}; the generator produces too few of them")
    out = dict(tables)
    out[main] = pd.concat([tables[main], *got_m], ignore_index=True)
    if hist and got_h:
        out[hist] = pd.concat([tables[hist], *got_h], ignore_index=True)
    return out


def _risk_score(df: pd.DataFrame, dom: dict[str, Any]) -> np.ndarray:
    """The same risk ranking the generator uses (without its random noise), so flag edits pick plausible customers."""
    from scipy import stats
    score = np.zeros(len(df))
    for col, w in dom["flag"]["risk"].items():
        if col not in df:
            continue
        v = df[col]
        if pd.api.types.is_datetime64_any_dtype(v):
            score += w * stats.zscore(-v.astype("int64").to_numpy(dtype=float))
        else:
            score += w * stats.zscore(np.log1p(v.to_numpy(dtype=float)))
    return np.nan_to_num(score)


def _set_flag(tables: dict[str, pd.DataFrame], cfg: DatasetConfig, rate: float, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    dom, main, hist, idc = _names(cfg)
    flag = dom["flag"]["name"]
    df = tables[main].copy()
    n = len(df)
    target = int(round(rate * n))
    cur = df[flag].to_numpy() == 1
    out = dict(tables)
    turn_on = np.array([], dtype=int)
    turn_off = np.array([], dtype=int)
    if target > cur.sum():
        cand = np.flatnonzero(~cur)
        score = _risk_score(df, dom)[cand] + rng.gumbel(size=len(cand))
        turn_on = cand[np.argsort(-score)[: target - int(cur.sum())]]
    elif target < cur.sum():
        cand = np.flatnonzero(cur)
        turn_off = rng.choice(cand, size=int(cur.sum()) - target, replace=False)
    df.loc[df.index[turn_on], flag] = 1
    df.loc[df.index[turn_off], flag] = 0
    out[main] = df
    if hist:
        h = tables[hist].copy()
        act = f"{flag}_activity"
        hd = dom["history"]
        ids_on, ids_off = df[idc].to_numpy()[turn_on], df[idc].to_numpy()[turn_off]
        for cid in ids_on:                              # give newly flagged customers a few spike months
            rows = np.flatnonzero(h[idc].to_numpy() == cid)
            if not len(rows):
                continue
            picks = rng.choice(rows, size=min(len(rows), int(rng.integers(1, 4))), replace=False)
            mult = rng.uniform(*hd["spike_count_multiplier"], size=len(picks))
            newc = np.round(h["txn_count"].to_numpy()[picks] * mult).astype(int)
            ratio = newc / np.maximum(h["txn_count"].to_numpy()[picks], 1)
            h.loc[h.index[picks], "txn_amount"] = np.round(h["txn_amount"].to_numpy()[picks] * ratio * hd["spike_ticket_multiplier"], 2)
            h.loc[h.index[picks], "txn_count"] = newc
            h.loc[h.index[picks], act] = 1
        for cid in ids_off:                             # replace an unflagged customer's spikes by their ordinary months
            rows = np.flatnonzero(h[idc].to_numpy() == cid)
            spike = rows[h[act].to_numpy()[rows] == 1]
            normal = rows[h[act].to_numpy()[rows] == 0]
            base = normal if len(normal) else rows
            if len(spike):
                h.loc[h.index[spike], "txn_count"] = int(np.median(h["txn_count"].to_numpy()[base]))
                h.loc[h.index[spike], "txn_amount"] = float(np.median(h["txn_amount"].to_numpy()[base]))
                h.loc[h.index[spike], act] = 0
        out[hist] = h
    return out


def _set_months(tables: dict[str, pd.DataFrame], cfg: DatasetConfig, months: int, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    dom, main, hist, idc = _names(cfg)
    if not hist:
        raise ChatEditError(f"domain {cfg.domain} has no history table")
    h = tables[hist]
    cur = int(h.groupby(idc)["month"].nunique().max()) if len(h) else 0
    out = dict(tables)
    if months < cur:
        keep = sorted(h["month"].unique())[-months:]
        out[hist] = h[h["month"].isin(keep)].reset_index(drop=True)
    elif months > cur:
        first = pd.Timestamp(h["month"].min())
        extra_months = [(first - pd.DateOffset(months=i)) for i in range(months - cur, 0, -1)]
        per = h.groupby(idc).agg(mean_count=("txn_count", "mean"), amount=("txn_amount", "sum"), count=("txn_count", "sum"))
        ticket = (per["amount"] / per["count"].clip(lower=1)).to_dict()
        lam = per["mean_count"].to_dict()
        rows = []
        act = [c for c in h.columns if c.endswith("_activity")]
        for cid in per.index:
            for mth in extra_months:
                c = int(rng.poisson(max(lam[cid], 0.1)))
                r = {idc: cid, "month": mth, "txn_count": c, "txn_amount": round(c * ticket[cid] * float(np.exp(0.2 * rng.standard_normal())), 2)}
                r.update({a: 0 for a in act})
                rows.append(r)
        add = pd.DataFrame(rows, columns=h.columns).astype(h.dtypes.to_dict(), errors="ignore")
        out[hist] = pd.concat([add, h], ignore_index=True).sort_values([idc, "month"], kind="stable").reset_index(drop=True)
    return out


def _set_outliers(tables: dict[str, pd.DataFrame], cfg: DatasetConfig, column: str, rate: float, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    dom, main, hist, idc = _names(cfg)
    df = tables[main].copy()
    if column not in df or not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]) or column == idc:
        raise ChatEditError(f"{column!r} is not a numeric column")
    mark = f"_outlier_{column}"
    if mark not in df:
        df[mark] = False
    cur = df[mark].to_numpy(dtype=bool)
    n = len(df)
    target = int(round(rate * n))
    clean = df.loc[~df[mark].astype(bool), column].astype(float)
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = max(q3 - q1, 1e-9)
    nonneg = float(clean.min()) >= 0
    is_int = pd.api.types.is_integer_dtype(df[column])
    if target > cur.sum():
        cand = np.flatnonzero(~cur)
        pick = rng.choice(cand, size=min(len(cand), target - int(cur.sum())), replace=False)
        up = np.ones(len(pick), dtype=bool) if nonneg else rng.random(len(pick)) < 0.5
        mag = rng.uniform(1.5, 3.0, size=len(pick)) * iqr
        vals = np.where(up, q3 + mag, q1 - mag)
        df.loc[df.index[pick], column] = np.round(vals).astype("int64") if is_int else np.round(vals, 2)
        df.loc[df.index[pick], mark] = True
    elif target < cur.sum():
        cand = np.flatnonzero(cur)
        pick = rng.choice(cand, size=int(cur.sum()) - target, replace=False)
        repl = rng.choice(clean.to_numpy(), size=len(pick))
        df.loc[df.index[pick], column] = repl.astype("int64") if is_int else np.round(repl, 2)
        df.loc[df.index[pick], mark] = False
    out = dict(tables)
    out[main] = df
    return out


def _apply(tables: dict[str, pd.DataFrame], cfg: DatasetConfig, eff: DatasetConfig, op: Any, seed: int) -> tuple[dict[str, pd.DataFrame], DatasetConfig]:
    dom, main, hist, idc = _names(cfg)
    rng = np.random.default_rng(seed)
    n = len(tables[main])
    if isinstance(op, (SetRows, ScaleRows)):
        target = op.rows if isinstance(op, SetRows) else max(1, int(round(n * op.factor)))
        if target > MAX_ROWS:
            raise ChatEditError(f"at most {MAX_ROWS:,} rows are supported here")
        if target > n:
            return _append_rows(tables, eff, target - n, seed), eff
        if target < n:
            drop = rng.choice(n, size=n - target, replace=False)
            return _drop_customers(tables, cfg, drop), eff
        return tables, eff
    if isinstance(op, ScaleSegment):
        if op.column not in tables[main].columns:
            raise ChatEditError(f"unknown column {op.column!r}")
        mask = _match(tables[main][op.column], op.value).to_numpy()
        k = int(mask.sum())
        target = int(round(k * op.factor))
        if k == 0 and op.factor > 0:
            raise ChatEditError(f"no rows with {op.column} = {op.value!r}")
        if target > k:
            if len(tables[main]) + target - k > MAX_ROWS:
                raise ChatEditError(f"at most {MAX_ROWS:,} rows are supported here")
            return _append_rows(tables, eff, target - k, seed, op.column, op.value), eff
        if target < k:
            drop = rng.choice(np.flatnonzero(mask), size=k - target, replace=False)
            return _drop_customers(tables, cfg, drop), eff
        return tables, eff
    if isinstance(op, SetFlagRate):
        flag = dom["flag"]["name"]
        eff = eff.model_copy(update={"flag": FlagSpec(name=flag, rate=op.rate)})
        return _set_flag(tables, cfg, op.rate, rng), eff
    if isinstance(op, SetHistoryMonths):
        eff = eff.model_copy(update={"history_months": op.months})
        return _set_months(tables, cfg, op.months, rng), eff
    if isinstance(op, SetOutlierRate):
        return _set_outliers(tables, cfg, op.column, op.rate, rng), eff
    raise ChatEditError(f"unsupported op {op!r}")


def parse_op(d: dict[str, Any]) -> Any:
    try:
        return _OP.validate_python(d)
    except ValueError as e:
        raise ChatEditError(f"invalid edit: {e}") from e


@lru_cache(maxsize=8)
def _materialize_cached(key: str, state_json: str) -> tuple[dict[str, pd.DataFrame], str]:
    state = EditState.model_validate_json(state_json)
    cfg = state.config
    tables = {k: v.copy() for k, v in generate_dataset(cfg).tables.items()}
    eff = cfg
    for i, raw in enumerate(state.ops):
        tables, eff = _apply(tables, cfg, eff, parse_op(raw), _seed(cfg, i))
    return tables, eff.model_dump_json()


def materialize(state: EditState) -> tuple[dict[str, pd.DataFrame], DatasetConfig]:
    tables, eff = _materialize_cached(state.key(), state.model_dump_json())
    return {k: v.copy() for k, v in tables.items()}, DatasetConfig.model_validate_json(eff)


# ---------------------------------------------------------------- summaries
def summarize(tables: dict[str, pd.DataFrame], cfg: DatasetConfig) -> dict[str, Any]:
    dom, main, hist, idc = _names(cfg)
    m = tables[main]
    out: dict[str, Any] = {"rows": len(m)}
    if dom["flag"]["name"] in m:
        out["flag_rate"] = round(float(m[dom["flag"]["name"]].mean()), 4)
    if hist:
        h = tables[hist]
        out["history_months"] = int(h.groupby(idc)["month"].nunique().max()) if len(h) else 0
    segs = {}
    for c in m.columns:
        if c == idc or str(c).startswith("_") or not (m[c].dtype == object or str(m[c].dtype) in ("str", "string")):
            continue
        vc = m[c].value_counts()
        if 1 < len(vc) <= 12:
            segs[c] = {str(k): int(v) for k, v in vc.items()}
    out["segments"] = segs
    out["outliers"] = {c[len("_outlier_"):]: int(m[c].sum()) for c in m.columns if str(c).startswith("_outlier_")}
    out["medians"] = {c: round(float(m[c].median()), 2) for c in m.columns if pd.api.types.is_numeric_dtype(m[c]) and not pd.api.types.is_bool_dtype(m[c]) and c != idc and not str(c).startswith("_") and c != dom["flag"]["name"]}
    return out


def config_diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat list of {path, before, after} for every summary value that changed."""
    rows: list[dict[str, Any]] = []

    def walk(path: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) or isinstance(b, dict):
            a, b = a or {}, b or {}
            for k in sorted(set(a) | set(b), key=str):
                walk(f"{path}.{k}" if path else str(k), a.get(k), b.get(k))
        elif a != b:
            rows.append({"path": path, "before": a, "after": b})
    walk("", before, after)
    return rows


def regenerated(before: dict[str, pd.DataFrame], after: dict[str, pd.DataFrame], cfg: DatasetConfig) -> dict[str, Any]:
    """Which tables/columns actually changed (everything else is byte-identical)."""
    dom, main, hist, idc = _names(cfg)
    out: dict[str, Any] = {}
    for t, a in after.items():
        b = before[t]
        info: dict[str, Any] = {"rows_before": len(b), "rows_after": len(a)}
        new_cols = [c for c in a.columns if c not in b.columns]
        common = min(len(a), len(b))
        changed = list(new_cols)
        if common:
            aa, bb = a.iloc[:common].reset_index(drop=True), b.iloc[:common].reset_index(drop=True)
            for c in b.columns:
                if c in a.columns and not aa[c].equals(bb[c]):
                    changed.append(c)
        info["columns_changed"] = sorted(set(changed))
        info["untouched"] = len(a) == len(b) and not info["columns_changed"]
        out[t] = info
    return out


# ------------------------------------------------------------ interpretation
_ROWS_NOUN = r"(?:customers?|clientes?|clients?|users?|rows?|records?|filas?|lignes?|shoppers?|people)"
_FLAG_WORDS = {"fraud", "fraudulent", "chargeback", "chargebacks"}


def _factor(clause: str) -> float | None:
    if re.search(r"\b(double|twice|2x)\b|\b2 times\b", clause):
        return 2.0
    if re.search(r"\btriple\b|\b3x\b|\b3 times\b", clause):
        return 3.0
    if re.search(r"\b(half|halve|halved)\b", clause):
        return 0.5
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", clause)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(more|fewer|less|higher|lower)", clause) or re.search(r"(?:by|up by|down by)\s+(\d+(?:\.\d+)?)\s*%", clause)
    if m:
        p = float(m.group(1)) / 100
        neg = (len(m.groups()) > 1 and m.group(2) in ("fewer", "less", "lower")) or re.search(r"\b(reduce|decrease|remove|drop|cut|fewer|less|lower)\b", clause)
        return 1 - p if neg else 1 + p
    if re.search(r"\b(more|increase|add|raise|higher|extra)\b", clause):
        return 1.5
    if re.search(r"\b(fewer|less|reduce|decrease|lower|remove|cut)\b", clause):
        return 0.7
    return None


def _column_for(clause: str, cols: list[str]) -> list[str]:
    words = set(re.findall(r"[a-z]+", clause))
    scored = []
    for c in cols:
        parts = set(c.lower().split("_"))
        hit = len(parts & words)
        if hit:
            scored.append((hit, c))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    best = scored[0][0]
    return [c for h, c in scored if h == best]


def interpret_rules(text: str, state: EditState, current: dict[str, pd.DataFrame], summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """English-canonical text -> ops. Returns (ops, questions). Ambiguity becomes a question, never a guess."""
    dom, main, hist, idc = _names(state.config)
    m = current[main]
    ops: list[dict[str, Any]] = []
    questions: list[str] = []
    t = strip_diacritics(text.lower())
    for clause in [c for c in re.split(r"\s+and\s+|;|,\s+then\s+|\s+then\s+|,(?=\s*[a-z])", t) if c.strip()]:
        made = False
        # outliers
        if re.search(r"outlier|anomal", clause):
            numeric = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c]) and not pd.api.types.is_bool_dtype(m[c]) and c != idc and not str(c).startswith("_") and c != dom["flag"]["name"]]
            cand = _column_for(clause, numeric)
            if len(cand) != 1:
                questions.append(f"Which column should get outliers? Options: {', '.join(numeric)}.")
                continue
            col = cand[0]
            cur = summary["outliers"].get(col, 0) / max(1, summary["rows"])
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", clause)
            f = _factor(clause)
            rate = float(pct.group(1)) / 100 if pct else (max(cur * 2, cur + 0.02) if f and f >= 1 else cur * (f or 0.5) if f else max(cur, 0.03))
            ops.append({"op": "set_outlier_rate", "column": col, "rate": round(min(rate, 0.5), 4)})
            continue
        # fraud / flag
        if any(w in clause for w in _FLAG_WORDS):
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", clause)
            cur = summary.get("flag_rate", 0.0)
            f = _factor(clause)
            if pct and not re.search(r"%\s*(more|fewer|less|higher|lower)|by\s+\d", clause):
                ops.append({"op": "set_flag_rate", "rate": float(pct.group(1)) / 100})
            elif f:
                ops.append({"op": "set_flag_rate", "rate": round(min(1.0, cur * f), 4)})
            else:
                questions.append("What fraud rate do you want (for example 'fraud 5%' or 'double the fraud')?")
            continue
        # history months
        hm = re.search(r"(\d+)\s*months?", clause)
        if hm and re.search(r"history|months?", clause) and hist and not re.search(_ROWS_NOUN, clause):
            ops.append({"op": "set_history_months", "months": int(hm.group(1))})
            continue
        # segment: any distinct value of a categorical column mentioned in the clause
        seg = None
        nclause = " " + re.sub(r"[^a-z0-9]+", " ", clause) + " "
        for col, counts in summary["segments"].items():
            for v in counts:
                if f" {re.sub(r'[^a-z0-9]+', ' ', strip_diacritics(v.lower())).strip()} " in nclause:
                    seg = (col, v)
                    break
            if seg:
                break
        if seg:
            f = _factor(clause)
            if f is None:
                questions.append(f"By how much should {seg[1]} change (for example 'double' or '50% more')?")
                continue
            ops.append({"op": "scale_segment", "column": seg[0], "value": seg[1], "factor": f})
            continue
        # total rows
        nm = re.search(rf"(\d[\d,]*)\s*(k|thousand)?\s*{_ROWS_NOUN}|{_ROWS_NOUN}\s*(?:to|=)\s*(\d[\d,]*)\s*(k|thousand)?", clause)
        if nm:
            raw = nm.group(1) or nm.group(3)
            unit = nm.group(2) or nm.group(4)
            n = int(float(raw.replace(",", "")) * (1000 if unit else 1))
            ops.append({"op": "set_rows", "rows": n})
            continue
        if re.search(_ROWS_NOUN, clause):
            f = _factor(clause)
            if f:
                ops.append({"op": "scale_rows", "factor": f})
                continue
        if not made:
            questions.append(f"I could not tell what to change in '{clause.strip()}'.")
    return ops, questions


_OPS_SYSTEM = ("You translate a user's chat message into edits for a synthetic dataset. Reply with ONE JSON array of ops, no prose. Allowed ops: "
               '{"op":"set_rows","rows":int} {"op":"scale_rows","factor":number} {"op":"scale_segment","column":str,"value":str,"factor":number} '
               '{"op":"set_flag_rate","rate":0-1} {"op":"set_outlier_rate","column":str,"rate":0-0.5} {"op":"set_history_months","months":int}. '
               "Use only the columns and values given. If unclear reply [].")


def interpret(text: str, state: EditState, language: str | None = None, client: llm.LLMClient | None | bool = True) -> dict[str, Any]:
    tables, eff = materialize(state)
    summary = summarize(tables, eff)
    lang = language or detect_language(text)
    canon, _ = to_canonical(text, lang)
    ops, questions = interpret_rules(canon, state, tables, summary)
    source = "rules"
    c = llm.get_client() if client is True else (client or None)
    if not ops and c is not None:
        prompt = json.dumps({"message": text, "columns": list(tables[_names(state.config)[1]].columns), "segments": summary["segments"], "summary": {k: v for k, v in summary.items() if k != "medians"}}, ensure_ascii=False)
        try:
            reply, _ = llm.cached_complete(c, _OPS_SYSTEM, prompt, max_tokens=512)
            proposed = llm.extract_json(reply)
            valid = []
            if isinstance(proposed, list):
                for d in proposed:
                    op = parse_op(d)          # same validation as rule-based ops
                    valid.append(op.model_dump())
            if valid:
                ops, questions, source = valid, [], "llm"
        except (llm.LLMError, ChatEditError):
            pass
    return {"language": lang, "canonical_text": canon, "ops": ops, "questions": questions, "source": source}


def apply_message(state: EditState, text: str, language: str | None = None, client: llm.LLMClient | None | bool = True) -> dict[str, Any]:
    """Interpret the message, apply the ops incrementally, and report before/after, the diff and what was regenerated."""
    interp = interpret(text, state, language, client)
    lang = interp["language"]
    before_tables, before_cfg = materialize(state)
    before = summarize(before_tables, before_cfg)
    if not interp["ops"]:
        return {**interp, "state": json.loads(state.model_dump_json()), "applied": 0, "before": before, "after": before, "diff": [], "regenerated": {},
                "message": (interp["questions"][0] if interp["questions"] else msg(lang, "edit_none")), "preview": None}
    new_ops = [parse_op(o).model_dump() for o in interp["ops"]]
    new_state = EditState(config=state.config, ops=[*state.ops, *new_ops])
    after_tables, after_cfg = materialize(new_state)         # replay is deterministic, so `before_tables` equals the prefix result
    after = summarize(after_tables, after_cfg)
    diff = config_diff(before, after)
    regen = regenerated(before_tables, after_tables, state.config)
    main = _names(state.config)[1]
    return {**interp, "state": json.loads(new_state.model_dump_json()), "applied": len(new_ops), "before": before, "after": after, "diff": diff,
            "regenerated": regen, "message": msg(lang, "edit_applied", n=len(new_ops)),
            "preview": {"table": main, "rows": json.loads(after_tables[main].loc[:, [c for c in after_tables[main].columns if not str(c).startswith("_")]].head(20).to_json(orient="records", date_format="iso", force_ascii=False))}}
