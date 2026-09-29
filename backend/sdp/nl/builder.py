"""Turn a confirmed DatasetConfig into tables (customers + optional monthly history) using the locale pack."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sdp.locale import get_locale
from sdp.nl.parser import DatasetConfig, domains
from sdp.relational.integrity import check_integrity
from sdp.relational.schema import Column, ForeignKey, RelationshipGraph, Table
from sdp.scoring.constraints import check_constraints
from sdp.scoring.locale_validity import locale_validity

ANCHOR = date(2025, 12, 1)


@dataclass
class GeneratedDataset:
    tables: dict[str, pd.DataFrame]
    graph: RelationshipGraph
    config: DatasetConfig
    locale_columns: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        """Confirm what was asked for and that the output is valid (locale, rules, integrity, requested rates)."""
        main = self.tables[domains()[self.config.domain]["table"]]
        integrity = check_integrity(self.graph, self.tables).to_dict()
        out: dict[str, Any] = {
            "rows": {k: len(v) for k, v in self.tables.items()},
            "integrity": integrity,
            "constraints": check_constraints(main, self.config.rules),
            "locale": (locale_validity(main, locale_column="locale", columns=self.locale_columns) if self.config.locale_mix
                       else locale_validity(main, locale=self.config.locale, columns=self.locale_columns)),
        }
        if self.config.flag:
            achieved = float(main[self.config.flag.name].mean())
            out["flag"] = {"name": self.config.flag.name, "requested": self.config.flag.rate, "achieved": achieved,
                           "count": int(main[self.config.flag.name].sum())}
        return out


def _correlated_z(n: int, rng: np.random.Generator, cols: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    z: dict[str, np.ndarray] = {}
    for c in cols:
        if c["type"] != "money":
            continue
        base = rng.standard_normal(n)
        parent = c.get("corr_with")
        if parent in z:
            r = c["corr"]
            base = r * z[parent] + np.sqrt(1 - r * r) * base
        z[c["name"]] = base
    return z


def generate_dataset(cfg: DatasetConfig) -> GeneratedDataset:
    dom = domains()[cfg.domain]
    mix = cfg.locale_mix
    codes = list(mix) if mix else [cfg.locale]
    rng = np.random.default_rng(cfg.seed)
    n = cfg.rows
    idc = dom["id_column"]
    locs = rng.choice(codes, size=n, p=[mix[c] for c in codes]) if mix else np.array([cfg.locale] * n)
    packs = {c: get_locale(c) for c in codes}
    dp = 2
    income_median = np.array([float(packs[l].data["scale"]["income_median"]) for l in locs])

    df = pd.DataFrame({idc: np.arange(1, n + 1)})
    ident = dom["identity"]
    if "name" in ident:
        df["name"] = [packs[l].person_name(rng) for l in locs]
    if "phone" in ident:
        df["phone"] = [packs[l].phone(rng) for l in locs]
    if "national_id" in ident:
        df["national_id"] = [packs[l].national_id(rng) for l in locs]
    if "address" in ident:
        addrs = [packs[l].address(rng) for l in locs]
        df["address"], df["city"] = [a[0] for a in addrs], [a[1] for a in addrs]
    if "date_of_birth" in ident:
        df["date_of_birth"] = [packs[l].format_date(packs[l].birth_date(rng)) for l in locs]

    if mix:
        df["locale"] = locs
        df["currency"] = [packs[l].data["currency"]["code"] for l in locs]
    z = _correlated_z(n, rng, dom["columns"])
    score = rng.standard_normal(n)  # latent risk noise
    weights = dom["flag"]["risk"]
    for c in dom["columns"]:
        name, t = c["name"], c["type"]
        if t == "category":
            keys = list(c["values"])
            df[name] = rng.choice(keys, size=n, p=np.array(list(c["values"].values())) / sum(c["values"].values()))
        elif t == "money":
            median = income_median * c.get("median_multiple", 1.0)  # per-row: each row is priced in its own locale currency
            df[name] = np.round(median * np.exp(c["sigma"] * z[name]), dp)
        elif t == "date":
            lo, hi = c["min_years_ago"] * 365, c["max_years_ago"] * 365
            days = rng.integers(lo, hi + 1, size=n)
            df[name] = pd.Timestamp(ANCHOR) - pd.to_timedelta(days, unit="D")
    if cfg.flag:
        for col, w in weights.items():
            if col not in df:
                continue
            v = df[col]
            if pd.api.types.is_datetime64_any_dtype(v):
                score += w * stats.zscore(-v.astype("int64").to_numpy(dtype=float))  # newer -> riskier
            else:
                score += w * stats.zscore(np.log1p(v.to_numpy(dtype=float)))
        k = round(cfg.flag.rate * n)
        flagged = np.zeros(n, dtype=int)
        flagged[np.argsort(-score)[:k]] = 1  # exact requested count
        df[cfg.flag.name] = flagged

    tables = {dom["table"]: df}
    tcols = [Column(name=c, dtype=_dtype(df[c])) for c in df.columns]
    tables_schema = [Table(name=dom["table"], columns=tcols, primary_key=[idc], row_count=n)]
    fks: list[ForeignKey] = []

    if cfg.history_months and "history" in dom:
        h = dom["history"]
        months = cfg.history_months
        end = pd.Timestamp(h["end_month"])
        month_starts = [(end - pd.DateOffset(months=i)).date() for i in range(months - 1, -1, -1)]
        cust = np.repeat(df[idc].to_numpy(), months)
        month = np.tile(np.array(month_starts, dtype="datetime64[ns]"), n)
        activity = np.exp(0.5 * rng.standard_normal(n))
        if "monthly_income" in df:
            activity *= np.clip(df["monthly_income"].to_numpy() / income_median, 0.2, 5) ** 0.3
        lam = np.repeat(h["count_base"] * activity, months)
        spike = np.zeros(n * months, dtype=bool)
        cmult = np.ones(n * months)
        tmult = np.ones(n * months)
        if cfg.flag:
            lo, hi = h["spike_count_multiplier"]
            for i in np.flatnonzero(df[cfg.flag.name].to_numpy() == 1):
                picks = rng.choice(months, size=min(months, int(rng.integers(1, 4))), replace=False)
                idx = i * months + picks
                spike[idx] = True
                cmult[idx] = rng.uniform(lo, hi, size=len(idx))
                tmult[idx] = h["spike_ticket_multiplier"]
        count = rng.poisson(lam * cmult)
        ticket = np.repeat(income_median, months) * h["ticket_fraction_of_income"] * np.exp(0.4 * rng.standard_normal(n * months)) * tmult
        hist = pd.DataFrame({idc: cust, "month": month, "txn_count": count, "txn_amount": np.round(count * ticket, dp)})
        if cfg.flag:
            hist[f"{cfg.flag.name}_activity"] = spike.astype(int)
        tables[h["table"]] = hist
        tables_schema.append(Table(name=h["table"], columns=[Column(name=c, dtype=_dtype(hist[c])) for c in hist.columns],
                                   primary_key=[idc, "month"], row_count=len(hist)))
        fks.append(ForeignKey(child_table=h["table"], child_columns=[idc], parent_table=dom["table"], parent_columns=[idc],
                              cardinality="1:N", nullable=False, source="user"))

    graph = RelationshipGraph(tables=tables_schema, foreign_keys=fks)
    spec = {"national_id": "national_id" if "national_id" in df else None, "phone": "phone" if "phone" in df else None,
            "name": ["name"], "date": ["date_of_birth"] if "date_of_birth" in df else []}
    spec = {k: v for k, v in spec.items() if v}
    spec["text"] = [c for c in ("name", "address", "city") if c in df]
    return GeneratedDataset(tables=tables, graph=graph, config=cfg, locale_columns=spec)


def _dtype(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "str"
