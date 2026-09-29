"""P1. Pluggable exporters. An exporter turns tables (whole DataFrames or an iterator of chunks, for streaming) into bytes on a sink.

    @register_exporter
    class MyFormat:
        name, extension, mime = "myfmt", "my", "text/plain"
        def export(self, tables, schema, sink, **options): ...

`schema` is a `dict[str, TableSchema]` (types, nullability, primary and foreign keys) so formats such as SQL can preserve structure.
Hidden bookkeeping columns (leading underscore) never leave the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, BinaryIO, Iterable, Iterator, Protocol, Union

import pandas as pd

TableData = Union[pd.DataFrame, Iterable[pd.DataFrame]]


class ExportError(ValueError):
    pass


@dataclass
class ColumnDef:
    name: str
    kind: str                    # int | float | str | bool | date | datetime
    nullable: bool = True
    max_len: int | None = None   # for str
    unique: bool = False


@dataclass
class ForeignKeyDef:
    columns: list[str]
    ref_table: str
    ref_columns: list[str]


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnDef]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeyDef] = field(default_factory=list)

    def column(self, name: str) -> ColumnDef:
        return next(c for c in self.columns if c.name == name)


def kind_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        nn = s.dropna()
        return "date" if len(nn) and bool((nn == nn.dt.normalize()).all()) else "datetime"
    return "str"


def visible_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not str(c).startswith("_")]


def schema_from(tables: dict[str, pd.DataFrame], graph: Any | None = None) -> dict[str, TableSchema]:
    """Derive column types/nullability from the data, and keys from a `RelationshipGraph` when given."""
    out: dict[str, TableSchema] = {}
    for name, df in tables.items():
        cols = []
        for c in visible_columns(df):
            s = df[c]
            k = kind_of(s)
            ml = int(s.dropna().astype(str).str.len().max()) if k == "str" and s.notna().any() else None
            cols.append(ColumnDef(name=str(c), kind=k, nullable=bool(s.isna().any()), max_len=ml))
        out[name] = TableSchema(name=name, columns=cols)
    if graph is not None:
        for t in graph.tables:
            if t.name in out:
                ts = out[t.name]
                names = {c.name for c in ts.columns}
                ts.primary_key = [c for c in t.primary_key if c in names]
                for c in ts.columns:
                    if c.name in ts.primary_key:
                        c.nullable = False
        for fk in graph.foreign_keys:
            if fk.child_table in out and fk.parent_table in out:
                out[fk.child_table].foreign_keys.append(ForeignKeyDef(list(fk.child_columns), fk.parent_table, list(fk.parent_columns)))
    return out


def chunks(data: TableData, size: int = 50_000) -> Iterator[pd.DataFrame]:
    """Normalise a DataFrame or an iterable of DataFrames into chunks (a lone frame is split so writers stay bounded)."""
    if isinstance(data, pd.DataFrame):
        for i in range(0, max(len(data), 1), size):
            yield data.iloc[i:i + size]
    else:
        yield from data


class Exporter(Protocol):
    name: str
    extension: str
    mime: str

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, **options: Any) -> dict[str, Any]: ...


REGISTRY: dict[str, Exporter] = {}


def register_exporter(cls: Any) -> Any:
    inst = cls() if isinstance(cls, type) else cls
    for attr in ("name", "extension", "mime", "export"):
        if not hasattr(inst, attr):
            raise ExportError(f"exporter is missing {attr!r}")
    REGISTRY[inst.name] = inst
    return cls


def get_exporter(name: str) -> Exporter:
    if name not in REGISTRY:
        raise ExportError(f"unknown export format {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def export_bytes(tables: dict[str, TableData], fmt: str, schema: dict[str, TableSchema] | None = None, **options: Any) -> tuple[bytes, dict[str, Any]]:
    import io
    if schema is None:
        frames = {k: v for k, v in tables.items() if isinstance(v, pd.DataFrame)}
        if len(frames) != len(tables):
            raise ExportError("a schema is required when exporting streamed chunks")
        schema = schema_from(frames, options.pop("graph", None))
    buf = io.BytesIO()
    info = get_exporter(fmt).export(tables, schema, buf, **options)
    return buf.getvalue(), info
