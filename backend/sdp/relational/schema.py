"""Relationship graph: tables, primary keys, foreign keys. JSON in/out and editable."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Column(BaseModel):
    name: str
    dtype: Literal["int", "float", "str", "bool", "datetime"] = "str"
    nullable: bool = True


class Table(BaseModel):
    name: str
    columns: list[Column]
    primary_key: list[str] = Field(default_factory=list)  # composite allowed
    row_count: int | None = None

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}.{name}")


class ForeignKey(BaseModel):
    child_table: str
    child_columns: list[str]
    parent_table: str
    parent_columns: list[str]
    cardinality: Literal["1:1", "1:N"] = "1:N"
    nullable: bool = True
    confidence: float = 1.0
    source: Literal["inferred", "ddl", "user"] = "inferred"
    stats: dict[str, float] = Field(default_factory=dict)  # mean/max children per parent, orphan_rate, ...

    @property
    def key(self) -> str:
        """Stable id used by cardinality configs and reports."""
        return f"{self.child_table}({','.join(self.child_columns)})->{self.parent_table}"

    @property
    def is_self_reference(self) -> bool:
        return self.child_table == self.parent_table


class RelationshipGraph(BaseModel):
    tables: list[Table]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)

    # ------------------------------------------------------------- lookups
    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(f"unknown table {name!r}")

    def fks_of(self, child: str) -> list[ForeignKey]:
        return [f for f in self.foreign_keys if f.child_table == child]

    def fks_to(self, parent: str) -> list[ForeignKey]:
        return [f for f in self.foreign_keys if f.parent_table == parent]

    def fk(self, key: str) -> ForeignKey:
        for f in self.foreign_keys:
            if f.key == key:
                return f
        raise KeyError(key)

    def junction_tables(self) -> list[str]:
        """Tables with >=2 FKs to distinct parents: the N:N bridges."""
        out = []
        for t in self.tables:
            parents = {f.parent_table for f in self.fks_of(t.name) if not f.is_self_reference}
            if len(parents) >= 2:
                out.append(t.name)
        return out

    def many_to_many(self) -> list[dict[str, str]]:
        res = []
        for j in self.junction_tables():
            ps = sorted({f.parent_table for f in self.fks_of(j) if not f.is_self_reference})
            res += [{"junction": j, "left": a, "right": b} for i, a in enumerate(ps) for b in ps[i + 1:]]
        return res

    # ------------------------------------------------------------ ordering
    def topological_order(self) -> list[str]:
        """Parents before children (self-references ignored). Raises on cycles."""
        deps = {t.name: {f.parent_table for f in self.fks_of(t.name) if not f.is_self_reference}
                for t in self.tables}
        order: list[str] = []
        ready = sorted(n for n, d in deps.items() if not d)
        done: set[str] = set()
        while ready:
            n = ready.pop(0)
            order.append(n)
            done.add(n)
            ready = sorted(set(ready) | {m for m, d in deps.items() if m not in done and m not in ready and d <= done})
        if len(order) != len(deps):
            raise ValueError(f"cyclic foreign keys among: {sorted(set(deps) - done)}")
        return order

    # ---------------------------------------------------------- validation
    def validate_graph(self) -> list[str]:
        """Human-readable structural problems (empty list = graph is usable)."""
        errs: list[str] = []
        names = [t.name for t in self.tables]
        if len(set(names)) != len(names):
            errs.append("duplicate table names")
        for t in self.tables:
            cols = {c.name for c in t.columns}
            for pk in t.primary_key:
                if pk not in cols:
                    errs.append(f"{t.name}: primary key column {pk!r} does not exist")
        for f in self.foreign_keys:
            try:
                child, parent = self.table(f.child_table), self.table(f.parent_table)
            except KeyError as e:
                errs.append(f"{f.key}: {e.args[0]}")
                continue
            if len(f.child_columns) != len(f.parent_columns):
                errs.append(f"{f.key}: column count mismatch")
            for c in f.child_columns:
                if c not in {x.name for x in child.columns}:
                    errs.append(f"{f.key}: child column {c!r} missing")
            if sorted(f.parent_columns) != sorted(parent.primary_key):
                errs.append(f"{f.key}: parent columns {f.parent_columns} are not the primary key {parent.primary_key} of {parent.name}")
        if not errs:
            try:
                self.topological_order()
            except ValueError as e:
                errs.append(str(e))
        return errs

    # ------------------------------------------------------------- editing
    def set_primary_key(self, table: str, columns: list[str]) -> None:
        self.table(table).primary_key = list(columns)

    def add_foreign_key(self, child_table: str, child_columns: list[str], parent_table: str,
                        parent_columns: list[str] | None = None, cardinality: Literal["1:1", "1:N"] = "1:N",
                        nullable: bool = True) -> ForeignKey:
        parent_columns = parent_columns or list(self.table(parent_table).primary_key)
        fk = ForeignKey(child_table=child_table, child_columns=child_columns, parent_table=parent_table,
                        parent_columns=parent_columns, cardinality=cardinality, nullable=nullable, source="user")
        if any(f.key == fk.key for f in self.foreign_keys):
            raise ValueError(f"foreign key already exists: {fk.key}")
        self.foreign_keys.append(fk)
        return fk

    def remove_foreign_key(self, key: str) -> None:
        self.fk(key)
        self.foreign_keys = [f for f in self.foreign_keys if f.key != key]

    def update_foreign_key(self, key: str, **changes) -> ForeignKey:
        old = self.fk(key)
        new = old.model_copy(update={**changes, "source": "user"})
        self.foreign_keys = [new if f.key == key else f for f in self.foreign_keys]
        return new

    def add_many_to_many(self, left: str, right: str, junction: str | None = None) -> Table:
        """Declare an N:N by creating a junction table with a composite PK over two FKs."""
        junction = junction or f"{left}_{right}"
        lp, rp = self.table(left), self.table(right)
        if len(lp.primary_key) != 1 or len(rp.primary_key) != 1:
            raise ValueError("N:N helper needs single-column primary keys on both sides")
        lc, rc = f"{_singular(left)}_{lp.primary_key[0]}", f"{_singular(right)}_{rp.primary_key[0]}"
        if left == right:
            lc, rc = f"{lc}_a", f"{rc}_b"
        cols = [Column(name=lc, dtype=lp.column(lp.primary_key[0]).dtype, nullable=False),
                Column(name=rc, dtype=rp.column(rp.primary_key[0]).dtype, nullable=False)]
        table = Table(name=junction, columns=cols, primary_key=[lc, rc])
        self.tables.append(table)
        self.add_foreign_key(junction, [lc], left, nullable=False)
        self.add_foreign_key(junction, [rc], right, nullable=False)
        return table

    # ---------------------------------------------------------------- json
    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "RelationshipGraph":
        return cls.model_validate_json(text)

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["many_to_many"] = self.many_to_many()
        for f, fk in zip(d["foreign_keys"], self.foreign_keys):
            f["key"] = fk.key
        return d


def _singular(name: str) -> str:
    n = name.lower()
    if n.endswith("ies"):
        return n[:-3] + "y"
    return n[:-1] if n.endswith("s") and not n.endswith("ss") else n
