"""SQL connectors (PostgreSQL, MySQL/MariaDB, SQLite) over any DB-API 2.0 driver."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import pandas as pd

from sdp.connectors.base import ConnectorError, LoadResult, Target, mask, parse_target, py_value
from sdp.exporters.base import TableData, TableSchema, chunks
from sdp.exporters.formats import create_table_sql, order_tables, quote_ident

MODES = ("create", "append", "replace")


def _driver_connect(t: Target) -> Any:
    if t.dialect == "sqlite":
        con = sqlite3.connect(str(t.path) if t.path else ":memory:", isolation_level=None)   # we manage BEGIN/COMMIT ourselves
        con.execute("PRAGMA foreign_keys = ON")
        return con
    p = urlparse(t.url)
    kw = {"host": t.host, "port": p.port, "user": unquote(p.username or ""), "password": unquote(p.password or "")}
    if t.dialect == "postgres":
        try:
            import psycopg  # type: ignore
            return psycopg.connect(host=kw["host"], port=kw["port"] or 5432, user=kw["user"], password=kw["password"], dbname=t.database, connect_timeout=10)
        except ImportError:
            pass
        try:
            import psycopg2  # type: ignore
            return psycopg2.connect(host=kw["host"], port=kw["port"] or 5432, user=kw["user"], password=kw["password"], dbname=t.database, connect_timeout=10)
        except ImportError as e:
            raise ConnectorError("PostgreSQL needs a driver: pip install psycopg[binary]") from e
    if t.dialect == "mysql":
        try:
            import pymysql  # type: ignore
            return pymysql.connect(host=kw["host"], port=kw["port"] or 3306, user=kw["user"], password=kw["password"], database=t.database, charset="utf8mb4", connect_timeout=10)
        except ImportError as e:
            raise ConnectorError("MySQL needs a driver: pip install pymysql") from e
    raise ConnectorError(f"no SQL driver for {t.dialect}")


class SqlConnector:
    def __init__(self, url: str, connect: Callable[[], Any] | None = None) -> None:
        self.target = parse_target(url)
        if self.target.dialect == "mongo":
            raise ConnectorError("use MongoConnector for mongodb:// URLs")
        self.dialect = self.target.dialect
        self._connect = connect or (lambda: _driver_connect(self.target))
        self.ph = "?" if self.dialect == "sqlite" else "%s"

    # ---------------------------------------------------------- helpers
    def _exec(self, cur: Any, sql: str, params: Any = None) -> None:
        cur.execute(sql, params) if params is not None else cur.execute(sql)

    def _exists(self, cur: Any, table: str) -> bool:
        if self.dialect == "sqlite":
            self._exec(cur, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
        elif self.dialect == "postgres":
            self._exec(cur, "SELECT 1 FROM information_schema.tables WHERE table_schema = current_schema() AND table_name = %s", (table,))
        else:
            self._exec(cur, "SELECT 1 FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = %s", (table,))
        return cur.fetchone() is not None

    def _count(self, cur: Any, table: str) -> int:
        self._exec(cur, f"SELECT COUNT(*) FROM {quote_ident(table, self.dialect)}")
        return int(cur.fetchone()[0])

    def _insert(self, cur: Any, t: TableSchema, data: TableData, batch: int) -> int:
        q = lambda n: quote_ident(n, self.dialect)  # noqa: E731
        cols = [c.name for c in t.columns]
        kinds = [c.kind for c in t.columns]
        sql = f"INSERT INTO {q(t.name)} ({', '.join(q(c) for c in cols)}) VALUES ({', '.join([self.ph] * len(cols))})"
        n, buf = 0, []
        for ch in chunks(data):
            ch = ch[[c for c in cols if c in ch.columns]]
            for r in ch.itertuples(index=False, name=None):
                buf.append(tuple(py_value(v, k) for v, k in zip(r, kinds)))
                if len(buf) >= batch:
                    cur.executemany(sql, buf)
                    n, buf = n + len(buf), []
        if buf:
            cur.executemany(sql, buf)
            n += len(buf)
        return n

    # -------------------------------------------------------------- load
    def load(self, tables: dict[str, TableData], schema: dict[str, TableSchema], dry_run: bool = True, mode: str = "create", batch: int = 1000) -> LoadResult:
        if mode not in MODES:
            raise ConnectorError(f"mode must be one of {list(MODES)}")
        res = LoadResult(target=mask(self.target.url), dialect=self.dialect, mode=mode, dry_run=dry_run)
        t0 = time.perf_counter()
        order = order_tables(schema, list(tables))
        ddl_transactional = self.dialect in ("postgres", "sqlite")
        plan_only_ddl = dry_run and not ddl_transactional and mode != "append"      # MySQL commits DDL implicitly: never rehearse it for real
        try:
            con = self._connect()
        except ConnectorError:
            raise
        except Exception as e:  # noqa: BLE001 - driver-specific errors; never echo the URL
            res.errors.append(f"could not connect: {type(e).__name__}")
            res.seconds = time.perf_counter() - t0
            return res
        created: list[str] = []
        try:
            cur = con.cursor()
            if self.dialect == "sqlite":
                cur.execute("BEGIN")
            elif self.dialect == "mysql":
                cur.execute("SET FOREIGN_KEY_CHECKS = 0") if plan_only_ddl else None
            # pre-flight
            for n in order:
                exists = self._exists(cur, n)
                if mode == "create" and exists:
                    raise ConnectorError(f"table {n!r} already exists; use mode='replace' to drop it or mode='append' to add rows")
                if mode == "append" and not exists:
                    raise ConnectorError(f"table {n!r} does not exist; use mode='create'")
            if plan_only_ddl:
                res.strategy = "plan_only"
                res.notes.append("MySQL cannot roll back DDL, so a dry run validates the plan and target without creating tables.")
                for n in order:
                    res.tables[n] = sum(len(c) for c in chunks(tables[n]))
                    res.statements.append(create_table_sql(schema[n], self.dialect).strip().split("\n")[0])
                res.ok = True
                res.rolled_back = True
                res.notes.append("nothing was written")
                con.rollback()
                return res
            if mode == "replace":
                for n in reversed(order):
                    cur.execute(f"DROP TABLE IF EXISTS {quote_ident(n, self.dialect)}" + (" CASCADE" if self.dialect == "postgres" else ""))
            before: dict[str, int] = {}
            for n in order:
                if mode == "append":
                    before[n] = self._count(cur, n)
                else:
                    ddl = create_table_sql(schema[n], self.dialect)
                    cur.execute(ddl)
                    created.append(n)
                    if len(res.statements) < 5:
                        res.statements.append(ddl.strip().split("\n")[0])
                res.tables[n] = self._insert(cur, schema[n], tables[n], batch)
            for n in order:
                res.verified[n] = self._count(cur, n) - before.get(n, 0)
                if res.verified[n] != res.tables[n]:
                    raise ConnectorError(f"row count mismatch in {n!r}: wrote {res.tables[n]}, database has {res.verified[n]}")
            if dry_run:
                con.rollback()
                res.rolled_back = True
                if self.dialect == "sqlite":
                    pass
            else:
                con.commit()
                res.committed = True
            res.ok = True
        except Exception as e:  # noqa: BLE001
            msg = str(e) if isinstance(e, ConnectorError) else f"{type(e).__name__}: {str(e)[:300]}"
            res.errors.append(msg)
            try:
                con.rollback()
                res.rolled_back = True
            except Exception:  # noqa: BLE001
                pass
            if created and not ddl_transactional:                   # DDL was auto-committed: undo what this run created
                res.strategy = "compensating"
                try:
                    c2 = con.cursor()
                    for n in reversed(created):
                        c2.execute(f"DROP TABLE IF EXISTS {quote_ident(n, self.dialect)}")
                    con.commit()
                    res.notes.append(f"dropped {len(created)} table(s) created by this run")
                except Exception:  # noqa: BLE001
                    res.errors.append("cleanup failed; drop the tables created by this run manually: " + ", ".join(created))
        finally:
            try:
                if self.dialect == "mysql" and not plan_only_ddl:
                    pass
                con.close()
            except Exception:  # noqa: BLE001
                pass
            res.seconds = round(time.perf_counter() - t0, 3)
        return res
