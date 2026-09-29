"""MongoDB connector: one collection per table. Uses a multi-document transaction when the server supports it (replica set),
otherwise a plan-only dry run and compensating deletes (by inserted _id) if a real load fails part-way."""

from __future__ import annotations

import time
from typing import Any, Callable

from sdp.connectors.base import ConnectorError, LoadResult, mask, parse_target, py_value
from sdp.exporters.base import TableData, TableSchema, chunks
from sdp.exporters.formats import order_tables

MODES = ("create", "append", "replace")


class MongoConnector:
    def __init__(self, url: str, client_factory: Callable[[], Any] | None = None) -> None:
        self.target = parse_target(url)
        if self.target.dialect != "mongo":
            raise ConnectorError("MongoConnector needs a mongodb:// URL")
        if not self.target.database:
            raise ConnectorError("include the database name in the URL, e.g. mongodb://localhost:27017/synthetic")
        self._factory = client_factory

    def _client(self) -> Any:
        if self._factory:
            return self._factory()
        try:
            import pymongo  # type: ignore
        except ImportError as e:
            raise ConnectorError("MongoDB needs a driver: pip install pymongo") from e
        return pymongo.MongoClient(self.target.url, serverSelectionTimeoutMS=8000)

    def load(self, tables: dict[str, TableData], schema: dict[str, TableSchema], dry_run: bool = True, mode: str = "create", batch: int = 1000) -> LoadResult:
        if mode not in MODES:
            raise ConnectorError(f"mode must be one of {list(MODES)}")
        res = LoadResult(target=mask(self.target.url), dialect="mongo", mode=mode, dry_run=dry_run)
        t0 = time.perf_counter()
        order = order_tables(schema, list(tables))
        try:
            client = self._client()
            db = client[self.target.database]
            existing = set(db.list_collection_names())
        except Exception as e:  # noqa: BLE001
            res.errors.append(f"could not connect: {type(e).__name__}")
            res.seconds = time.perf_counter() - t0
            return res
        inserted: dict[str, list[Any]] = {}
        session = None
        try:
            for n in order:
                if mode == "create" and n in existing:
                    raise ConnectorError(f"collection {n!r} already exists; use mode='replace' or 'append'")
            use_txn = False
            try:
                session = client.start_session()
                session.start_transaction()
                use_txn = True
            except Exception:  # noqa: BLE001 - standalone server: no transactions
                session = None
            if not use_txn and dry_run:
                res.strategy = "plan_only"
                res.notes.append("this MongoDB deployment has no transactions (needs a replica set); dry run validates without writing")
                for n in order:
                    res.tables[n] = sum(len(c) for c in chunks(tables[n]))
                res.ok, res.rolled_back = True, True
                return res
            res.strategy = "transaction" if use_txn else "compensating"
            kw = {"session": session} if use_txn else {}
            for n in order:
                col = db[n]
                if mode == "replace" and n in existing:
                    if use_txn:
                        col.delete_many({}, **kw)
                    else:
                        raise ConnectorError(f"mode='replace' on existing collection {n!r} needs a transaction-capable server; drop it manually first")
                cols = [c.name for c in schema[n].columns]
                kinds = [c.kind for c in schema[n].columns]
                total, buf = 0, []
                inserted[n] = []
                for ch in chunks(tables[n]):
                    ch = ch[[c for c in cols if c in ch.columns]]
                    for r in ch.itertuples(index=False, name=None):
                        buf.append({c: py_value(v, k) for c, v, k in zip(ch.columns, r, kinds)})
                        if len(buf) >= batch:
                            inserted[n] += list(col.insert_many(buf, **kw).inserted_ids)
                            total, buf = total + len(buf), []
                if buf:
                    inserted[n] += list(col.insert_many(buf, **kw).inserted_ids)
                    total += len(buf)
                res.tables[n] = total
                res.verified[n] = len(inserted[n])
                if res.verified[n] != total:
                    raise ConnectorError(f"count mismatch in {n!r}")
            if use_txn:
                if dry_run:
                    session.abort_transaction()
                    res.rolled_back = True
                else:
                    session.commit_transaction()
                    res.committed = True
            else:
                res.committed = True
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.errors.append(str(e) if isinstance(e, ConnectorError) else f"{type(e).__name__}: {str(e)[:300]}")
            if session is not None:
                try:
                    session.abort_transaction()
                    res.rolled_back = True
                except Exception:  # noqa: BLE001
                    pass
            else:
                for n, ids in inserted.items():                       # compensating cleanup
                    if ids:
                        try:
                            db[n].delete_many({"_id": {"$in": ids}})
                        except Exception:  # noqa: BLE001
                            res.errors.append(f"cleanup failed for collection {n!r}")
                res.rolled_back = bool(inserted)
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            res.seconds = round(time.perf_counter() - t0, 3)
        return res
