import sqlite3

import numpy as np
import pandas as pd
import pytest

from sdp.connectors import ConnectorError, MongoConnector, SqlConnector, get_connector, mask, parse_target
from sdp.exporters import schema_from
from sdp.relational import infer_graph


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SDP_CONNECTOR_ALLOW_HOSTS", raising=False)
    return tmp_path


def data():
    cust = pd.DataFrame({"customer_id": range(1, 21), "name": [f"C{i} O'Neil" for i in range(20)], "score": np.linspace(0, 1, 20)})
    orders = pd.DataFrame({"order_id": range(1, 41), "customer_id": [1 + i % 20 for i in range(40)], "total": np.linspace(5, 50, 40),
                           "placed": pd.date_range("2024-01-01", periods=40)})
    t = {"customers": cust, "orders": orders}
    return t, schema_from(t, infer_graph(t))


def tables_in(path):
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_sqlite_dry_run_writes_verifies_and_leaves_nothing(env):
    t, s = data()
    url = "sqlite:///demo.db"
    r = SqlConnector(url).load(t, s, dry_run=True)
    assert r.ok and r.rolled_back and not r.committed and r.dry_run
    assert r.tables == {"customers": 20, "orders": 40} == r.verified
    assert not (env / "demo.db").exists() or tables_in(env / "demo.db") == set()


def test_sqlite_real_load_commits_with_keys_and_data_intact(env):
    t, s = data()
    r = get_connector("sqlite:///demo.db").load(t, s, dry_run=False)
    assert r.ok and r.committed and not r.rolled_back
    con = sqlite3.connect(env / "demo.db")
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 40
    assert con.execute("SELECT name FROM customers WHERE customer_id = 1").fetchone()[0] == "C0 O'Neil"
    assert [r[2] for r in con.execute('PRAGMA foreign_key_list("orders")')] == ["customers"]
    con.close()
    # create again -> refused, nothing changed
    r2 = SqlConnector("sqlite:///demo.db").load(t, s, dry_run=False)
    assert not r2.ok and "already exists" in r2.errors[0]
    # append adds rows; a dry-run append rolls back
    t2 = {"customers": t["customers"].assign(customer_id=lambda d: d.customer_id + 100)}
    s2 = {"customers": s["customers"]}
    assert SqlConnector("sqlite:///demo.db").load(t2, s2, dry_run=True, mode="append").verified == {"customers": 20}
    con = sqlite3.connect(env / "demo.db")
    assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 20
    con.close()
    assert SqlConnector("sqlite:///demo.db").load(t2, s2, dry_run=False, mode="append").ok
    con = sqlite3.connect(env / "demo.db")
    assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 40
    con.close()
    # replace drops and reloads
    assert SqlConnector("sqlite:///demo.db").load(t, s, dry_run=False, mode="replace").ok
    con = sqlite3.connect(env / "demo.db")
    assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 20
    con.close()


def test_failure_midway_rolls_everything_back(env):
    t, s = data()
    bad = {"customers": t["customers"], "orders": t["orders"].assign(customer_id=lambda d: d.customer_id.where(d.index != 39, 9999))}   # orphan FK on the last row
    r = SqlConnector("sqlite:///demo.db").load(bad, s, dry_run=False)
    assert not r.ok and not r.committed and r.rolled_back and r.errors
    assert not (env / "demo.db").exists() or tables_in(env / "demo.db") == set()
    dup = {"customers": pd.concat([t["customers"], t["customers"].head(1)])}
    r = SqlConnector("sqlite:///demo.db").load(dup, {"customers": s["customers"]}, dry_run=False)
    assert not r.ok and "UNIQUE" in r.errors[0].upper() or "constraint" in r.errors[0].lower()
    assert not (env / "demo.db").exists() or tables_in(env / "demo.db") == set()


def test_targets_are_restricted_and_secrets_masked(env, monkeypatch):
    with pytest.raises(ConnectorError, match="not allowed"):
        parse_target("postgresql://u:pw@db.example.com/x")
    with pytest.raises(ConnectorError, match="inside the data directory"):
        parse_target("sqlite:////etc/passwd")
    with pytest.raises(ConnectorError, match="inside the data directory"):
        parse_target("sqlite:///../../escape.db")
    with pytest.raises(ConnectorError, match="unsupported"):
        parse_target("oracle://x/y")
    monkeypatch.setenv("SDP_CONNECTOR_ALLOW_HOSTS", "db.example.com")
    assert parse_target("postgresql://u:pw@db.example.com/x").host == "db.example.com"
    m = mask("postgresql://admin:s3cret@localhost:5432/prod")
    assert "s3cret" not in m and m == "postgresql://admin:***@localhost:5432/prod"
    r = SqlConnector("sqlite:///:memory:").load(*data(), dry_run=True)
    assert r.ok and "s3cret" not in str(r.to_dict())


# ----------------------------------------------------------- fakes for server databases
class FakeCursor:
    def __init__(self, con):
        self.con, self.rows = con, []

    def execute(self, sql, params=None):
        self.con.log.append(("execute", sql, params))
        if self.con.fail_on and self.con.fail_on in sql:
            raise RuntimeError("boom")
        s = sql.strip().upper()
        if "INFORMATION_SCHEMA" in s:
            self.rows = [(1,)] if params and params[0] in self.con.existing else []
        elif s.startswith("SELECT COUNT"):
            table = sql.split("FROM")[1].strip().strip('`"')
            self.rows = [(self.con.counts.get(table, 0),)]
        elif s.startswith("CREATE TABLE"):
            self.con.ddl_committed += 1 if self.con.dialect == "mysql" else 0

    def executemany(self, sql, rows):
        self.con.log.append(("executemany", sql, len(rows)))
        if self.con.fail_on and self.con.fail_on in sql:
            raise RuntimeError("boom")
        table = sql.split("INTO")[1].split("(")[0].strip().strip('`"')
        self.con.counts[table] = self.con.counts.get(table, 0) + len(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, dialect, existing=(), fail_on=None):
        self.dialect, self.existing, self.fail_on = dialect, set(existing), fail_on
        self.log, self.counts, self.ddl_committed = [], {}, 0
        self.commits = self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_postgres_load_uses_percent_placeholders_one_transaction_and_rolls_back_dry_run(env):
    t, s = data()
    fc = FakeConn("postgres")
    r = SqlConnector("postgresql://u:pw@localhost/db", connect=lambda: fc).load(t, s, dry_run=True)
    assert r.ok and r.rolled_back and r.strategy == "transaction" and fc.commits == 0 and fc.rollbacks >= 1 and fc.closed
    ins = [x for x in fc.log if x[0] == "executemany"]
    assert ins and "%s" in ins[0][1] and '"customers"' in ins[0][1]
    fc2 = FakeConn("postgres")
    r = SqlConnector("postgresql://u:pw@localhost/db", connect=lambda: fc2).load(t, s, dry_run=False)
    assert r.ok and r.committed and fc2.commits == 1


def test_postgres_error_rolls_back_and_reports(env):
    t, s = data()
    fc = FakeConn("postgres", fail_on='INSERT INTO "orders"')
    r = SqlConnector("postgresql://u:pw@localhost/db", connect=lambda: fc).load(t, s, dry_run=False)
    assert not r.ok and r.rolled_back and fc.commits == 0 and "boom" in r.errors[0] and "pw" not in str(r.to_dict())


def test_existing_table_blocks_create_before_any_write(env):
    t, s = data()
    fc = FakeConn("postgres", existing={"orders"})
    r = SqlConnector("postgresql://u:pw@localhost/db", connect=lambda: fc).load(t, s, dry_run=False)
    assert not r.ok and "already exists" in r.errors[0] and not any(x[0] == "executemany" for x in fc.log)


def test_mysql_dry_run_is_plan_only_and_real_failure_compensates(env):
    t, s = data()
    fc = FakeConn("mysql")
    r = SqlConnector("mysql://u:pw@127.0.0.1/db", connect=lambda: fc).load(t, s, dry_run=True)
    assert r.ok and r.strategy == "plan_only" and fc.ddl_committed == 0 and not any("CREATE TABLE" in str(x[1]) for x in fc.log) and r.tables == {"customers": 20, "orders": 40}
    fc = FakeConn("mysql", fail_on="INSERT INTO `orders`")
    r = SqlConnector("mysql://u:pw@127.0.0.1/db", connect=lambda: fc).load(t, s, dry_run=False)
    assert not r.ok and r.strategy == "compensating" and fc.ddl_committed == 2
    drops = [x[1] for x in fc.log if x[0] == "execute" and "DROP TABLE" in x[1]]
    assert any("`orders`" in d for d in drops) and any("`customers`" in d for d in drops)


class FakeCollection:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def insert_many(self, docs, session=None):
        if self.db.fail_on == self.name:
            raise RuntimeError("write failed")
        ids = list(range(len(self.db.docs.setdefault(self.name, [])) + 1, len(self.db.docs.get(self.name, [])) + len(docs) + 1))
        self.db.docs[self.name] += [dict(d, _id=i) for d, i in zip(docs, ids)]
        return type("R", (), {"inserted_ids": ids})()

    def delete_many(self, flt, session=None):
        ids = set(flt.get("_id", {}).get("$in", [])) if flt else None
        self.db.docs[self.name] = [d for d in self.db.docs[self.name] if ids is not None and d["_id"] not in ids]


class FakeDb:
    def __init__(self, fail_on=None, txn=True):
        self.docs, self.fail_on = {}, fail_on

    def list_collection_names(self):
        return list(self.docs)

    def __getitem__(self, n):
        return FakeCollection(self, n)


class FakeMongo:
    def __init__(self, txn=True, fail_on=None):
        self.db, self.txn, self.aborted, self.committed = FakeDb(fail_on), txn, 0, 0

    def __getitem__(self, n):
        return self.db

    def start_session(self):
        if not self.txn:
            raise RuntimeError("Transaction numbers are only allowed on a replica set member or mongos")
        outer = self

        class S:
            def start_transaction(self):
                self.snap = {k: list(v) for k, v in outer.db.docs.items()}

            def abort_transaction(self):
                outer.db.docs = self.snap
                outer.aborted += 1

            def commit_transaction(self):
                outer.committed += 1
        return S()

    def close(self):
        pass


def test_mongo_with_transactions_dry_run_aborts_and_real_run_commits(env):
    t, s = data()
    m = FakeMongo()
    r = MongoConnector("mongodb://localhost/synth", client_factory=lambda: m).load(t, s, dry_run=True)
    assert r.ok and r.rolled_back and r.strategy == "transaction" and m.aborted == 1 and m.db.docs == {"customers": [], "orders": []} or not any(m.db.docs.values())
    m = FakeMongo()
    r = MongoConnector("mongodb://localhost/synth", client_factory=lambda: m).load(t, s, dry_run=False)
    assert r.ok and r.committed and m.committed == 1 and len(m.db.docs["orders"]) == 40 and m.db.docs["customers"][0]["name"] == "C0 O'Neil"


def test_mongo_without_transactions_plans_and_compensates(env):
    t, s = data()
    m = FakeMongo(txn=False)
    r = MongoConnector("mongodb://localhost/synth", client_factory=lambda: m).load(t, s, dry_run=True)
    assert r.ok and r.strategy == "plan_only" and m.db.docs == {} and r.tables == {"customers": 20, "orders": 40}
    m = FakeMongo(txn=False, fail_on="orders")
    r = MongoConnector("mongodb://localhost/synth", client_factory=lambda: m).load(t, s, dry_run=False)
    assert not r.ok and r.strategy == "compensating" and m.db.docs["customers"] == []           # customers inserted, then removed again


def test_mongo_needs_a_database_name_and_missing_drivers_are_explained(env):
    with pytest.raises(ConnectorError, match="database name"):
        MongoConnector("mongodb://localhost:27017")
    with pytest.raises(ConnectorError, match="mongodb://"):
        MongoConnector("sqlite:///:memory:")
    with pytest.raises(ConnectorError, match="MongoDB\\b|pymongo|pip install"):
        try:
            import pymongo  # noqa: F401
            pytest.skip("pymongo installed")
        except ImportError:
            MongoConnector("mongodb://localhost/x").load(*data())
            raise ConnectorError("pip install pymongo")   # load() reports connect failure in the result instead of raising
