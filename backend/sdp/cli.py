"""P2. Command line interface mirroring the REST endpoints, for CI and scripts.

Local (default, no server): jobs are stored under --data-dir / $SDP_DATA_DIR.
Remote: --url http://host:8000 --api-key KEY  (or $SDP_URL / $SDP_API_KEY).

  python -m sdp.cli generate --kind tabular --params '{"rows": 500, "seed": 1}' --wait --score --min-score 80 --out out/
  python -m sdp.cli jobs [--kind tabular]         python -m sdp.cli jobs get ID
  python -m sdp.cli score ID [--min-score 85] [--refresh]
  python -m sdp.cli rerun ID [--override '{"seed": 2}'] [--wait]     (no override = reproducibility check)
  python -m sdp.cli versions ID     python -m sdp.cli manifest ID     python -m sdp.cli download ID NAME --out PATH

Exit codes: 0 ok · 1 error · 2 quality gate failed (job failed, score below --min-score, or rerun not reproduced).
Output is JSON on stdout so it can be piped into jq.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_ERROR, EXIT_GATE = 0, 1, 2


class CliError(Exception):
    pass


# ------------------------------------------------------------------ clients
class LocalClient:
    def __init__(self, data_dir: str | None) -> None:
        from sdp import service
        from sdp.lineage import Store
        self.service, self.store = service, Store(Path(data_dir) if data_dir else None)

    def _wrap(self, fn, *a, **k):
        from sdp.lineage import JobNotFound
        try:
            return fn(*a, **k)
        except JobNotFound as e:
            raise CliError(f"not found: {e}") from e
        except ValueError as e:
            raise CliError(str(e)) from e
        except Exception as e:  # pydantic validation errors
            if e.__class__.__name__ == "ValidationError":
                raise CliError("; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from e  # type: ignore[attr-defined]
            raise

    def generate(self, kind, params, wait):
        from sdp import lineage
        m = self._wrap(self.service.submit, self.store, kind, params, None, True)
        return {**lineage.summary(m), "id": m["id"]}

    def job(self, jid):
        return self._wrap(self.store.read, jid)

    def jobs(self, kind):
        from sdp import lineage
        return [lineage.summary(m) for m in self.store.list(kind)]

    def versions(self, jid):
        from sdp import lineage
        return [lineage.summary(m) for m in self._wrap(self.store.versions, jid)]

    def rerun(self, jid, overrides, wait):
        from sdp import lineage
        m = self._wrap(self.service.rerun, self.store, jid, True, overrides or None)
        return lineage.summary(m)

    def score(self, jid, refresh):
        return {"job_id": jid, **self._wrap(self.service.score, self.store, jid, refresh)}

    def file(self, jid, name):
        return self._wrap(self.store.artifact, jid, name)


class RemoteClient:
    def __init__(self, url: str, key: str | None) -> None:
        if not key:
            raise CliError("remote mode needs --api-key (or $SDP_API_KEY)")
        self.url, self.key = url.rstrip("/"), key

    def _req(self, method: str, path: str, body: Any = None, raw: bool = False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"X-API-Key": self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                payload = r.read()
                return payload if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise CliError(f"HTTP {e.code}: {detail[:400]}") from e
        except urllib.error.URLError as e:
            raise CliError(f"cannot reach {self.url}: {e.reason}") from e

    def _poll(self, jid: str) -> dict[str, Any]:
        for _ in range(900):
            j = self._req("GET", f"/jobs/{jid}")
            if j["status"] in ("succeeded", "failed"):
                return j
            time.sleep(1)
        raise CliError("timed out waiting for the job")

    def generate(self, kind, params, wait):
        r = self._req("POST", "/generate", {"kind": kind, "params": params, "wait": wait})
        return r if wait or r["status"] in ("succeeded", "failed") else self._poll(r["id"])

    def job(self, jid):
        return self._req("GET", f"/jobs/{jid}/manifest")

    def jobs(self, kind):
        return self._req("GET", "/jobs" + (f"?kind={kind}" if kind else ""))

    def versions(self, jid):
        return self._req("GET", f"/jobs/{jid}/versions")

    def rerun(self, jid, overrides, wait):
        r = self._req("POST", f"/jobs/{jid}/rerun", {"wait": wait, "overrides": overrides or {}})
        return r if wait else self._poll(r["id"])

    def score(self, jid, refresh):
        return self._req("GET", f"/score/{jid}" + ("?refresh=true" if refresh else ""))

    def file(self, jid, name):
        return self._req("GET", f"/jobs/{jid}/files/{name}", raw=True)


# ---------------------------------------------------------------------- main
def _json_arg(text: str | None, path: str | None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return json.loads(text) if text else {}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sdp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SDP_URL"), help="remote API base URL (default: run locally)")
    ap.add_argument("--api-key", default=os.environ.get("SDP_API_KEY"))
    ap.add_argument("--data-dir", default=None, help="local job store (default $SDP_DATA_DIR or ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="start a job (mirrors POST /generate)")
    g.add_argument("--kind", required=True, choices=["tabular", "relational", "nl", "document"])
    g.add_argument("--params", help="JSON object")
    g.add_argument("--params-file")
    g.add_argument("--wait", action="store_true", default=True, help="(default) wait for completion")
    g.add_argument("--score", action="store_true", help="also compute the score")
    g.add_argument("--min-score", type=float, help="exit 2 if the score is below this (implies --score)")
    g.add_argument("--out", help="copy outputs and manifest into this directory")
    j = sub.add_parser("jobs", help="list jobs, or `jobs get ID`")
    j.add_argument("action", nargs="?", default="list", choices=["list", "get"])
    j.add_argument("id", nargs="?")
    j.add_argument("--kind")
    s = sub.add_parser("score", help="mirrors GET /score/{id}")
    s.add_argument("id")
    s.add_argument("--min-score", type=float)
    s.add_argument("--refresh", action="store_true")
    r = sub.add_parser("rerun", help="rerun from a manifest; exit 2 if the output differs")
    r.add_argument("id")
    r.add_argument("--override", help="JSON parameter overrides (makes it a new version)")
    r.add_argument("--wait", action="store_true", default=True)
    for name in ("versions", "manifest"):
        sub.add_parser(name).add_argument("id")
    d = sub.add_parser("download")
    d.add_argument("id")
    d.add_argument("name")
    d.add_argument("--out", required=True)
    return ap


def run(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        client = RemoteClient(a.url, a.api_key) if a.url else LocalClient(a.data_dir)
        emit = lambda obj: print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))  # noqa: E731
        if a.cmd == "generate":
            job = client.generate(a.kind, _json_arg(a.params, a.params_file), True)
            out: dict[str, Any] = {"job": job}
            code = EXIT_OK if job["status"] == "succeeded" else EXIT_GATE
            if job["status"] == "succeeded" and (a.score or a.min_score is not None):
                sc = client.score(job["id"], False)
                out["score"] = {k: sc[k] for k in ("score", "label", "kind") if k in sc}
                if a.min_score is not None and sc["score"] < a.min_score:
                    out["gate"] = f"score {sc['score']:.1f} is below --min-score {a.min_score}"
                    code = EXIT_GATE
            if a.out and job["status"] == "succeeded":
                dest = Path(a.out)
                dest.mkdir(parents=True, exist_ok=True)
                for name in job["files"]:
                    (dest / name).write_bytes(client.file(job["id"], name))
                (dest / "manifest.json").write_text(json.dumps(client.job(job["id"]), indent=2, default=str, ensure_ascii=False), encoding="utf-8")
                out["written_to"] = str(dest)
            emit(out)
            return code
        if a.cmd == "jobs":
            if a.action == "get":
                if not a.id:
                    raise CliError("jobs get needs an id")
                emit(client.job(a.id))
            else:
                emit(client.jobs(a.kind))
            return EXIT_OK
        if a.cmd == "score":
            sc = client.score(a.id, a.refresh)
            emit({k: sc[k] for k in ("job_id", "score", "label", "kind") if k in sc} | {"report_keys": sorted(sc.get("report", {}))[:12]})
            return EXIT_GATE if a.min_score is not None and sc["score"] < a.min_score else EXIT_OK
        if a.cmd == "rerun":
            job = client.rerun(a.id, json.loads(a.override) if a.override else None, True)
            emit(job)
            return EXIT_GATE if job["status"] != "succeeded" or (not a.override and job.get("reproduced") is False) else EXIT_OK
        if a.cmd == "versions":
            emit(client.versions(a.id))
            return EXIT_OK
        if a.cmd == "manifest":
            emit(client.job(a.id))
            return EXIT_OK
        if a.cmd == "download":
            Path(a.out).write_bytes(client.file(a.id, a.name))
            emit({"written": a.out})
            return EXIT_OK
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
