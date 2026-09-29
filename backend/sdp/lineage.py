"""P4. Reproducibility, versioning and lineage.

Every job writes a manifest next to its outputs:
  request (the exact parameters), seed, schema, dataset fingerprint, model {engine, package version, code fingerprint,
  library versions}, outputs (name, sha256, size) + an overall output hash, scores, timing, and lineage (root_id, parent_id,
  version number). "Rerun from manifest" replays the stored request; because generation is seeded, an unchanged code
  fingerprint must reproduce the same output hash, and the new manifest records whether it did.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import platform
import secrets
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import sdp

_LOCK = threading.RLock()


def data_dir() -> Path:
    return Path(os.environ.get("SDP_DATA_DIR", "data")).resolve()


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash of every source and data file that influences generation (py, json). Changes when the code or a locale pack changes."""
    root = Path(sdp.__file__).parent
    h = hashlib.sha256()
    for p in sorted(list(root.rglob("*.py")) + list(root.rglob("*.json"))):
        if "__pycache__" in p.parts or "fonts" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def model_info(engine: str = "gaussian_copula") -> dict[str, Any]:
    import numpy
    import pandas
    import scipy
    return {"engine": engine, "package_version": sdp.__version__, "code_fingerprint": code_fingerprint(),
            "python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobNotFound(KeyError):
    pass


class Store:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else data_dir()
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        if not job_id.replace("-", "").isalnum():
            raise JobNotFound(job_id)
        return self.root / "jobs" / job_id

    def new_id(self) -> str:
        return f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{secrets.token_hex(3)}"

    def create(self, kind: str, request: dict[str, Any], parent_id: str | None = None) -> dict[str, Any]:
        jid = self.new_id()
        parent = self.read(parent_id) if parent_id else None
        root_id = parent["lineage"]["root_id"] if parent else jid
        version = 1 + sum(1 for m in self.list() if m["lineage"]["root_id"] == root_id)
        m = {"id": jid, "kind": kind, "status": "queued", "created_at": now(), "started_at": None, "finished_at": None,
             "request": request, "request_hash": sha256_bytes(canonical(request).encode())[:16],
             "seed": request.get("seed"), "schema": None, "dataset": None, "model": model_info(request.get("method", "gaussian_copula")),
             "outputs": {"files": [], "output_hash": None}, "scores": None, "error": None,
             "progress": {"done": 0, "total": None, "phase": "queued", "failed": 0}, "cancel_requested": False,
             "lineage": {"root_id": root_id, "parent_id": parent_id, "version": version, "reproduced": None, "code_changed": None}}
        self.path(jid).mkdir(parents=True)
        self.write(m)
        return m

    def write(self, m: dict[str, Any]) -> None:
        p = self.path(m["id"]) / "manifest.json"
        tmp = p.with_suffix(".tmp")
        with _LOCK:
            tmp.write_text(json.dumps(m, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
            for attempt in range(50):  # Windows refuses to replace a file another thread is reading
                try:
                    os.replace(tmp, p)
                    return
                except PermissionError:
                    if attempt == 49:
                        raise
                    time.sleep(0.02)

    def read(self, job_id: str) -> dict[str, Any]:
        p = self.path(job_id) / "manifest.json"
        if not p.exists():
            raise JobNotFound(job_id)
        for attempt in range(50):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (PermissionError, json.JSONDecodeError):  # caught mid-replace
                if attempt == 49:
                    raise
                time.sleep(0.02)
        raise JobNotFound(job_id)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with _LOCK:  # read-modify-write must be atomic, or a cancel and a progress write lose each other's update
            m = self.read(job_id)
            for k, v in fields.items():
                if isinstance(v, dict) and isinstance(m.get(k), dict) and k == "lineage":
                    m[k].update(v)
                else:
                    m[k] = v
            self.write(m)
            return m

    def list(self, kind: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        out = []
        for d in sorted((self.root / "jobs").iterdir(), reverse=True):
            try:
                m = self.read(d.name)
            except (JobNotFound, json.JSONDecodeError):
                continue
            if kind is None or m["kind"] == kind:
                out.append(m)
        return out[:limit] if limit else out

    def versions(self, job_id: str) -> list[dict[str, Any]]:
        root = self.read(job_id)["lineage"]["root_id"]
        return sorted((m for m in self.list() if m["lineage"]["root_id"] == root), key=lambda m: m["lineage"]["version"])

    def save_artifact(self, job_id: str, name: str, payload: bytes) -> dict[str, Any]:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError("bad artifact name")
        (self.path(job_id) / name).write_bytes(payload)
        return {"name": name, "sha256": sha256_bytes(payload), "bytes": len(payload)}

    def artifact_path(self, job_id: str, name: str) -> Path:
        """Validated path to an output file, for streaming downloads of files too large to read into memory."""
        if not re.fullmatch(r"(docs/)?[A-Za-z0-9_][\w.-]*", name):
            raise JobNotFound(name)
        p = self.path(job_id) / name
        if not p.is_file() or name == "manifest.json":
            raise JobNotFound(name)
        return p

    def artifact(self, job_id: str, name: str) -> bytes:
        if not re.fullmatch(r"(docs/)?[A-Za-z0-9_][\w.-]*", name):  # one level of subfolder (docs/) is allowed, nothing else
            raise JobNotFound(name)
        p = self.path(job_id) / name
        if not p.exists() or name == "manifest.json":
            raise JobNotFound(name)
        return p.read_bytes()


def output_hash(files: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical(sorted((f["name"], f["sha256"]) for f in files)).encode())


def summary(m: dict[str, Any]) -> dict[str, Any]:
    """Row for lists/history: no bulky request payloads."""
    sc = (m.get("scores") or {})
    return {"id": m["id"], "kind": m["kind"], "status": m["status"], "created_at": m["created_at"], "finished_at": m["finished_at"],
            "seed": m["seed"], "score": sc.get("score"), "label": sc.get("label"), "version": m["lineage"]["version"],
            "root_id": m["lineage"]["root_id"], "parent_id": m["lineage"]["parent_id"], "reproduced": m["lineage"]["reproduced"],
            "code_changed": m["lineage"]["code_changed"], "output_hash": m["outputs"]["output_hash"],
            "files": [f["name"] for f in m["outputs"]["files"]], "error": m["error"], "progress": m.get("progress"),
            "cancel_requested": m.get("cancel_requested", False)}
