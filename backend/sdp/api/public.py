"""P2. Public REST API (API key required) + UI history endpoints (open, same-origin).

  POST /generate            start a job (202, or 200 with wait=true)
  GET  /jobs/{id}           status, request, outputs, lineage
  GET  /score/{id}          Trust Score / validation score for a finished job
  GET  /jobs, /jobs/{id}/manifest, /jobs/{id}/versions, POST /jobs/{id}/rerun, GET /jobs/{id}/files/{name}
OpenAPI docs: /docs and /openapi.json (use "Authorize" with your key).

Keys: SDP_API_KEYS="key1,key2". If unset, one ephemeral key is generated at startup and logged, so the API is never open by accident.
Send it as `X-API-Key: <key>` or `Authorization: Bearer <key>`. The UI's /api/* endpoints are unauthenticated: keep the app behind
your own network boundary or proxy in production.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, Security
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from sdp import lineage, service
from sdp.lineage import JobNotFound, Store

log = logging.getLogger("sdp.api")
_EPHEMERAL = secrets.token_urlsafe(24)
if not os.environ.get("SDP_API_KEYS"):
    log.warning("SDP_API_KEYS is not set; the public API accepts only this ephemeral key: %s", _EPHEMERAL)

_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key (or use Authorization: Bearer)")
_bearer = HTTPBearer(auto_error=False)


def valid_keys() -> list[str]:
    raw = os.environ.get("SDP_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys or [_EPHEMERAL]


def require_api_key(key: str | None = Security(_header), bearer: HTTPAuthorizationCredentials | None = Security(_bearer)) -> str:
    supplied = key or (bearer.credentials if bearer else None)
    if supplied and any(secrets.compare_digest(supplied, k) for k in valid_keys()):
        return supplied
    raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


class GenerateRequest(BaseModel):
    kind: str = Field(description="tabular | relational | nl | document")
    params: dict[str, Any] = Field(default_factory=dict, description="kind-specific parameters (see the *Params models in sdp/service.py)")
    wait: bool = Field(False, description="run inline and return the finished job")

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"kind": "tabular", "params": {"rows": 500, "seed": 1, "rules": ["age must be at least 25"]}, "wait": True},
        {"kind": "relational", "params": {"dataset": "shop_full", "seed": 1, "rules": ["orders.discount <= 0.3"]}},
        {"kind": "nl", "params": {"text": "5,000 Pakistani bank customers, 3% fraud, 12 months of history"}},
        {"kind": "document", "params": {"doc_type": "statement", "count": 3, "spec": {"locale": "ur-PK", "native": True}}}]})


class RerunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wait: bool = False
    overrides: dict[str, Any] = Field(default_factory=dict, description="change parameters (then it is a new version, not a reproducibility check)")


def _store() -> Store:
    return Store()


def _not_found(e: Exception) -> HTTPException:
    return HTTPException(404, f"not found: {e}")


def _links(job_id: str, prefix: str = "") -> dict[str, str]:
    return {"self": f"{prefix}/jobs/{job_id}", "score": f"{prefix}/score/{job_id}", "manifest": f"{prefix}/jobs/{job_id}/manifest"}


def _start(body: GenerateRequest, response: Response, prefix: str = "") -> dict[str, Any]:
    try:
        m = service.submit(_store(), body.kind, body.params, wait=body.wait)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # pydantic ValidationError -> readable 422
        if e.__class__.__name__ == "ValidationError":
            raise HTTPException(422, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from e  # type: ignore[attr-defined]
        raise
    response.status_code = 200 if body.wait else 202
    return {**lineage.summary(m), "links": _links(m["id"], prefix)}


# ------------------------------------------------------------- public API
public = APIRouter(dependencies=[Depends(require_api_key)], tags=["public API"])


@public.post("/generate", summary="Start a generation job", status_code=202)
def generate(body: GenerateRequest, response: Response) -> dict[str, Any]:
    return _start(body, response)


@public.get("/jobs", summary="List jobs (newest first)")
def jobs(kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return [lineage.summary(m) for m in _store().list(kind, min(max(limit, 1), 500))]


@public.get("/jobs/{job_id}", summary="Job status, request, outputs and lineage")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        m = _store().read(job_id)
    except JobNotFound as e:
        raise _not_found(e) from e
    return {**lineage.summary(m), "request": m["request"], "outputs": m["outputs"], "lineage": m["lineage"], "links": _links(job_id)}


@public.get("/jobs/{job_id}/progress", summary="Progress of a running job: done/total, percent, phase, failures, ETA")
def get_progress(job_id: str) -> dict[str, Any]:
    try:
        return service.progress(_store(), job_id)
    except JobNotFound as e:
        raise _not_found(e) from e


@public.post("/jobs/{job_id}/cancel", summary="Cancel a queued or running job (it keeps what it already finished)")
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return lineage.summary(service.cancel(_store(), job_id))
    except JobNotFound as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@public.get("/jobs/{job_id}/download.zip", summary="ZIP of all outputs and the manifest (finished or cancelled jobs)")
def download_zip(job_id: str) -> FileResponse:
    import os
    try:
        path = service.zip_job(_store(), job_id)
    except JobNotFound as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return FileResponse(path, media_type="application/zip", filename=f"job-{job_id}.zip", background=BackgroundTask(os.unlink, path))


@public.get("/jobs/{job_id}/manifest", summary="Full manifest: schema, seed, config, model version, outputs, scores")
def manifest(job_id: str) -> dict[str, Any]:
    try:
        return _store().read(job_id)
    except JobNotFound as e:
        raise _not_found(e) from e


@public.get("/jobs/{job_id}/versions", summary="All versions in this job's lineage")
def versions(job_id: str) -> list[dict[str, Any]]:
    try:
        return [lineage.summary(m) for m in _store().versions(job_id)]
    except JobNotFound as e:
        raise _not_found(e) from e


@public.post("/jobs/{job_id}/rerun", summary="Rerun from the manifest (reproducibility check, or a new version with overrides)", status_code=202)
def rerun(job_id: str, body: RerunRequest, response: Response) -> dict[str, Any]:
    try:
        m = service.rerun(_store(), job_id, wait=body.wait, overrides=body.overrides or None)
    except JobNotFound as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    response.status_code = 200 if body.wait else 202
    return {**lineage.summary(m), "links": _links(m["id"])}


@public.get("/jobs/{job_id}/files/{name:path}", summary="Download one output file (e.g. report.json, docs/<id>.pdf)")
def files(job_id: str, name: str) -> Response:
    try:
        path = _store().artifact_path(job_id, name)
    except JobNotFound as e:
        raise _not_found(e) from e
    media = ("text/csv; charset=utf-8" if name.endswith(".csv") else "application/pdf" if name.endswith(".pdf") else "application/sql" if name.endswith(".sql")
             else "application/gzip" if name.endswith(".gz") else "application/x-ndjson" if name.endswith(".jsonl") else "application/json")
    return FileResponse(path, media_type=media, filename=name.split("/")[-1])          # streamed from disk: large outputs are never read into memory


@public.get("/score/{job_id}", summary="Trust Score (or validation score) for a finished job")
def get_score(job_id: str, refresh: bool = False) -> dict[str, Any]:
    try:
        return {"job_id": job_id, **service.score(_store(), job_id, refresh)}
    except JobNotFound as e:
        raise _not_found(e) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


# ------------------------------------------------- UI history (no key needed)
history = APIRouter(prefix="/api/history", tags=["ui history"])


@history.get("")
def h_list(limit: int = 100) -> list[dict[str, Any]]:
    return [lineage.summary(m) for m in _store().list(None, min(max(limit, 1), 500))]


@history.post("/generate")
def h_generate(body: GenerateRequest, response: Response) -> dict[str, Any]:
    return _start(body, response, "/api/history")


@history.get("/{job_id}")
def h_get(job_id: str) -> dict[str, Any]:
    try:
        return _store().read(job_id)
    except JobNotFound as e:
        raise _not_found(e) from e


@history.get("/{job_id}/versions")
def h_versions(job_id: str) -> list[dict[str, Any]]:
    return versions(job_id)


@history.post("/{job_id}/rerun")
def h_rerun(job_id: str, body: RerunRequest, response: Response) -> dict[str, Any]:
    return rerun(job_id, body, response)


@history.get("/{job_id}/progress")
def h_progress(job_id: str) -> dict[str, Any]:
    return get_progress(job_id)


@history.post("/{job_id}/cancel")
def h_cancel(job_id: str) -> dict[str, Any]:
    return cancel_job(job_id)


@history.get("/{job_id}/download.zip")
def h_zip(job_id: str) -> FileResponse:
    return download_zip(job_id)


@history.get("/{job_id}/score")
def h_score(job_id: str, refresh: bool = False) -> dict[str, Any]:
    return get_score(job_id, refresh)


@history.get("/{job_id}/files/{name}")
def h_file(job_id: str, name: str) -> Response:
    return files(job_id, name)
