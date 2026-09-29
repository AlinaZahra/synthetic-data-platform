"""Visualize endpoints: `GET /visualize/{dataset_id}?type=...` (API key) and the same at `/api/visualize/{dataset_id}` for the UI (no key).

Responses are small summaries (histogram bins, category shares, correlation matrices, orphan counts), never rows. `dataset_id` is a built-in
dataset (customers, students, employees, shop_full, bank_customers, ecommerce_customers, documents-demo) or the id of a finished saved run.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from sdp import visualize as viz
from sdp.api.public import _store, require_api_key

ui = APIRouter(prefix="/api/visualize", tags=["visualize"])
public = APIRouter(dependencies=[Depends(require_api_key)], tags=["public API"])


def _run(dataset_id: str, type: str, column: str | None, table: str | None, fk: str | None, target: str | None, bins: int, limit: int,
         rows: int, seed: int, locale: str) -> dict[str, Any]:
    if type not in viz.TYPES:
        raise HTTPException(422, f"unknown type {type!r}; available: {list(viz.TYPES)}")
    try:
        bundle = viz.resolve(dataset_id, _store(), rows=rows, seed=seed, locale=locale)
        return viz.payload(bundle, type, column=column, table=table, fk=fk, target=target, bins=bins, limit=limit, seed=seed)
    except viz.DatasetNotFound as e:
        raise HTTPException(404, f"unknown dataset {dataset_id!r}; see GET /api/visualize for the list") from e
    except viz.NotReady as e:
        raise HTTPException(409, str(e)) from e
    except (viz.VisualizeError, ValueError) as e:
        raise HTTPException(422, str(e)) from e


_DOC = "One chart's data as small JSON. `type`: " + ", ".join(viz.TYPES)


@ui.get("")
def catalog() -> dict[str, Any]:
    """Datasets and saved runs that can be visualized, with the chart types each supports."""
    return viz.catalog(_store())


@ui.get("/{dataset_id}", summary=_DOC)
def visualize_ui(dataset_id: str, type: str = Query(...), column: str | None = None, table: str | None = None, fk: str | None = None, target: str | None = None,
                 bins: int = Query(20, ge=5, le=viz.MAX_BINS), limit: int = Query(12, ge=3, le=viz.MAX_CATS), rows: int = Query(1000, ge=100, le=viz.MAX_ROWS),
                 seed: int = 0, locale: str = "ur-PK") -> dict[str, Any]:
    return _run(dataset_id, type, column, table, fk, target, bins, limit, rows, seed, locale)


@public.get("/visualize/{dataset_id}", summary=_DOC)
def visualize_public(dataset_id: str, type: str = Query(...), column: str | None = None, table: str | None = None, fk: str | None = None, target: str | None = None,
                     bins: int = Query(20, ge=5, le=viz.MAX_BINS), limit: int = Query(12, ge=3, le=viz.MAX_CATS), rows: int = Query(1000, ge=100, le=viz.MAX_ROWS),
                     seed: int = 0, locale: str = "ur-PK") -> dict[str, Any]:
    return _run(dataset_id, type, column, table, fk, target, bins, limit, rows, seed, locale)
