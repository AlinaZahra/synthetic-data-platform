"""Every explicit generate/export/download is recorded in History (a background job with the same request, seed and code version).

Live previews (settings changing on a page) are NOT recorded: only actions where the user asks for the data. Failures here never break the
user's request. Turn off with SDP_AUTOSAVE=0.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sdp import service
from sdp.lineage import Store

log = logging.getLogger("sdp.autosave")


def enabled() -> bool:
    return os.environ.get("SDP_AUTOSAVE", "1") != "0"


def autosave(kind: str, params: dict[str, Any]) -> str | None:
    """Queue a job for this request; returns its id, or None if recording is off or the request cannot be replayed."""
    if not enabled():
        return None
    try:
        return service.submit(Store(), kind, params, wait=False)["id"]
    except Exception as e:  # noqa: BLE001 - history is a convenience, never a reason to fail the request
        log.warning("autosave skipped (%s): %s", kind, e)
        return None


def tabular_params(req: Any, seed: int | None) -> dict[str, Any]:
    c = req.config
    return {"dataset": req.dataset, "sample": req.sample, "data": req.data, "method": req.method, "rows": c.rows, "seed": seed if seed is not None else 0, "rules": req.rules,
            "rule_mode": req.rule_mode, "null_rate": c.null_rate, "outlier_rate": c.outlier_rate, "outlier_method": c.outlier_method, "edge_cases": c.edge_cases}


def relational_params(req: Any) -> dict[str, Any]:
    from sdp.service import RelationalParams
    raw = req.model_dump(exclude_none=True)
    keep = {k: v for k, v in raw.items() if k in RelationalParams.model_fields}
    if raw.get("dataset") == "demo":
        keep["dataset"] = "shop"
    return keep
