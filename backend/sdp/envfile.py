"""Minimal .env loader (no dependency). Real environment variables always win. Looks in the project root and in backend/."""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> list[Path]:
    loaded = []
    here = Path(__file__).resolve()
    for d in (here.parents[2], here.parents[1]):        # project root, backend/
        f = d / ".env"
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)
        loaded.append(f)
    return loaded
