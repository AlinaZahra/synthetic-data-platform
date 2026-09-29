"""Generation config: the single object that (with the fitted model) fully determines output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

OutlierMethod = Literal["iqr", "zscore", "scale"]


class GenConfig(BaseModel):
    """Same `GenConfig` + same fitted model => byte-identical DataFrame.

    null_rate / outlier_rate map column name -> fraction of rows in [0, 1].
    outlier_method:
      iqr     value pushed 1.5-3 IQRs beyond the Tukey fences
      zscore  value placed 4-6 standard deviations from the mean
      scale   value multiplied by a factor in [5, 10]
    seed=None draws a random seed, which is recorded in the result so it can be replayed.
    """

    model_config = ConfigDict(extra="forbid")

    rows: int = Field(1000, ge=1, le=5_000_000)
    seed: int | None = 0
    null_rate: dict[str, float] = Field(default_factory=dict)
    outlier_rate: dict[str, float] = Field(default_factory=dict)
    outlier_method: OutlierMethod = "iqr"
    # scenario packs (sdp.edgecases): pack name -> fraction of rows; affected rows are tagged in a hidden `_edge_case` column
    edge_cases: dict[str, float] = Field(default_factory=dict)

    @field_validator("edge_cases")
    @classmethod
    def _edge_packs_valid(cls, v: dict[str, float]) -> dict[str, float]:
        from sdp.edgecases import ALL_PACKS
        for name, r in v.items():
            if name not in ALL_PACKS:
                raise ValueError(f"unknown edge-case pack {name!r}; available: {ALL_PACKS}")
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"rate for {name!r} must be within [0, 1], got {r}")
        return v

    @field_validator("null_rate", "outlier_rate")
    @classmethod
    def _rates_in_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for col, r in v.items():
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"rate for {col!r} must be within [0, 1], got {r}")
        return v
