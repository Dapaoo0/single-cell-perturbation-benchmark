"""Hard provenance and prediction gates for real scGPT runs."""

import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from perturb_benchmark.predictions import KEY_COLUMNS, validate_predictions

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


class ScGPTManifest(BaseModel):
    """Minimum evidence required before predictions may be called scGPT."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scgpt_commit: str
    gears_commit: str
    checkpoint_sha256: str
    vocabulary_sha256: str
    execution_device: Literal["cpu", "cuda"]
    prediction_mode: Literal["full-perturbation", "embedding"]
    control_sampling_seed: int
    control_cells: int = Field(ge=1)
    fold: int = Field(ge=0)
    gene_overlap: int = Field(ge=1)
    package_versions: dict[str, str]

    @field_validator("scgpt_commit", "gears_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if not COMMIT_PATTERN.fullmatch(value):
            raise ValueError("source commit must be a 7-to-40-character hexadecimal Git ID")
        return value

    @field_validator("checkpoint_sha256", "vocabulary_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("artifact hash must be a lowercase 64-character SHA256")
        return value


def validate_scgpt_manifest(path: Path | str) -> ScGPTManifest:
    """Load a complete strict manifest from JSON."""

    with Path(path).open(encoding="utf-8") as handle:
        return ScGPTManifest.model_validate(json.load(handle))


def validate_scgpt_predictions(
    frame: pd.DataFrame,
    expected_keys: pd.DataFrame,
    *,
    prediction_mode: Literal["full-perturbation", "embedding"] = "full-perturbation",
) -> None:
    """Require exact frozen CPU keys and an honest model label."""

    validate_predictions(frame)
    missing = set(KEY_COLUMNS).difference(expected_keys.columns)
    if missing:
        raise ValueError(f"expected keys are missing columns: {sorted(missing)}")
    if expected_keys.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("frozen CPU target keys must be unique")
    observed = set(frame[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    expected = set(expected_keys[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("scGPT predictions do not match frozen CPU target keys")
    allowed_label = "scgpt" if prediction_mode == "full-perturbation" else "scgpt_embedding"
    labels = set(frame["model"].astype(str))
    if labels != {allowed_label}:
        raise ValueError(f"{prediction_mode} predictions must use model label {allowed_label!r}")
