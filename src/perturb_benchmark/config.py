"""Validated project configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """Base class for immutable validated settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetConfig(FrozenModel):
    name: str
    geo_accession: str
    bioproject: str
    provenance_url: str
    processed_data_url: str | None
    condition_column: str
    control_token: str
    batch_column: str | None


class QualityConfig(FrozenModel):
    min_cells_per_condition: int = Field(ge=1)


class GeneSetConfig(FrozenModel):
    strategy: str


class SplitConfig(FrozenModel):
    name: str
    n_folds: int = Field(ge=2)


class BootstrapConfig(FrozenModel):
    iterations: int = Field(ge=1)


class PathsConfig(FrozenModel):
    raw_anndata: Path
    metadata_dir: Path
    processed_dir: Path
    results_dir: Path
    figures_dir: Path


class ProjectConfig(FrozenModel):
    name: str
    random_seed: int
    dataset: DatasetConfig
    quality: QualityConfig
    gene_set: GeneSetConfig
    split: SplitConfig
    bootstrap: BootstrapConfig
    paths: PathsConfig


def load_config(path: Path) -> ProjectConfig:
    """Load and validate one YAML project configuration."""

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    project = raw.pop("project")
    return ProjectConfig(**project, **raw)

