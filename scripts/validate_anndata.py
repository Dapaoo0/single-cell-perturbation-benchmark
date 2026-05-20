"""Validate the configured AnnData and export schema evidence."""

import argparse
import json
from pathlib import Path

import anndata as ad

from perturb_benchmark.config import load_config
from perturb_benchmark.data_validation import inspect_anndata
from perturb_benchmark.perturbation import build_condition_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    adata = ad.read_h5ad(config.paths.raw_anndata)
    summary = inspect_anndata(
        adata,
        condition_column=config.dataset.condition_column,
        control_token=config.dataset.control_token,
    )
    config.paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.metadata_dir / "adata_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    batch = (
        adata.obs[config.dataset.batch_column]
        if config.dataset.batch_column is not None
        else None
    )
    metadata = build_condition_metadata(
        adata.obs[config.dataset.condition_column],
        batch,
        control_token=config.dataset.control_token,
    )
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(config.paths.processed_dir / "perturbation_metadata.csv", index=False)


if __name__ == "__main__":
    main()

