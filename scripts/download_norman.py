"""Download the configured processed Norman artifact with provenance."""

import argparse
from pathlib import Path

from perturb_benchmark.config import load_config
from perturb_benchmark.provenance import download_with_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.dataset.processed_data_url is None:
        raise SystemExit("processed_data_url is not frozen; verify the GEARS artifact first")
    download_with_provenance(
        config.dataset.processed_data_url,
        config.paths.raw_anndata,
        config.paths.metadata_dir / "source_provenance.json",
    )


if __name__ == "__main__":
    main()

