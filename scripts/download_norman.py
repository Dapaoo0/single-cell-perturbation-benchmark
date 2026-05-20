"""Download the configured processed Norman artifact with provenance."""

import argparse
import json
from pathlib import Path

from perturb_benchmark.config import load_config
from perturb_benchmark.provenance import (
    download_with_provenance,
    extract_zip_member,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config.dataset.processed_data_url is None:
        raise SystemExit("processed_data_url is not frozen; verify the GEARS artifact first")
    provenance_path = config.paths.metadata_dir / "source_provenance.json"
    record = download_with_provenance(
        config.dataset.processed_data_url,
        config.paths.source_archive,
        provenance_path,
    )
    member = extract_zip_member(
        config.paths.source_archive,
        "perturb_processed.h5ad",
        config.paths.raw_anndata,
    )
    record["archive_member"] = member
    record["extracted_path"] = config.paths.raw_anndata.as_posix()
    record["extracted_bytes"] = config.paths.raw_anndata.stat().st_size
    record["extracted_sha256"] = sha256_file(config.paths.raw_anndata)
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
