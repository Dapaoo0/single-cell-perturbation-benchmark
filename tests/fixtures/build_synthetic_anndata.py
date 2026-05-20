"""Write the deterministic offline AnnData fixture used by the CPU pipeline."""

import argparse
from pathlib import Path

from perturb_benchmark.pipeline import build_synthetic_anndata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_synthetic_anndata().write_h5ad(args.output)


if __name__ == "__main__":
    main()
