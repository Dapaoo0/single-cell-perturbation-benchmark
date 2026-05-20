"""Render deterministic benchmark figures."""

import argparse
from pathlib import Path

from perturb_benchmark.visualization import make_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    for path in make_figures(args.project_root):
        print(path)


if __name__ == "__main__":
    main()
