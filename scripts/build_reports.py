"""Build English reports from validated artifacts."""

import argparse
from pathlib import Path

from perturb_benchmark.reporting import build_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_reports(args.project_root))


if __name__ == "__main__":
    main()
