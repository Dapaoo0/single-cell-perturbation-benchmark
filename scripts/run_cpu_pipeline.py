"""Run the leakage-aware CPU perturbation benchmark."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from perturb_benchmark.pipeline import run_cpu_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest = run_cpu_pipeline(args.config, args.project_root, offline=args.offline)
    print(json.dumps(asdict(manifest), indent=2))


if __name__ == "__main__":
    main()
