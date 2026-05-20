"""Record whether this host can truthfully run full scGPT training."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import yaml


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/scgpt.yaml"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch_version = package_version("torch")
    cuda_available = False
    cuda_version = None
    if torch_version is not None:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
    checkpoint = config["checkpoint"]
    checkpoint_complete = all(
        checkpoint.get(key)
        for key in ("path", "checkpoint_sha256", "vocabulary_path", "vocabulary_sha256")
    )
    ready = cuda_available and checkpoint_complete and package_version("scgpt") is not None
    report = {
        "status": "ready_for_probe" if ready else "blocked_full_training",
        "reason": None
        if ready
        else "CUDA, the pinned scGPT environment, and real checkpoint hashes are required.",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "scgpt": package_version("scgpt"),
        "gears": package_version("gears"),
        "checkpoint_complete": checkpoint_complete,
        "sources": config["sources"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
