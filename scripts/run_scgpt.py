"""Fail-closed entry point for a pinned real scGPT perturbation run."""

import argparse
from pathlib import Path

import yaml

from perturb_benchmark.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/scgpt.yaml"))
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = config["checkpoint"]
    required = ("path", "checkpoint_sha256", "vocabulary_path", "vocabulary_sha256")
    if not all(checkpoint.get(key) for key in required):
        raise SystemExit("Refusing scGPT run: real checkpoint and vocabulary hashes are missing.")
    for path_key, hash_key in (
        ("path", "checkpoint_sha256"),
        ("vocabulary_path", "vocabulary_sha256"),
    ):
        path = Path(checkpoint[path_key])
        if not path.is_file() or sha256_file(path) != checkpoint[hash_key]:
            raise SystemExit(f"Refusing scGPT run: {path_key} is absent or has the wrong SHA256.")
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Refusing scGPT run: PyTorch is unavailable.") from error
    if config["execution"]["requires_cuda"] and not torch.cuda.is_available():
        raise SystemExit("Refusing scGPT run: CUDA is required but unavailable.")
    raise SystemExit(
        "Environment gate passed, but full fold training must be executed in the pinned GPU "
        "workflow documented in docs/SCGPT_SETUP.md. No prediction was fabricated."
    )


if __name__ == "__main__":
    main()
