import json
from pathlib import Path

from perturb_benchmark.pipeline import run_cpu_pipeline


def test_offline_pipeline_produces_valid_manifest(tmp_path: Path) -> None:
    manifest = run_cpu_pipeline(
        config_path=Path("tests/fixtures/synthetic_config.yaml"),
        project_root=tmp_path,
        offline=True,
    )
    assert manifest.n_test_conditions > 0
    assert list(manifest.models) == [
        "no_change",
        "perturbed_mean",
        "matching_mean",
        "additive_sum",
        "calibrated_additive",
        "ridge",
    ]
    assert all(item.sha256 and item.bytes > 0 for item in manifest.artifacts)
    saved = json.loads((tmp_path / "results/metadata/run_manifest.json").read_text())
    assert saved["n_test_conditions"] == manifest.n_test_conditions


def test_offline_pipeline_is_scientifically_reproducible(tmp_path: Path) -> None:
    first = run_cpu_pipeline(
        Path("tests/fixtures/synthetic_config.yaml"), tmp_path / "one", offline=True
    )
    second = run_cpu_pipeline(
        Path("tests/fixtures/synthetic_config.yaml"), tmp_path / "two", offline=True
    )
    first_hashes = {item.path: item.sha256 for item in first.artifacts}
    second_hashes = {item.path: item.sha256 for item in second.artifacts}
    assert first_hashes == second_hashes
