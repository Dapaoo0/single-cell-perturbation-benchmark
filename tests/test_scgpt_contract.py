import json
from pathlib import Path

import pandas as pd
import pytest

from perturb_benchmark.scgpt_contract import (
    ScGPTManifest,
    validate_scgpt_manifest,
    validate_scgpt_predictions,
)


def valid_manifest() -> ScGPTManifest:
    return ScGPTManifest(
        scgpt_commit="c" * 40,
        gears_commit="d" * 40,
        checkpoint_sha256="a" * 64,
        vocabulary_sha256="b" * 64,
        execution_device="cuda",
        prediction_mode="full-perturbation",
        control_sampling_seed=42,
        control_cells=32,
        fold=0,
        gene_overlap=100,
        package_versions={"torch": "2.1.2"},
    )


def test_manifest_requires_real_checkpoint_checksum() -> None:
    with pytest.raises(ValueError, match="64-character SHA256"):
        ScGPTManifest(
            scgpt_commit="abc1234",
            gears_commit="def5678",
            checkpoint_sha256="unknown",
            vocabulary_sha256="b" * 64,
            execution_device="cpu",
            prediction_mode="full-perturbation",
            control_sampling_seed=42,
            control_cells=32,
            fold=0,
            gene_overlap=100,
            package_versions={},
        )


def test_validate_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(valid_manifest().model_dump_json(), encoding="utf-8")
    assert validate_scgpt_manifest(path) == valid_manifest()


def test_prediction_gate_rejects_wrong_keys() -> None:
    expected = pd.DataFrame(
        {"condition": ["A+B"], "fold": [0], "replicate": ["global"], "gene": ["G1"]}
    )
    prediction = expected.assign(true_delta=1.0, pred_delta=0.5, model="scgpt")
    changed = prediction.assign(gene="G2")
    with pytest.raises(ValueError, match="frozen CPU target keys"):
        validate_scgpt_predictions(changed, expected)


def test_embedding_mode_cannot_be_labeled_scgpt() -> None:
    expected = pd.DataFrame(
        {"condition": ["A+B"], "fold": [0], "replicate": ["global"], "gene": ["G1"]}
    )
    prediction = expected.assign(true_delta=1.0, pred_delta=0.5, model="scgpt")
    with pytest.raises(ValueError, match="scgpt_embedding"):
        validate_scgpt_predictions(prediction, expected, prediction_mode="embedding")


def test_manifest_json_contains_no_unknown_hashes() -> None:
    payload = json.loads(valid_manifest().model_dump_json())
    assert payload["checkpoint_sha256"] != "unknown"
