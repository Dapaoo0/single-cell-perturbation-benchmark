from pathlib import Path

from perturb_benchmark.config import load_config


def test_load_config_freezes_primary_design() -> None:
    cfg = load_config(Path("config/project.yaml"))

    assert cfg.dataset.geo_accession == "GSE133344"
    assert cfg.split.name == "seen-components-unseen-combination"
    assert cfg.split.n_folds == 5
    assert cfg.bootstrap.iterations == 2000
    assert cfg.random_seed == 42
    assert cfg.paths.source_archive == Path("data/raw/norman.zip")
