# Beyond Memorizing Single Genes

### Can a model predict a genetic combination it has never seen?

This repository turns that question into a hard, reproducible benchmark: predict held-out
**double-gene CRISPR activation responses** in the Norman Perturb-seq dataset while both component
singles remain visible during training.

The benchmark is deliberately difficult to fool. Test combinations never enter fitting, feature
selection, calibration, or hyperparameter tuning. Every model receives the same genes, targets,
folds, and evaluation contract.

## At a glance

- **91,205 cells · 5,045 genes · 131 double perturbations**
- **Five frozen unseen-combination folds**
- **Six serious CPU baselines**, from no-change and additivity to nested-tuned Ridge
- **Paired uncertainty**, response-gene recovery, and interaction-aware failure analysis
- **Checksummed artifacts** and a complete offline end-to-end test

> **Current result status:** the CPU benchmark is complete. Full scGPT training has not been run
> because this machine has no CUDA runtime and no verified checkpoint was supplied. The repository
> reports that boundary plainly—no synthetic predictions, no inflated claim.

## Run the entire benchmark offline

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run python scripts/run_cpu_pipeline.py --config tests/fixtures/synthetic_config.yaml --project-root .test-run --offline
```

Start with the [methods](docs/METHODS.md), inspect the [scientific limitations](docs/LIMITATIONS.md),
or follow the [real-data reproduction guide](docs/REPRODUCIBILITY.md).

This is a benchmark of CRISPR activation/overexpression—not knockout, not unseen-gene prediction,
and not a claim of causal mechanism discovery.
