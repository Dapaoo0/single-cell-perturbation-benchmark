# Reproducibility

## CPU environment

Install Python 3.11 and uv, then run:

```bash
uv sync --locked --extra dev
uv lock --check
uv run ruff check .
uv run pytest -q
```

## Offline end-to-end check

```bash
uv run python scripts/run_cpu_pipeline.py --config tests/fixtures/synthetic_config.yaml --project-root .test-run --offline
uv run python scripts/make_figures.py --project-root .test-run
uv run python scripts/build_reports.py --project-root .test-run
```

The manifest lists every scientific output, byte size, and SHA256. Two clean offline runs must have
identical artifact hashes.

## Real Norman data

```bash
uv run python scripts/download_norman.py --config config/project.yaml
uv run python scripts/validate_anndata.py --config config/project.yaml
uv run python scripts/run_cpu_pipeline.py --config config/project.yaml --project-root .
uv run python scripts/make_figures.py --project-root .
uv run python scripts/build_reports.py --project-root .
```

Raw data and full predictions are ignored by Git. Source URLs, commit IDs, sizes, checksums, schema
summaries, frozen splits, compact metric tables, and figures are retained.

## scGPT

See `docs/SCGPT_SETUP.md`. Absence of a real validated prediction artifact must remain visible as
“scGPT was not run.”
