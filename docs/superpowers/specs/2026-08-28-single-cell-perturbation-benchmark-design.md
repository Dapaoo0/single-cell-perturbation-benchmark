# Single-Cell Perturbation Benchmark Design

## Purpose

Build a reproducible research repository that tests whether a fine-tuned scGPT perturbation model adds predictive value beyond transparent biological and linear baselines when predicting held-out double-gene CRISPRa responses in the Norman et al. Perturb-seq dataset.

The project is a critical benchmark, not a model demonstration. A negative result is valid if the split is leakage-aware, all models share the same targets, and uncertainty and failure cases are reported honestly.

## Scientific claim boundary

The primary claim concerns unseen **combinations** during task-specific training: both component genes have observed single-gene perturbations in training, while their double perturbation is held out. This is not unseen-gene prediction, zero-shot prediction, causal mechanism discovery, or proof that the Norman dataset was absent from scGPT pretraining.

The experiment is Norman CRISPR activation/overexpression. Documentation must not describe it generically as knockout.

## Delivery stages

### Stage 0: scGPT feasibility gate

Before the full analysis, reproduce a minimal scGPT perturbation run on a tiny Norman subset. Record the scGPT and GEARS commits, checkpoint URL and SHA256, Python, PyTorch, CUDA, hardware, gene vocabulary mapping, input expression scale, and prediction scale.

The current workstation has no detected NVIDIA runtime. Stage 0 therefore separates CPU-verifiable integration from GPU execution. If full perturbation fine-tuning cannot be reproduced in a pinned environment, stop the full-model claim and label any fallback explicitly as an `scGPT embedding baseline`.

### Stage 1: reproducible CPU core

Implement source acquisition, checksum recording, AnnData schema validation, canonical perturbation parsing, QC, batch-aware pseudo-bulk construction, train-only gene selection, leakage-aware folds, baselines, metrics, bootstrap, interaction summaries, figures, and reports. The CPU core must run independently after an scGPT prediction artifact has been saved.

### Stage 2: scGPT execution

Fine-tune one pinned scGPT perturbation workflow per frozen fold. All model selection uses training data only. Inference averages predictions over a documented, reproducibly sampled set of control cells. Save predictions in the common long-format schema so all downstream analyses remain CPU-only.

### Stage 3: scientific reporting

Compare paired predictions on identical held-out combinations and genes. Report uncertainty, undefined metrics, systematic-response sensitivity, responsive-gene recovery, interaction-stratified errors, and representative failures without cherry-picking.

## Data provenance and validation

- Original experiment: Norman et al. 2019, GEO `GSE133344`, BioProject `PRJNA551220`.
- Practical processed-data source: pinned GEARS Norman artifact.
- Record source URL, download date, byte size, SHA256, GEARS version/commit, and a transformation note distinguishing original experiment from convenience representation.
- Fail on empty files, missing control, duplicate gene identifiers, unparseable conditions, missing double perturbations, or inconsistent matrices.
- Export exact observed cell, gene, condition, batch, single, and double counts; never copy approximate counts from documentation.

## Expression representation

Inspect `adata.X`, layers, and raw data before transformation. Detect whether values are counts, normalized, log-transformed, or scaled. Never normalize twice.

Prefer batch/replicate-aware pseudo-bulk from counts when counts are available, followed by a single documented normalization. If only processed log expression is available, preserve that scale and explicitly state that additive and interaction quantities are scale-dependent.

For each batch or experimental replicate `r` and condition `c`:

```text
mean[c,r]  = mean expression across eligible cells
delta[c,r] = mean[c,r] - matched_control_mean[r]
```

Aggregate replicate deltas only after matched-control subtraction. If no usable replicate field exists, use a global control and record that limitation.

## Perturbation notation

Canonical parser outputs control, single, or double and a sorted gene tuple. `A+B`, `B+A`, `A+ctrl`, and `ctrl+A` map deterministically. Unsupported multiplicities or malformed conditions fail closed.

The parser produces a metadata table with original condition, canonical condition, type, component genes, cell count, and available batch counts.

## Gene set

The model gene universe is frozen before test outcomes are examined. It is the intersection of:

1. genes present in the processed dataset;
2. genes in the pinned scGPT vocabulary;
3. genes passing expression/QC requirements determined without test response magnitudes.

If variable-gene selection is needed, fit it within each training fold. Every model must predict the same ordered genes within a fold. Save the list and checksum.

## Primary split

Use a deterministic five-fold edge split over eligible double perturbations:

- both component single perturbations exist and remain in training;
- the canonical double condition is assigned to exactly one test fold;
- all cells and pseudo-bulk summaries for test doubles are absent from training;
- remaining eligible doubles may be used for training;
- swapped gene order cannot cross folds;
- fold assignment is frozen and saved before model results are inspected.

Name this design `seen-components / unseen-combination`. Random-cell evaluation is optional and may only be shown as a leakage demonstration, never as the headline result.

## Models

- **B0 No-change:** predict zero delta.
- **B1 Perturbed mean:** predict the mean training perturbation delta.
- **B2 Matching mean:** predict `(delta_A + delta_B) / 2` from training singles.
- **B3 Additive sum:** predict `delta_A + delta_B` from training singles.
- **B4 Calibrated additive:** learn scalar calibration and intercept using training doubles only.
- **B5 Ridge:** multi-output Ridge from component-gene encoding, with alpha selected using nested train-only validation.
- **M1 scGPT:** pinned perturbation workflow fine-tuned using training cells only.

Additive-family models must assert that both single components come exclusively from training. Ridge and calibrated additive provide the distinction between a fixed biological formula and train-calibrated linear response.

## Common prediction contract

Every model exports:

```text
condition, fold, replicate, gene, true_delta, pred_delta, model
```

The evaluator rejects duplicate keys, missing conditions, different gene order, NaN predictions, shape mismatches, or nonidentical paired test sets.

## Evaluation

### Primary estimand

The headline comparison is the paired difference between scGPT and the strongest prespecified non-neural baseline on median condition-level Pearson correlation of delta expression across the fixed gene set.

This metric is not sufficient alone. Report alongside it:

- condition-level Spearman;
- MAE and RMSE;
- correlation after subtracting the training mean perturbation response;
- train-defined responsive-gene correlation;
- top-20 and top-50 absolute-response overlap;
- direction accuracy for recovered responsive genes.

No-change has undefined Pearson and Spearman because its prediction is constant. Display `N/A`, never zero, and compare it using error metrics.

Report the number and cause of all undefined correlations. Do not silently impute them.

### Uncertainty

Use paired bootstrap over held-out perturbation conditions, with 2,000 replicates and fixed seed. Report effect size and percentile 95% interval. Fold and training-seed summaries are reported separately from the condition bootstrap.

### Systematic-response sensitivity

Evaluate predictions both on the full delta and after subtracting the train-derived mean perturbation response. This tests whether models reproduce condition-specific effects rather than only the common difference between perturbed and control cells.

## Interaction analysis

Define the observed scale-dependent residual:

```text
interaction_AB = delta_AB - (delta_A + delta_B)
```

Because this is exactly the additive model's error vector, correlation between interaction strength and `scGPT minus additive` gain is mathematically coupled and cannot support a claim that scGPT understands genetic interactions.

Use interaction magnitude only as a prespecified difficulty descriptor. Compare absolute errors of every model across frozen interaction-strength strata. Any exploratory gain correlation must be labeled coupled/descriptive. Where feasible, add a cross-fitted analysis in which interaction strength and predictive gain are measured on disjoint gene panels.

## Failure analysis

Rank cases systematically using saved metrics, then show representative high-, middle-, and low-performing combinations. Include response magnitude, cell count, batch coverage, interaction magnitude, per-model errors, responsive-gene recovery, and heatmaps generated from result files.

## Engineering architecture

Use Python 3.11 with `uv` for the CPU environment and a separately pinned scGPT environment. Core logic lives in a package under `src/`; scripts are thin CLI wrappers; notebooks only call package APIs and render saved outputs. Raw and generated large artifacts are ignored by Git, while small metadata, frozen splits, result summaries, figures, and provenance are tracked.

Modules have focused responsibilities: configuration, acquisition/provenance, schema validation, parsing, pseudo-bulk, splits, baselines, scGPT adapter, prediction validation, metrics, response genes, interactions, bootstrap, visualization, and reporting.

## Error handling

Fail closed on missing provenance, unsupported expression scale, absent matched controls when batch matching is required, empty eligible split, train/test overlap, missing component singles, test-fitted gene selection, mismatched model targets, nonfinite predictions, or checksum mismatch.

## Testing and verification

- Unit tests cover parsing, canonical ordering, pseudo-bulk, matched controls, delta scale, eligibility, disjointness, additive formulas, calibration isolation, metrics, undefined correlations, top-K recovery, interaction residuals, and prediction contracts.
- Integration fixtures produce a complete CPU run without network or GPU.
- CI runs lock verification, lint, tests, and the offline fixture pipeline.
- A clean real-data CPU rerun must reproduce checksums and tracked summary tables.
- scGPT execution records separate hardware and environment manifests and is never silently simulated.

## Definition of done

The repository is complete when a reader can determine the exact data and scale, held-out unit, leakage assertions, strongest baseline, scGPT provenance, paired results, uncertainty, responsive-gene recovery, systematic-response sensitivity, interaction-analysis boundary, failures, limitations, and reproduction commands without reading implementation internals.

Full-model completion additionally requires a real pinned scGPT prediction artifact. Without it, the repository may be released only as a validated CPU benchmark framework or clearly labeled embedding-baseline study.
