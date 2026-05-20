# Methods

## Data and target

The benchmark uses the GEARS processed representation of the Norman et al. A549 Perturb-seq
experiment (GSE133344). Conditions are canonicalized as control, single-gene, or sorted double-gene
CRISPRa labels. Conditions below the configured minimum cell count are removed. Expression is
averaged within each condition and the matched control mean is subtracted. The released convenience
artifact has no batch or replicate column, so the primary run uses one explicitly recorded global
control rather than pretending that batches exist.

## Validation split

Eligible double perturbations require a measured single perturbation for both components. Each
eligible double appears in exactly one of five deterministic test folds. All component singles are
training information, while the tested combination itself is excluded from fitting, calibration,
hyperparameter selection, and training-response summaries.

## Models

The six CPU comparators are no change, mean training-double response, mean of the two component
single responses, sum of the component responses, a globally calibrated additive response fitted on
training doubles, and multi-output Ridge. Ridge alpha is selected by condition-level inner
cross-validation. A full scGPT model is admitted only with validated checkpoint and vocabulary
hashes, exact source commits, CUDA evidence, per-fold provenance, and exact prediction-key matching.

## Evaluation

Metrics are calculated per held-out condition across a fixed ordered gene set. Constant-vector
correlations are undefined and retained as missing values. Results include Pearson, Spearman, MAE,
RMSE, train-mean-centered correlation, top-k response-gene recovery, direction accuracy, and paired
condition bootstrap intervals. Interaction residual magnitude is a descriptive difficulty measure;
its mathematical coupling to additive error forbids an independent mechanism claim.
