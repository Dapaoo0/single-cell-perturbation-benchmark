# Results

## Current completion state

The best CPU baseline by median condition-level Pearson correlation was ridge (0.935). scGPT was not run, so no comparative scGPT claim is made.

scGPT feasibility status: `blocked_full_training`.

## CPU baseline summary

| model | median Pearson | median MAE |
|---|---:|---:|
| additive_sum | 0.922 | 0.012 |
| calibrated_additive | 0.922 | 0.012 |
| matching_mean | 0.922 | 0.013 |
| no_change | N/A | 0.022 |
| perturbed_mean | 0.616 | 0.018 |
| ridge | 0.935 | 0.011 |

Each test row is a held-out double-gene CRISPRa combination whose component single-gene
perturbations remain available to training. Correlation for the no-change baseline is `N/A` by
definition because its prediction vector is constant; it is not replaced by zero.

## Interpretation boundary

Interaction magnitude is reported as a descriptive difficulty variable. Because the interaction
residual is also the additive model's error vector, associations between interaction strength and
gain over additivity are mathematically coupled and are not evidence of mechanistic understanding.
