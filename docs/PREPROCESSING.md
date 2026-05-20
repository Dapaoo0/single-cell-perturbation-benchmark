# Preprocessing decision

## Evidence

The pinned GEARS AnnData contains 91,205 A549 cells and 5,045 genes. `adata.X`
is a sparse nonnegative continuous matrix with values up to approximately 7.18.
It is already processed and must not be normalized again. The dense `counts`
layer is nonnegative and integer-like, but the retained 5,045-gene panel does
not contain the full original library; recomputing cell-level library-size
normalization from that subset would not reproduce `adata.X`.

The processed object retains `condition`, `cell_type`, `dose_val`, `control`,
and `condition_name`, but no gem group, batch, replicate, sample, or library
field. Batch-matched controls therefore cannot be reconstructed from this
artifact.

## Frozen primary representation

The primary benchmark uses condition-level means of the existing `adata.X`
values and subtracts the global control mean on the same scale:

```text
delta_condition = mean_X(condition) - mean_X(ctrl)
```

This choice matches the expression representation consumed by the official
GEARS/scGPT perturbation workflow and keeps every model on the same target.
Additive and interaction residuals are explicitly interpreted as quantities on
this processed expression scale, not as raw-molecule additivity.

The `counts` layer is retained for a secondary sensitivity analysis after the
core paired benchmark. It is not used to manufacture a different target for
the CPU baselines while scGPT is evaluated on `adata.X`.

## Condition handling

Source labels such as `A+ctrl` and `ctrl+A` are merged into canonical single
conditions. Double labels are alphabetically ordered, so `A+B` and `B+A` are
the same experimental condition. All 237 canonical conditions pass the frozen
minimum of 30 cells; no condition is removed by that threshold.
