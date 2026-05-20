# Data dictionary

## Cell-level input

| Field | Meaning |
|---|---|
| `adata.X` | Expression matrix; its scale is inspected before transformation |
| `adata.obs[condition]` | Original perturbation condition label |
| configured batch column | Optional experimental batch, replicate, library, or gem group |
| `adata.var_names` | Unique gene identifiers |

## Canonical perturbation metadata

| Field | Meaning |
|---|---|
| `condition_original` | Sorted unique source labels joined by `|` |
| `condition_canonical` | `ctrl`, `A`, or alphabetically ordered `A+B` |
| `perturbation_type` | `control`, `single`, or `double` |
| `gene_a`, `gene_b` | Canonical component genes |
| `n_cells` | Number of source cells |
| `n_batches` | Number of represented experimental batches |
