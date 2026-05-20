# Data provenance

## Original experiment

- Study: Norman et al., *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes* (2019)
- Assay: pooled Perturb-seq with CRISPR activation/overexpression, including single and combinatorial perturbations
- GEO: `GSE133344`
- BioProject: `PRJNA551220`
- GEO record: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344>

## Processed representation

The execution pipeline uses the Norman archive referenced by GEARS `PertData.load`
at GEARS commit `f374e43e197b295016d80395d7a54ddb81cc6769`.

- Dataverse access URL: <https://dataverse.harvard.edu/api/access/datafile/6154020>
- Archive name: `norman.zip`
- Archive bytes: `168758985`
- Archive SHA256: `c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877`
- Selected member: `norman/perturb_processed.h5ad`
- Extracted bytes: `2228610012`
- Extracted SHA256: `23ffb0fac6a847ff927cf7509d80d85052bfefbfb97610786a2dafaaefa0b6a0`

The machine-readable record is `results/metadata/source_provenance.json`.
GEARS is a convenience representation and did not generate the original experiment.

## Validated schema

- Cells: `91,205`
- Genes: `5,045`
- Original condition labels: `284`
- Canonical conditions: `237` (`1` control, `105` singles, `131` doubles)
- Minimum cells per canonical condition: `49`
- Cell type: A549
- Expression matrix: sparse, nonnegative continuous
- `counts` layer: dense, nonnegative integer-like counts
- Retained batch/replicate field: none
