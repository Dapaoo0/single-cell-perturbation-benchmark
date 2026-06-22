# Data provenance

## Original experiment

- Study: Norman et al., *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes* (2019)
- Cell line: K562 (human chronic myelogenous leukemia cell line)
- Assay: pooled Perturb-seq with SunTag CRISPR activation (CRISPRa)/overexpression, including single and combinatorial perturbations
- Publication: [Norman et al., Science (2019)](https://doi.org/10.1126/science.aax4438)
- GEO: `GSE133344`
- BioProject: `PRJNA551220`
- GEO record: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344>
- Source sample: [BioSample SAMN12142227 / GEO GSM3906020](https://www.ncbi.nlm.nih.gov/biosample/SAMN12142227), which identifies the source as K562 cells

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
- `cell_type` column: present; annotation values are not recorded in the saved schema summary
- Expression matrix: sparse, nonnegative continuous
- `counts` layer: dense, nonnegative integer-like counts
- Retained batch/replicate field: none

K562 is the biological cell-line identity reported by the original study and source sample record.
The saved `results/metadata/adata_summary.json` lists the processed object's `cell_type` column but
does not enumerate its values. The source-based identification above is therefore separate from
an audit of that processed annotation; this documentation correction does not modify the AnnData,
expression values, or benchmark results.
