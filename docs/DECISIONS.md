# Decision record

1. Treat Norman perturbations as CRISPR activation/overexpression, never knockout.
2. Use five-fold seen-components/unseen-combination validation as the primary estimand.
3. Use processed `adata.X` as the primary target because it matches the GEARS/scGPT perturbation
   workflow; reserve the integer-like `counts` layer for sensitivity analysis.
4. Use a global control because the provided artifact has no batch identifier.
5. Include strong transparent and linear baselines before interpreting a foundation model.
6. Preserve undefined no-change correlations as `N/A`.
7. Describe interaction-stratified gains as coupled diagnostics, not mechanism evidence.
8. Separate the CPU environment from a candidate pinned GPU environment.
9. Do not label embeddings as full scGPT and do not synthesize missing scGPT predictions.
10. Keep GitHub-facing descriptions and scientific reports in English.
