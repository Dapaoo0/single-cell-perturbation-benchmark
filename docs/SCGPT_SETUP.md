# scGPT execution gate

The CPU benchmark does not imply that scGPT was run. Full scGPT perturbation results require a
separate CUDA host, the pinned source revisions in `config/scgpt.yaml`, and a real pretrained
checkpoint plus vocabulary whose SHA256 values are computed locally.

## Current state

This workstation has no detected NVIDIA runtime. The environment check therefore records
`blocked_full_training`; the repository must say “scGPT was not run” and must not make a model
comparison claim.

## Reproduction procedure

1. Create the candidate environment with `environment-scgpt.yml` on a CUDA 12.1-compatible host.
2. Run a minimal import and Norman data-loader probe. Export the final package inventory after the
   probe; the candidate environment file is not evidence that the probe succeeded.
3. Download the intended pretrained checkpoint from an official scGPT release, compute SHA256 for
   the checkpoint and vocabulary, and enter their local paths and hashes in `config/scgpt.yaml`.
4. Run `scripts/check_scgpt_environment.py`. Continue only if it reports `ready_for_probe`.
5. Fine-tune separately for each frozen fold using only its training cells. Sample 32 control cells
   with seed 42 for inference, save long-format predictions, and create one `ScGPTManifest` per fold.
6. Validate predictions against the exact frozen CPU keys with `validate_scgpt_predictions` before
   including them in any report.

Embedding-only experiments must use the model label `scgpt_embedding`. They are not interchangeable
with the full perturbation model.
