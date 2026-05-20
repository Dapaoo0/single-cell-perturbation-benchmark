# Limitations

- The primary source is a processed convenience representation, not raw sequencing counts.
- The artifact contains one cell line and no batch/replicate label, limiting generalization and
  preventing batch-matched primary inference.
- A held-out combination is not an unseen-gene or zero-shot task because both component singles are
  observed.
- Pretraining overlap with Norman data is not known; this benchmark cannot establish data novelty.
- Pseudo-bulk evaluation does not measure cell-state distributional fidelity.
- Interaction magnitude is algebraically tied to additive-model error and is descriptive only.
- Full scGPT training remains pending a compatible GPU environment and a verified official
  checkpoint. CPU-only results cannot answer whether scGPT adds predictive value.
- Multiple conditions can share perturbation genes, so condition bootstrap captures condition-level
  uncertainty but not every possible dependency structure.
