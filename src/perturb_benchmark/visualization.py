"""Deterministic figures built only from validated saved tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

PALETTE = "colorblind"


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "perturb-benchmark"})
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return [png, svg]


def make_figures(project_root: Path | str) -> list[Path]:
    """Render model, response-gene, and interaction diagnostics."""

    root = Path(project_root)
    summary = pd.read_csv(root / "results/metrics/model_summary.csv")
    response = pd.read_csv(root / "results/tables/response_gene_metrics.csv")
    interaction = pd.read_csv(root / "results/tables/interaction_diagnostics.csv")
    sns.set_theme(style="whitegrid")
    plt.rcParams["svg.hashsalt"] = "perturb-benchmark"
    outputs: list[Path] = []

    metadata_path = root / "data/processed/condition_metadata.csv"
    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(data=metadata, x="n_cells", bins=20, ax=ax)
        ax.set(title="Cells per canonical condition", xlabel="Cell count", ylabel="Conditions")
        outputs.extend(_save(fig, root / "figures/qc_condition_counts"))

    deltas_path = root / "results/predictions/pseudobulk_deltas.csv.gz"
    if deltas_path.is_file():
        deltas = pd.read_csv(deltas_path)
        condition_column = deltas.columns[0]
        non_control = deltas.loc[deltas[condition_column] != "ctrl"].reset_index(drop=True)
        matrix = non_control.drop(columns=condition_column).to_numpy(dtype=float)
        if len(non_control) >= 2 and matrix.shape[1] >= 2:
            coordinates = PCA(n_components=2).fit_transform(matrix)
            pca_frame = pd.DataFrame(
                {
                    "PC1": coordinates[:, 0],
                    "PC2": coordinates[:, 1],
                    "kind": non_control[condition_column].astype(str).str.count(r"\+").map(
                        {0: "single", 1: "double"}
                    ),
                }
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=pca_frame, x="PC1", y="PC2", hue="kind", ax=ax)
            ax.set(title="PCA of pseudo-bulk perturbation responses")
            outputs.extend(_save(fig, root / "figures/response_pca"))

        magnitudes = np.sqrt(np.mean(matrix**2, axis=1))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(magnitudes, bins=20, ax=ax)
        ax.set(title="Perturbation response magnitude", xlabel="Response RMSE", ylabel="Conditions")
        outputs.extend(_save(fig, root / "figures/response_magnitude"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=summary, x="median_pearson", y="model", hue="model", palette=PALETTE, ax=ax)
    ax.set(title="Median condition-level Pearson correlation", xlabel="Pearson r", ylabel="")
    for index, row in summary.reset_index(drop=True).iterrows():
        if pd.isna(row["median_pearson"]):
            ax.text(0.01, index, "N/A (constant prediction)", va="center", fontsize=8)
    outputs.extend(_save(fig, root / "figures/model_correlation"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=summary, x="median_mae", y="model", hue="model", palette=PALETTE, ax=ax)
    ax.set(title="Median condition-level absolute error", xlabel="MAE", ylabel="")
    outputs.extend(_save(fig, root / "figures/model_mae"))

    response_summary = response.groupby("model", as_index=False)["topk_recovery"].median()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=response_summary,
        x="topk_recovery",
        y="model",
        hue="model",
        palette=PALETTE,
        ax=ax,
    )
    ax.set(title="Median top-k response-gene recovery", xlabel="Recovered fraction", ylabel="")
    outputs.extend(_save(fig, root / "figures/response_gene_recovery"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=interaction,
        x="interaction_magnitude",
        y="rmse",
        hue="model",
        palette=PALETTE,
        ax=ax,
    )
    ax.set(
        title="Interaction magnitude and prediction error",
        xlabel="Interaction RMSE",
        ylabel="Model RMSE",
    )
    outputs.extend(_save(fig, root / "figures/interaction_difficulty"))
    return outputs
