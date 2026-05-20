from pathlib import Path
from zipfile import ZipFile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturb_benchmark.data_validation import inspect_anndata
from perturb_benchmark.provenance import extract_zip_member, sha256_file


def test_validation_reports_schema_control_and_sparse_fraction() -> None:
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)),
        obs=pd.DataFrame(
            {"condition": ["ctrl", "A+ctrl"], "gemgroup": ["g1", "g1"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.layers["counts"] = sparse.csr_matrix(
        np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    )

    summary = inspect_anndata(adata, "condition", "ctrl")

    assert summary.n_cells == 2
    assert summary.n_genes == 2
    assert summary.has_control is True
    assert summary.matrix_kind == "sparse"
    assert summary.zero_fraction == 0.5
    assert summary.integer_like is True
    assert "gemgroup" in summary.candidate_batch_columns
    assert summary.layer_evidence["counts"]["integer_like"] is True
    assert summary.layer_evidence["counts"]["maximum"] == 4.0


def test_validation_rejects_duplicate_gene_ids() -> None:
    with pytest.warns(UserWarning, match="Variable names are not unique"):
        adata = ad.AnnData(
            X=np.ones((2, 2)),
            obs=pd.DataFrame({"condition": ["ctrl", "A"]}, index=["c1", "c2"]),
            var=pd.DataFrame(index=["G1", "G1"]),
        )

    with pytest.raises(ValueError, match="duplicate gene"):
        inspect_anndata(adata, "condition", "ctrl")


def test_validation_rejects_missing_control() -> None:
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame({"condition": ["A", "B"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    with pytest.raises(ValueError, match="control"):
        inspect_anndata(adata, "condition", "ctrl")


def test_sha256_file_matches_known_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")

    assert sha256_file(source) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_extract_zip_member_selects_unique_h5ad_by_basename(tmp_path: Path) -> None:
    archive = tmp_path / "norman.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("norman/perturb_processed.h5ad", b"validated")
        bundle.writestr("norman/metadata.txt", b"metadata")
    destination = tmp_path / "data" / "perturb_processed.h5ad"

    extracted = extract_zip_member(archive, "perturb_processed.h5ad", destination)

    assert extracted == "norman/perturb_processed.h5ad"
    assert destination.read_bytes() == b"validated"
