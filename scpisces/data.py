from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_vocabulary(path: str | Path) -> dict[str, int]:
    vocabulary = json.loads(Path(path).read_text())
    if vocabulary.get("<pad>") != 0:
        raise ValueError("The vocabulary must map <pad> to ID 0")
    if "<unk>" not in vocabulary:
        raise ValueError("The vocabulary must contain <unk>")
    if vocabulary["<unk>"] == vocabulary["<pad>"]:
        raise ValueError("<unk> and <pad> must use different IDs")
    values = list(vocabulary.values())
    if min(values) < 0 or len(values) != len(set(values)):
        raise ValueError("Vocabulary IDs must be unique non-negative integers")
    return {str(key): int(value) for key, value in vocabulary.items()}


def read_pathway_names(
    adata: ad.AnnData, names_key: str, num_pathways: int
) -> list[str]:
    candidate_keys = [
        names_key,
        "Reactome_pathway_cols",
        "pathways_mtx_cols",
        "pathway_names",
    ]
    for key in candidate_keys:
        if key in adata.uns:
            names = [str(item) for item in list(adata.uns[key])]
            if len(names) != num_pathways:
                raise ValueError(
                    f"adata.uns[{key!r}] has {len(names)} names for "
                    f"{num_pathways} pathway columns"
                )
            return names
    return [f"Pathway_{index}" for index in range(num_pathways)]


def pathway_matrix(adata: ad.AnnData, key: str) -> np.ndarray | None:
    candidate_keys = [key, "Reactome_pathway", "pathways_mtx"]
    for candidate in candidate_keys:
        if candidate in adata.obsm:
            values = adata.obsm[candidate]
            if sparse.issparse(values):
                values = values.toarray()
            return np.asarray(values, dtype=np.float32)
    return None


def fit_pathway_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-6))


class AnnDataCellDataset(Dataset):
    """Tokenize non-zero genes from an AnnData expression matrix."""

    def __init__(
        self,
        adata: ad.AnnData,
        vocabulary: dict[str, int],
        max_genes: int,
        class_mapping: dict[str, int] | None = None,
        label_key: str | None = None,
        pathway_key: str = "Reactome_pathway",
        pathway_mean: np.ndarray | None = None,
        pathway_std: np.ndarray | None = None,
        total_counts_key: str = "total_counts",
        gene_selection: Literal["random", "top_expression", "first"] = "random",
        seed: int = 2026,
        training: bool = False,
        require_pathways: bool = False,
    ) -> None:
        if max_genes < 1:
            raise ValueError("max_genes must be positive")
        self.adata = adata
        self.vocabulary = vocabulary
        self.max_genes = max_genes
        self.class_mapping = class_mapping
        self.label_key = label_key
        self.total_counts_key = total_counts_key
        self.gene_selection = gene_selection
        self.seed = seed
        self.training = training
        self.epoch = 0
        self.pad_token_id = vocabulary["<pad>"]
        self.unk_token_id = vocabulary["<unk>"]
        self.gene_ids = np.asarray(
            [vocabulary.get(str(gene), self.unk_token_id) for gene in adata.var_names],
            dtype=np.int64,
        )

        self.pathways = pathway_matrix(adata, pathway_key)
        if require_pathways and self.pathways is None:
            raise ValueError(
                f"Training data must contain pathway scores in adata.obsm[{pathway_key!r}]"
            )
        if self.pathways is not None:
            if pathway_mean is not None and pathway_std is not None:
                self.pathways = (self.pathways - pathway_mean) / pathway_std
            self.pathways = np.nan_to_num(self.pathways).astype(np.float32)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.adata.n_obs

    def _row(self, index: int) -> tuple[np.ndarray, np.ndarray, float]:
        row = self.adata.X[index]
        if sparse.issparse(row):
            row = row.tocsr()
            positions = row.indices
            values = row.data
            matrix_total = float(values.sum())
        else:
            dense = np.asarray(row).reshape(-1)
            positions = np.flatnonzero(dense > 0)
            values = dense[positions]
            matrix_total = float(dense.sum())
        if self.total_counts_key in self.adata.obs:
            total = float(self.adata.obs.iloc[index][self.total_counts_key])
        else:
            total = matrix_total
        return positions, np.asarray(values, dtype=np.float32), total

    def _select(self, values: np.ndarray, index: int) -> np.ndarray:
        count = len(values)
        if count <= self.max_genes:
            return np.arange(count)
        if self.gene_selection == "top_expression":
            selected = np.argpartition(values, -self.max_genes)[-self.max_genes :]
            return selected[np.argsort(values[selected])[::-1]]
        if self.gene_selection == "first":
            return np.arange(self.max_genes)
        epoch_offset = self.epoch if self.training else 0
        rng = np.random.default_rng(self.seed + index + epoch_offset * len(self))
        return rng.choice(count, self.max_genes, replace=False)

    def __getitem__(self, index: int) -> dict:
        positions, values, total_counts = self._row(index)
        selected = self._select(values, index)
        positions = positions[selected]
        values = values[selected]

        label = -1
        label_name = None
        if self.label_key is not None and self.label_key in self.adata.obs:
            label_name = str(self.adata.obs.iloc[index][self.label_key])
            if self.class_mapping is not None:
                label = self.class_mapping.get(label_name, -1)

        pathway_target = (
            self.pathways[index]
            if self.pathways is not None
            else np.empty(0, dtype=np.float32)
        )
        return {
            "gene_ids": torch.from_numpy(self.gene_ids[positions]),
            "expression_values": torch.from_numpy(values),
            "total_counts": torch.tensor(total_counts, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "pathway_target": torch.from_numpy(pathway_target),
            "cell_id": str(self.adata.obs_names[index]),
            "label_name": label_name,
        }


def collate_cells(
    batch: list[dict], max_genes: int, pad_token_id: int = 0
) -> dict:
    gene_ids = pad_sequence(
        [item["gene_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    expression_values = pad_sequence(
        [item["expression_values"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    if gene_ids.shape[1] < max_genes:
        extra = max_genes - gene_ids.shape[1]
        gene_ids = F_pad(gene_ids, extra, pad_token_id)
        expression_values = F_pad(expression_values, extra, 0.0)
    elif gene_ids.shape[1] > max_genes:
        gene_ids = gene_ids[:, :max_genes]
        expression_values = expression_values[:, :max_genes]

    return {
        "gene_ids": gene_ids,
        "expression_values": expression_values,
        "padding_mask": gene_ids.eq(pad_token_id),
        "total_counts": torch.stack([item["total_counts"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "pathway_targets": torch.stack(
            [item["pathway_target"] for item in batch]
        ),
        "cell_ids": [item["cell_id"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
    }


def F_pad(tensor: torch.Tensor, extra: int, value: int | float) -> torch.Tensor:
    padding = torch.full(
        (tensor.shape[0], extra), value, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=1)
