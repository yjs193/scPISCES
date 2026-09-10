from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    counts: np.ndarray,
    mode: str,
    beta: float = 0.9992,
    maximum: float = 20.0,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    if mode == "effective":
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    elif mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class-weighting mode: {mode}")
    weights /= weights.mean()
    weights = np.clip(weights, 0.05, maximum)
    return torch.tensor(weights, dtype=torch.float32)


class ScPISCESLoss(nn.Module):
    """Joint classification, reconstruction, pathway, and consistency loss."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
        lambda_reconstruction: float = 1.0,
        lambda_pathway: float = 1.0,
        lambda_consistency: float = 1.5,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_pathway = lambda_pathway
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        pathway_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dense_classification = F.cross_entropy(
            output["dense_logits"],
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        sparse_classification = F.cross_entropy(
            output["sparse_logits"],
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        if output["guided_raw"].shape != pathway_targets.shape:
            raise ValueError(
                "Pathway target shape does not match the guided branch: "
                f"{tuple(pathway_targets.shape)} versus "
                f"{tuple(output['guided_raw'].shape)}"
            )
        pathway_alignment = 1.0 - F.cosine_similarity(
            output["guided_raw"], pathway_targets, dim=1, eps=1e-8
        ).mean()

        dense_probability = output["dense_logits"].softmax(dim=-1)
        sparse_probability = output["sparse_logits"].softmax(dim=-1)
        dense_log_probability = output["dense_logits"].log_softmax(dim=-1)
        sparse_log_probability = output["sparse_logits"].log_softmax(dim=-1)
        consistency = F.kl_div(
            sparse_log_probability, dense_probability, reduction="batchmean"
        ) + F.kl_div(
            dense_log_probability, sparse_probability, reduction="batchmean"
        )

        classification = dense_classification + sparse_classification
        total = (
            classification
            + self.lambda_reconstruction * output["reconstruction_loss"]
            + self.lambda_pathway * pathway_alignment
            + self.lambda_consistency * consistency
        )
        return {
            "total": total,
            "classification": classification,
            "dense_classification": dense_classification,
            "sparse_classification": sparse_classification,
            "reconstruction": output["reconstruction_loss"],
            "pathway": pathway_alignment,
            "consistency": consistency,
            "l0": output["l0"],
        }
