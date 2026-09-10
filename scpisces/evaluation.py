from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score,
)


def move_batch(batch: dict, device: torch.device) -> dict:
    tensor_keys = (
        "gene_ids",
        "expression_values",
        "padding_mask",
        "total_counts",
        "labels",
        "pathway_targets",
    )
    return {
        **batch,
        **{
            key: batch[key].to(device, non_blocking=True)
            for key in tensor_keys
            if key in batch
        },
    }


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "ari": float(adjusted_rand_score(labels, predictions)),
        "ami": float(adjusted_mutual_info_score(labels, predictions)),
    }


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    head: str = "dense",
    amp: bool = True,
) -> tuple[dict, pd.DataFrame]:
    if head not in {"dense", "sparse"}:
        raise ValueError("head must be 'dense' or 'sparse'")
    model.eval()
    predicted_ids: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    cell_ids: list[str] = []
    label_names: list[str | None] = []

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(device, amp):
            output = model(
                batch["gene_ids"],
                batch["expression_values"],
                batch["total_counts"],
                batch["padding_mask"],
            )
            logits = output[f"{head}_logits"]
            probability = logits.softmax(dim=-1)
        confidence, prediction = probability.max(dim=-1)
        predicted_ids.append(prediction.cpu().numpy())
        confidences.append(confidence.float().cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        cell_ids.extend(raw_batch["cell_ids"])
        label_names.extend(raw_batch["label_names"])

    prediction_ids = np.concatenate(predicted_ids)
    confidence_values = np.concatenate(confidences)
    label_ids = np.concatenate(labels)
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "predicted_cell_type": [class_names[index] for index in prediction_ids],
            "confidence": confidence_values,
        }
    )
    valid = label_ids >= 0
    metrics: dict = {}
    if valid.any():
        frame["true_cell_type"] = label_names
        metrics = classification_metrics(label_ids[valid], prediction_ids[valid])
        metrics["n_evaluated_cells"] = int(valid.sum())
    return metrics, frame
