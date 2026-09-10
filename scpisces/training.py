from __future__ import annotations

import math

import numpy as np
import torch

from .evaluation import autocast_context, move_batch


def cosine_learning_rate_factor(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    minimum_factor: float,
) -> float:
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return minimum_factor + (1.0 - minimum_factor) * cosine


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    objective: torch.nn.Module,
    device: torch.device,
    scaler,
    gradient_clip_norm: float,
    amp: bool,
) -> dict[str, float]:
    model.train()
    model.sae.neuron_fire_count.zero_()
    totals: dict[str, list[float]] = {}
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if (batch["labels"] < 0).any():
            raise ValueError("Training data contain labels absent from class_mapping")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            output = model(
                batch["gene_ids"],
                batch["expression_values"],
                batch["total_counts"],
                batch["padding_mask"],
            )
            losses = objective(output, batch["labels"], batch["pathway_targets"])
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        model.sae.normalize_decoder_()
        for name, value in losses.items():
            totals.setdefault(name, []).append(float(value.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in totals.items()}
