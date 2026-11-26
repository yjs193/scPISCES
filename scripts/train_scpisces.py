#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scpisces.checkpoint import load_weights, sha256
from scpisces.data import (
    AnnDataCellDataset,
    collate_cells,
    fit_pathway_standardizer,
    load_vocabulary,
    pathway_matrix,
    read_pathway_names,
)
from scpisces.evaluation import predict
from scpisces.losses import ScPISCESLoss, compute_class_weights
from scpisces.model import ScPISCES, ScPISCESConfig
from scpisces.training import cosine_learning_rate_factor, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune scPISCES with validation-only model selection"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--val-h5ad", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_checkpoint(
    model: ScPISCES,
    config: dict,
    model_config: ScPISCESConfig,
    class_names: list[str],
    pathway_names: list[str],
    pathway_mean: np.ndarray,
    pathway_std: np.ndarray,
    vocabulary_hash: str,
    epoch: int,
    validation_metrics: dict,
) -> dict:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "config": config,
        "model_config": model_config.to_dict(),
        "class_names": class_names,
        "pathway_names": pathway_names,
        "pathway_standardization": {
            "mean": pathway_mean.tolist(),
            "std": pathway_std.tolist(),
        },
        "vocab_sha256": vocabulary_hash,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    model_options = config["model"]
    training = config["training"]
    data_options = config["data"]
    seed = int(config.get("seed", 2026))
    set_seed(seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp = device.type == "cuda" and not args.no_amp

    vocabulary = load_vocabulary(args.vocab)
    train_adata = ad.read_h5ad(args.train_h5ad)
    val_adata = ad.read_h5ad(args.val_h5ad)
    label_key = data_options.get("label_key", "celltype")
    if label_key not in train_adata.obs or label_key not in val_adata.obs:
        raise ValueError(f"Both splits must contain adata.obs[{label_key!r}]")

    class_names = sorted(train_adata.obs[label_key].astype(str).unique().tolist())
    class_mapping = {name: index for index, name in enumerate(class_names)}
    unseen = sorted(set(val_adata.obs[label_key].astype(str)) - set(class_names))
    if unseen:
        raise ValueError(f"Validation contains classes absent from training: {unseen}")

    pathway_key = data_options.get("pathway_key", "Reactome_pathway")
    train_pathways = pathway_matrix(train_adata, pathway_key)
    val_pathways = pathway_matrix(val_adata, pathway_key)
    if train_pathways is None or val_pathways is None:
        raise ValueError("Both train and validation data require pathway scores")
    if train_pathways.shape[1] != val_pathways.shape[1]:
        raise ValueError("Train and validation pathway dimensions differ")
    pathway_mean, pathway_std = fit_pathway_standardizer(train_pathways)
    pathway_names = read_pathway_names(
        train_adata,
        data_options.get("pathway_names_key", "Reactome_pathway_cols"),
        train_pathways.shape[1],
    )

    max_genes = int(data_options["max_genes"])
    dataset_arguments = {
        "vocabulary": vocabulary,
        "max_genes": max_genes,
        "class_mapping": class_mapping,
        "label_key": label_key,
        "pathway_key": pathway_key,
        "pathway_mean": pathway_mean,
        "pathway_std": pathway_std,
        "total_counts_key": data_options.get("total_counts_key", "total_counts"),
        "gene_selection": data_options.get("gene_selection", "random"),
        "seed": seed,
        "require_pathways": True,
    }
    train_dataset = AnnDataCellDataset(
        train_adata, training=True, **dataset_arguments
    )
    val_dataset = AnnDataCellDataset(val_adata, training=False, **dataset_arguments)
    collate = partial(
        collate_cells, max_genes=max_genes, pad_token_id=vocabulary["<pad>"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_config = ScPISCESConfig(
        vocab_size=max(vocabulary.values()) + 1,
        num_classes=len(class_names),
        num_pathways=train_pathways.shape[1],
        padding_idx=vocabulary["<pad>"],
        **model_options,
    )
    model = ScPISCES(model_config).to(device)
    if args.pretrained_checkpoint:
        report = load_weights(model, args.pretrained_checkpoint, encoder_only=True)
        print(json.dumps({"pretrained_load": report}, indent=2))

    backbone_names = (
        "gene_embed",
        "value_embed",
        "total_counts_embed",
        "cls_token",
        "blocks",
        "norm",
    )
    backbone_parameters = []
    sparse_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith(backbone_names):
            backbone_parameters.append(parameter)
        else:
            sparse_parameters.append(parameter)
    learning_rate = float(training["learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": learning_rate
                * float(training.get("backbone_learning_rate_scale", 0.9)),
            },
            {"params": sparse_parameters, "lr": learning_rate},
        ],
        weight_decay=float(training.get("weight_decay", 0.05)),
    )
    minimum_factor = float(training["minimum_learning_rate"]) / learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_learning_rate_factor(
            epoch,
            int(training["epochs"]),
            int(training.get("warmup_epochs", 0)),
            minimum_factor,
        ),
    )

    counts = (
        train_adata.obs[label_key]
        .astype(str)
        .value_counts()
        .reindex(class_names)
        .to_numpy()
    )
    class_weights = compute_class_weights(
        counts,
        training.get("class_weighting", "none"),
        beta=float(training.get("effective_number_beta", 0.9992)),
        maximum=float(training.get("maximum_class_weight", 20.0)),
    )
    objective = ScPISCESLoss(
        label_smoothing=float(training.get("label_smoothing", 0.1)),
        class_weights=class_weights,
        lambda_reconstruction=float(training.get("lambda_reconstruction", 1.0)),
        lambda_pathway=float(training.get("lambda_pathway", 1.0)),
        lambda_consistency=float(training.get("lambda_consistency", 1.5)),
    ).to(device)
    scaler = make_scaler(amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    selection_head = training.get("prediction_head", "dense")
    selection_metric = training.get("selection_metric", "macro_f1")
    best_value = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict] = []
    vocabulary_hash = sha256(args.vocab)
    started = time.time()

    for epoch in range(1, int(training["epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            objective,
            device,
            scaler,
            float(training.get("gradient_clip_norm", 1.0)),
            amp,
        )
        dense_metrics, _ = predict(
            model, val_loader, device, class_names, head="dense", amp=amp
        )
        sparse_metrics, _ = predict(
            model, val_loader, device, class_names, head="sparse", amp=amp
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_dense_{key}": value for key, value in dense_metrics.items()},
            **{f"val_sparse_{key}": value for key, value in sparse_metrics.items()},
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "sparse_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)

        selected_metrics = dense_metrics if selection_head == "dense" else sparse_metrics
        if selection_metric not in selected_metrics:
            raise ValueError(
                f"selection_metric must be one of {sorted(selected_metrics)}"
            )
        selected_value = float(selected_metrics[selection_metric])
        payload = make_checkpoint(
            model,
            config,
            model_config,
            class_names,
            pathway_names,
            pathway_mean,
            pathway_std,
            vocabulary_hash,
            epoch,
            {"dense": dense_metrics, "sparse": sparse_metrics},
        )
        torch.save(payload, args.output_dir / "last.pth")
        if selected_value > best_value:
            best_value = selected_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(payload, args.output_dir / "best.pth")
        else:
            stale_epochs += 1
        scheduler.step()

        if (
            epoch >= int(training.get("min_epochs", 1))
            and stale_epochs >= int(training.get("patience", 10))
        ):
            break

    best_payload = torch.load(
        args.output_dir / "best.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_payload["model"], strict=True)
    best_metrics, best_predictions = predict(
        model, val_loader, device, class_names, head=selection_head, amp=amp
    )
    best_predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    summary = {
        "best_epoch": best_epoch,
        "selection_head": selection_head,
        "selection_metric": selection_metric,
        "best_selection_value": best_value,
        "validation_metrics": best_metrics,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
