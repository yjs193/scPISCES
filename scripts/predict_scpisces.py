#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import anndata as ad
import torch
from torch.utils.data import DataLoader

from scpisces.checkpoint import checkpoint_payload, load_weights, sha256
from scpisces.data import AnnDataCellDataset, collate_cells, load_vocabulary
from scpisces.evaluation import predict
from scpisces.model import ScPISCES, ScPISCESConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate cells with scPISCES")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Fine-tuning config required only for legacy research checkpoints",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        help="Ordered JSON class-name list required only for legacy checkpoints",
    )
    parser.add_argument("--head", choices=("dense", "sparse"))
    parser.add_argument("--label-key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def legacy_model_config(
    payload: dict,
    config: dict,
    vocabulary: dict[str, int],
    class_names: list[str],
) -> ScPISCESConfig:
    state = payload.get("model", payload)
    embed_dim = state["gene_embed.weight"].shape[1]
    num_pathways = state["sae.encoder_guided.weight"].shape[0]
    num_free = state["sae.encoder_free.weight"].shape[0]
    sparse_expansion = (num_pathways + num_free) // (2 * embed_dim)
    return ScPISCESConfig(
        vocab_size=state["gene_embed.weight"].shape[0],
        num_classes=len(class_names),
        num_pathways=num_pathways,
        padding_idx=vocabulary["<pad>"],
        embed_dim=embed_dim,
        sae_expansion=sparse_expansion,
        **{
            key: value
            for key, value in config["model"].items()
            if key not in {"embed_dim", "sae_expansion"}
        },
    )


def main() -> None:
    args = parse_args()
    payload = checkpoint_payload(args.checkpoint)
    vocabulary = load_vocabulary(args.vocab)
    expected_hash = payload.get("vocab_sha256")
    observed_hash = sha256(args.vocab)
    if expected_hash and expected_hash != observed_hash:
        raise ValueError("Vocabulary SHA-256 does not match the checkpoint")

    is_public_checkpoint = all(
        key in payload for key in ("model_config", "config", "class_names")
    )
    if is_public_checkpoint:
        config = payload["config"]
        model_config = ScPISCESConfig(**payload["model_config"])
        class_names = [str(item) for item in payload["class_names"]]
    else:
        if args.config is None or args.class_names is None:
            raise ValueError(
                "Legacy checkpoints require --config and --class-names. The class "
                "order must come from the reference training split, not query labels."
            )
        config = json.loads(args.config.read_text())
        class_names = [str(item) for item in json.loads(args.class_names.read_text())]
        model_config = legacy_model_config(
            payload, config, vocabulary, class_names
        )

    data_options = config["data"]
    training = config["training"]
    model = ScPISCES(model_config)
    report = load_weights(
        model,
        args.checkpoint,
        require_all_compatible=is_public_checkpoint,
    )
    print(json.dumps({"checkpoint_load": report}, indent=2))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model.to(device)
    adata = ad.read_h5ad(args.h5ad)
    max_genes = int(data_options["max_genes"])
    class_mapping = {name: index for index, name in enumerate(class_names)}
    dataset = AnnDataCellDataset(
        adata,
        vocabulary=vocabulary,
        max_genes=max_genes,
        class_mapping=class_mapping if args.label_key else None,
        label_key=args.label_key,
        total_counts_key=data_options.get("total_counts_key", "total_counts"),
        gene_selection=data_options.get("gene_selection", "random"),
        seed=int(config.get("seed", 2026)),
        training=False,
        require_pathways=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(training["batch_size"]),
        shuffle=False,
        collate_fn=partial(
            collate_cells,
            max_genes=max_genes,
            pad_token_id=vocabulary["<pad>"],
        ),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    head = args.head or training.get("prediction_head", "dense")
    metrics, predictions = predict(
        model,
        loader,
        device,
        class_names,
        head=head,
        amp=device.type == "cuda" and not args.no_amp,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "prediction_head": head,
        "n_cells": len(predictions),
        **metrics,
    }
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
