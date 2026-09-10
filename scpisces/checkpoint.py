from __future__ import annotations

import hashlib
from pathlib import Path

import torch


ENCODER_PREFIXES = (
    "gene_embed.",
    "value_embed.",
    "total_counts_embed.",
    "cls_token",
    "blocks.",
    "norm.",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_payload(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload if isinstance(payload, dict) else {"model": payload}


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_weights(
    model: torch.nn.Module,
    path: str | Path,
    encoder_only: bool = False,
    require_all_compatible: bool = False,
) -> dict:
    payload = checkpoint_payload(path)
    state = payload.get("model", payload)
    state = _clean_state_dict(state)
    if encoder_only:
        state = {
            key: value
            for key, value in state.items()
            if key.startswith(ENCODER_PREFIXES)
        }

    target = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    skipped = sorted(set(state) - set(compatible))
    incompatible = model.load_state_dict(compatible, strict=False)
    report = {
        "loaded_tensors": len(compatible),
        "skipped_tensors": skipped,
        "missing_tensors": list(incompatible.missing_keys),
        "unexpected_tensors": list(incompatible.unexpected_keys),
    }
    if require_all_compatible and (skipped or incompatible.missing_keys):
        raise RuntimeError(f"Checkpoint is not fully compatible: {report}")
    return report
