from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TransformerBlock


@dataclass(frozen=True)
class ScPISCESConfig:
    vocab_size: int
    num_classes: int
    num_pathways: int
    padding_idx: int = 0
    embed_dim: int = 128
    depth: int = 12
    num_heads: int = 8
    drop_path_rate: float = 0.1
    attention_dropout: float = 0.1
    sae_expansion: int = 16
    k_guided: int = 64
    k_free: int = 64
    minimum_free_features: int = 128
    prototype_temperature: float = 0.1
    mask_padding: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class PrototypicalHead(nn.Module):
    """Temperature-scaled cosine classifiers for dense and sparse views."""

    def __init__(
        self,
        dense_dim: int,
        sparse_dim: int,
        num_classes: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.dense_prototypes = nn.Parameter(torch.empty(num_classes, dense_dim))
        self.sparse_prototypes = nn.Parameter(torch.empty(num_classes, sparse_dim))
        nn.init.trunc_normal_(self.dense_prototypes, std=0.02)
        nn.init.trunc_normal_(self.sparse_prototypes, std=0.02)

    def forward(
        self, dense: torch.Tensor, sparse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dense_logits = F.normalize(dense, dim=-1) @ F.normalize(
            self.dense_prototypes, dim=-1
        ).T
        sparse_logits = F.normalize(sparse, dim=-1) @ F.normalize(
            self.sparse_prototypes, dim=-1
        ).T
        return dense_logits / self.temperature, sparse_logits / self.temperature


class HybridSparseAutoencoder(nn.Module):
    """Pathway-guided and pathway-agnostic Top-K sparse branches."""

    def __init__(
        self,
        input_dim: int,
        num_pathways: int,
        num_free_features: int,
        k_guided: int,
        k_free: int,
    ) -> None:
        super().__init__()
        if num_pathways < 1:
            raise ValueError("scPISCES requires at least one pathway feature")
        if num_free_features < 1:
            raise ValueError("num_free_features must be positive")
        self.num_guided = num_pathways
        self.num_free = num_free_features
        self.k_guided = min(k_guided, num_pathways)
        self.k_free = min(k_free, num_free_features)
        self.hidden_dim = num_pathways + num_free_features

        # Separate bias parameters preserve the parameterization used in the
        # research implementation while avoiding unused Linear biases.
        self.encoder_guided = nn.Linear(input_dim, num_pathways, bias=False)
        self.encoder_guided_bias = nn.Parameter(torch.zeros(num_pathways))
        self.encoder_free = nn.Linear(input_dim, num_free_features, bias=False)
        self.encoder_free_bias = nn.Parameter(torch.zeros(num_free_features))
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("neuron_fire_count", torch.zeros(self.hidden_dim))

        nn.init.kaiming_uniform_(self.encoder_guided.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.encoder_free.weight, a=5**0.5)
        with torch.no_grad():
            self.decoder.weight[:, :num_pathways].copy_(
                self.encoder_guided.weight.T
            )
            self.decoder.weight[:, num_pathways:].copy_(self.encoder_free.weight.T)
            self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    @staticmethod
    def _topk(values: torch.Tensor, k: int) -> torch.Tensor:
        selected_values, selected_indices = torch.topk(values, k, dim=-1)
        sparse = torch.zeros_like(values)
        return sparse.scatter(-1, selected_indices, selected_values)

    def forward(self, dense: torch.Tensor) -> dict[str, torch.Tensor]:
        centered = dense - self.decoder_bias
        guided_weight = F.normalize(self.encoder_guided.weight, dim=1)
        free_weight = F.normalize(self.encoder_free.weight, dim=1)

        guided_raw = centered @ guided_weight.T + self.encoder_guided_bias
        guided_sparse = self._topk(guided_raw, self.k_guided)

        free_raw = centered @ free_weight.T + self.encoder_free_bias
        free_sparse = F.relu(self._topk(free_raw, self.k_free))
        sparse = torch.cat([guided_sparse, free_sparse], dim=-1)

        if self.training:
            with torch.no_grad():
                self.neuron_fire_count.add_((sparse != 0).sum(dim=0))

        reconstruction = self.decoder(sparse) + self.decoder_bias
        return {
            "sparse": sparse,
            "guided_raw": guided_raw,
            "guided_sparse": guided_sparse,
            "free_sparse": free_sparse,
            "reconstruction": reconstruction,
            "reconstruction_loss": F.mse_loss(reconstruction, dense),
            "l0": (sparse != 0).float().sum(dim=-1).mean(),
        }


class ScPISCES(nn.Module):
    """Sequencing-depth-aware encoder with hybrid sparse programs."""

    def __init__(self, config: ScPISCESConfig) -> None:
        super().__init__()
        self.config = config
        self.gene_embed = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.padding_idx
        )
        self.value_embed = nn.Linear(1, config.embed_dim)
        self.total_counts_embed = nn.Linear(1, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_paths = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim,
                    config.num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    attention_dropout=config.attention_dropout,
                    drop_path=drop_paths[index],
                    norm_layer=norm_layer,
                )
                for index in range(config.depth)
            ]
        )
        self.norm = norm_layer(config.embed_dim)

        self.dense_dim = config.embed_dim * 2
        target_sparse_dim = self.dense_dim * config.sae_expansion
        num_free = max(
            config.minimum_free_features,
            target_sparse_dim - config.num_pathways,
        )
        self.sae = HybridSparseAutoencoder(
            input_dim=self.dense_dim,
            num_pathways=config.num_pathways,
            num_free_features=num_free,
            k_guided=config.k_guided,
            k_free=config.k_free,
        )
        self.pch_head = PrototypicalHead(
            dense_dim=self.dense_dim,
            sparse_dim=self.sae.hidden_dim,
            num_classes=config.num_classes,
            temperature=config.prototype_temperature,
        )

    def encode(
        self,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        total_counts: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = gene_ids.eq(self.config.padding_idx)
        gene_tokens = self.gene_embed(gene_ids) + self.value_embed(
            expression_values.unsqueeze(-1)
        )
        cls = self.cls_token + self.total_counts_embed(total_counts.reshape(-1, 1, 1))
        tokens = torch.cat([cls, gene_tokens], dim=1)

        attention_padding_mask = None
        if self.config.mask_padding:
            cls_mask = torch.zeros(
                padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device
            )
            attention_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        for block in self.blocks:
            tokens = block(tokens, attention_padding_mask)
        tokens = self.norm(tokens)

        cls_features = tokens[:, 0]
        gene_features = tokens[:, 1:]
        if self.config.mask_padding:
            valid = (~padding_mask).unsqueeze(-1).to(gene_features.dtype)
            pooled = (gene_features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        else:
            pooled = gene_features.mean(dim=1)
        return torch.cat([cls_features, pooled], dim=-1)

    def forward(
        self,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        total_counts: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dense = self.encode(
            gene_ids, expression_values, total_counts, padding_mask=padding_mask
        )
        sparse_output = self.sae(dense)
        dense_logits, sparse_logits = self.pch_head(
            dense, sparse_output["sparse"]
        )
        return {
            "dense": dense,
            "dense_logits": dense_logits,
            "sparse_logits": sparse_logits,
            **sparse_output,
        }
