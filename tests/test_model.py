import torch

from scpisces.losses import ScPISCESLoss
from scpisces.model import ScPISCES, ScPISCESConfig


def test_model_forward_and_objective():
    config = ScPISCESConfig(
        vocab_size=32,
        num_classes=3,
        num_pathways=5,
        embed_dim=16,
        depth=2,
        num_heads=4,
        sae_expansion=2,
        k_guided=2,
        k_free=3,
        minimum_free_features=8,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    )
    model = ScPISCES(config)
    gene_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    values = torch.tensor([[1.0, 2.0, 0.5, 0.0], [0.5, 1.5, 0.0, 0.0]])
    total_counts = torch.tensor([100.0, 80.0])
    output = model(gene_ids, values, total_counts, gene_ids.eq(0))

    assert output["dense_logits"].shape == (2, 3)
    assert output["sparse_logits"].shape == (2, 3)
    assert output["guided_raw"].shape == (2, 5)
    assert torch.isfinite(output["reconstruction_loss"])

    objective = ScPISCESLoss()
    losses = objective(output, torch.tensor([0, 2]), torch.randn(2, 5))
    assert torch.isfinite(losses["total"])
