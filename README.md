# scPISCES

Implementation of **scPISCES**, a pathway-informed sparse-program learning
framework for interpretable single-cell annotation.

scPISCES combines a sequencing-depth-aware transcriptomic encoder with two
sparse branches: a pathway-guided branch aligned to cell-level pathway scores
and a pathway-agnostic branch that captures programs not covered by the pathway
library. Dense and sparse predictions are coupled through learnable class
prototypes and bidirectional distribution matching.

## Repository layout

```text
scPISCES/
|-- scpisces/                 # Installable model and data package
|-- scripts/                  # Fine-tuning and prediction entrypoints
|-- configs/                  # Example fine-tuning configuration
|-- tests/                    # Lightweight model smoke tests
|-- pyproject.toml
`-- requirements.txt
```

Pretraining, benchmark validation, and manuscript analysis pipelines are not
included in this release.

## Installation

```bash
git clone git@github.com:yjs193/scPISCES.git
cd scPISCES
conda create -n scpisces python=3.11 -y
conda activate scpisces
pip install -e .
```

FlashAttention is optional. If it is unavailable, scPISCES uses PyTorch
scaled dot-product attention with the same learned parameters.

## Input format

Fine-tuning expects separate training and validation `.h5ad` files. Both files
must contain:

- gene symbols in `adata.var_names`;
- cell-type labels in `adata.obs[label_key]`;
- cell-level pathway scores in `adata.obsm[pathway_key]`;
- pathway names in `adata.uns[pathway_names_key]` or an equivalent key;
- raw sequencing depth in `adata.obs[total_counts_key]` when available.

Expression values in `adata.X` should already have undergone the normalization
and log transformation selected for the study. The training script does not
normalize the expression matrix or calculate pathway scores. Validation labels
are used only for checkpoint selection. A final query/test set must not be
passed to the training command.

The vocabulary is a JSON mapping from gene symbol to integer ID. `<pad>` must
have ID 0 and `<unk>` must be present.

## Fine-tuning

```bash
python scripts/train_scpisces.py \
  --config configs/finetune_example.json \
  --train-h5ad /path/to/train.h5ad \
  --val-h5ad /path/to/validation.h5ad \
  --vocab /path/to/gene_vocab.json \
  --pretrained-checkpoint /path/to/pretrained_encoder.pth \
  --output-dir outputs/example
```

The best checkpoint is selected exclusively from the validation metric defined
by `training.selection_metric`. The output directory contains `best.pth`,
`last.pth`, `history.csv`, `config.json`, and validation predictions from the
selected checkpoint.

For long-tailed references, set `training.class_weighting` to `effective`,
`inverse`, or `sqrt_inverse`. Effective-number weighting uses

```text
w_c = (1 - beta) / (1 - beta ^ n_c),
```

followed by mean normalization and clipping.

## Prediction

```bash
python scripts/predict_scpisces.py \
  --checkpoint outputs/example/best.pth \
  --h5ad /path/to/query.h5ad \
  --vocab /path/to/gene_vocab.json \
  --output outputs/example/query_predictions.csv
```

The prediction file contains `cell_id`, `predicted_cell_type`, and
`confidence`. If the requested label column is present, the script also writes
ground-truth labels and a separate metrics JSON file. Prediction never updates
model parameters.

Legacy research checkpoints can also be used by adding `--config` and an
ordered `--class-names` JSON file. The class order must be recovered from the
reference training split; the prediction script never infers it from query
labels.

## Reproducibility notes

- Model configuration, class names, pathway order, vocabulary SHA-256, and the
  validation selection metric are stored in each public-format checkpoint.
- Gene subsampling is deterministic for validation and prediction.
- Padding-aware attention and pooling are enabled by the example configuration.
- Existing research checkpoints can be loaded explicitly with their matching
  model configuration and reference-derived class order.

## License

MIT License. See [LICENSE](LICENSE).
