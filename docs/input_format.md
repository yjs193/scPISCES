# Input and model assets

## AnnData input

| Field | Prediction | Fine-tuning |
| --- | --- | --- |
| `adata.X` | Cells by genes; prepared non-negative expression values | Same representation and preprocessing as the starting encoder |
| `adata.var_names` | Gene identifiers matching the vocabulary | Same identifier convention |
| `adata.obs_names` | Unique cell identifiers | Unique cell identifiers |
| `adata.obs['total_counts']` | Original library size, retained before normalization | Original library size, retained before normalization |
| `adata.obs['celltype']` | Not required | Reference cell labels; configurable via `data.label_key` |
| `adata.obsm['Reactome_pathway']` | Not required | Cells by pathways, with identical pathway-column order in both input files |
| `adata.uns['Reactome_pathway_cols']` | Not required | Ordered pathway identifiers matching the score columns |

The entrypoints do not perform QC, normalization or log transformation for you.
Use preprocessing compatible with your encoder; do not normalize an already
normalized matrix again. Keep the original sequencing-depth covariate rather
than substituting the sum of log-transformed values. The loader has a matrix-sum
fallback, but that is not raw library size after transformation.

Spatial coordinates are optional plotting metadata, not annotation inputs.
The bundled tutorial CSVs contain only the numerical columns needed for their
figures; they are not substitutes for full `.h5ad` prediction inputs.

## Vocabulary and checkpoint

The vocabulary is a JSON mapping from gene/token strings to integer IDs, for
example `{"<pad>": 0, "<unk>": 1, "GENE_A": 2}`. This example is a schema,
not a usable biological vocabulary. IDs must be unique and non-negative;
`<pad>` must be 0, and `<unk>` must use a different ID.

Keep the exact vocabulary paired with the checkpoint, including token order.
Changing an ID changes the embedding associated with a gene. Do not append
tokens or replace a human vocabulary with a mouse vocabulary just to make a file
load. Unmapped genes receive `<unk>`; inspect coverage before inference:

```python
import json
import anndata as ad

vocab = json.load(open("/path/to/vocab.json"))
adata = ad.read_h5ad("/path/to/query.h5ad", backed="r")
matched = adata.var_names.isin(vocab)
print(f"Vocabulary coverage: {matched.sum():,}/{adata.n_vars:,} genes ({matched.mean():.1%})")
adata.file.close()
```

Low coverage usually warrants checking gene symbols versus Ensembl identifiers,
species, capitalization and duplicate genes. Coverage alone does not demonstrate
that a model is suitable for a new tissue.

Checkpoints produced by the public training script store model configuration,
ordered class names and the vocabulary-file SHA-256. Prediction checks this hash.
Legacy research checkpoints additionally need `--config` and `--class-names`.
The latter is an **ordered JSON list from the original reference model**, not
the unique labels found in a query. Load only trusted checkpoint files.

## Configuration fields

The fine-tuning JSON has `model`, `data` and `training` sections plus an optional
`seed`. Keep architecture settings consistent with the starting encoder.

| Section | Required or important entries |
| --- | --- |
| `model` | Encoder and sparse-branch settings accepted by `ScPISCESConfig`; vocabulary size, class count and pathway count are inferred from the supplied data |
| `data` | `max_genes`; optional `label_key`, `pathway_key`, `pathway_names_key`, `total_counts_key`, `gene_selection` |
| `training` | `batch_size`, `learning_rate`, `minimum_learning_rate`, `epochs`; optional `prediction_head`, `selection_metric`, `patience` and loss weights |

`k_guided` and `k_free` are separate per-branch Top-K budgets, not a shared
total. A guided feature's pathway identity depends on the stored pathway order;
feature numbers cannot be transferred between separately fitted models.
