# scPISCES

Implementation of **scPISCES**, a pathway-informed sparse-program learning
framework for interpretable single-cell annotation.


## Repository layout

```text
scPISCES/
|-- scpisces/                 # Installable model and data package
|-- scripts/                  # Fine-tuning and prediction entrypoints
|-- tutorials/                # Executed notebooks and compact example data
|-- pyproject.toml
`-- requirements.txt
```

Pretraining and full benchmark pipelines are not included in this release.

## Tutorials

The [tutorials](tutorials/) include saved figures and outputs that can be viewed
directly on GitHub:

- [MERFISH cell annotation and spatial maps](tutorials/Tutorial_Annotation_MERFISH.ipynb)
- [Pathway-guided programs in Pan-GI](tutorials/Tutorial_Pathway_Programs.ipynb)
- [Marker and candidate-gene prioritization](tutorials/Tutorial_Marker_Genes.ipynb)
- [Program ablation and prediction sensitivity](tutorials/Tutorial_Program_Ablation.ipynb)

Each notebook includes compact example data and can be rerun on a CPU. See the
[tutorial guide](tutorials/README.md) for setup.

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



## Fine-tuning

```bash
python scripts/train_scpisces.py \
  --train-h5ad /path/to/train.h5ad \
  --val-h5ad /path/to/validation.h5ad \
  --output-dir outputs/example
```


## License

MIT License. See [LICENSE](LICENSE).
