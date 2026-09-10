# scPISCES

Implementation of **scPISCES**, a pathway-informed sparse-program learning
framework for interpretable single-cell annotation.


## Repository layout

```text
scPISCES/
|-- scpisces/                 # Installable model and data package
|-- scripts/                  # Fine-tuning and prediction entrypoints
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



## Fine-tuning

```bash
python scripts/train_scpisces.py \
  --train-h5ad /path/to/train.h5ad \
  --val-h5ad /path/to/validation.h5ad \
  --output-dir outputs/example
```


## License

MIT License. See [LICENSE](LICENSE).
