# Quick start

## View the results

Open the [tutorial gallery](../tutorials/README.md). The notebooks retain their
plots and tables, so reading them on GitHub needs no installation. The examples
use real archived study outputs, not simulated cells or newly trained models.

## Rerun the visualizations

Python 3.10 or newer is required; Python 3.11 is used by the notebook CI.
From a clone of this repository:

```bash
python -m pip install -r tutorials/requirements.txt
python scripts/run_tutorials.py --list
python scripts/run_tutorials.py --notebook annotation --html
```

Run all five tutorials with `--all --html`. Executed copies, HTML pages and a
JSON run summary go to `tutorials/outputs/executed/`; original notebooks are
unchanged. Figure PNGs and editable-text SVGs go to `tutorials/outputs/`.
The runner checks bundled data against SHA-256 checksums before execution.

For interactive editing, run `jupyter lab tutorials/`. Choose a kernel from the
environment where the tutorial requirements were installed. The examples need
neither a GPU nor the full scPISCES package. Arial is used when available;
otherwise the plots use a local sans-serif fallback. Fonts are not bundled.

## Annotate new cells

Install the model package with `python -m pip install -e .`. Supply a compatible
fine-tuned checkpoint, its exact gene vocabulary and a prepared `.h5ad` file:

```bash
python scripts/predict_scpisces.py \
  --checkpoint /path/to/best.pth \
  --vocab /path/to/vocab.json \
  --h5ad /path/to/query.h5ad \
  --output outputs/query_annotations.csv
```

CUDA is the default. Add `--device cpu --no-amp` for CPU inference, or lower
`--batch-size` if GPU memory is limited. The output CSV contains cell IDs,
predicted labels and confidence scores; a companion `.metrics.json` records
the run. Ground-truth labels are optional and are not used to predict labels.

Model weights, gene vocabularies and full expression atlases are **not included**
in the visualization bundle. No automatic model download is configured in this
release. See [input and model assets](input_format.md) before using your own data.

## Fine-tune a model

Use the configuration distributed with your compatible model assets. All
required arguments are shown below; `/path/to/...` entries are user-supplied:

```bash
python scripts/train_scpisces.py \
  --config /path/to/finetune.json \
  --vocab /path/to/vocab.json \
  --pretrained-checkpoint /path/to/pretrained.pth \
  --train-h5ad /path/to/reference_train.h5ad \
  --val-h5ad /path/to/reference_validation.h5ad \
  --output-dir outputs/finetune
```

The public entrypoint writes `best.pth`, `last.pth`, `history.csv` and run
metadata. It selects a model using reference validation data only. Pretraining
and full benchmark pipelines are outside this source release.

Next: [input format](input_format.md) or [common questions](faq.md).
