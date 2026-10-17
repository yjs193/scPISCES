# scPISCES

**Pathway-informed sparse programs for interpretable single-cell annotation.**

[![Tutorial checks](https://github.com/yjs193/scPISCES/actions/workflows/tutorials.yml/badge.svg)](https://github.com/yjs193/scPISCES/actions/workflows/tutorials.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

scPISCES connects cell annotation with pathway-aligned sparse programs and
associated genes. Explore the saved results below, or use the model package
to fine-tune and annotate cells with your own compatible model assets.

[Quick start](docs/quickstart.md) | [Tutorial gallery](tutorials/README.md) |
[Input format](docs/input_format.md) | [FAQ](docs/faq.md)

## Explore the results

Every notebook includes executed figures, short English explanations and the
compact numerical data needed to rerun its visualizations on a CPU.

| Task | Notebook | Main views |
| --- | --- | --- |
| Cell annotation | [MERFISH](tutorials/Tutorial_Annotation_MERFISH.ipynb) | Model UMAP and spatial subtype recovery |
| Functional programs | [Pan-GI pathways](tutorials/Tutorial_Pathway_Programs.ipynb) | Program alignment and pathway activity atlas |
| Gene prioritization | [Markers and candidates](tutorials/Tutorial_Marker_Genes.ipynb) | Gene networks, expression maps and dot plots |
| Spatial interpretation | [MERFISH programs](tutorials/Tutorial_Spatial_Programs_MERFISH.ipynb) | Cell identity, programs, pathways, genes and neighborhoods |
| Tumor microenvironment | [Open-ST](tutorials/Tutorial_Tumor_Microenvironment_OpenST.ipynb) | Tissue compartments and cell-label correspondence |

[![MERFISH cell identity, guided programs, pathways and genes](tutorials/preview/spatial_programs.png)](tutorials/Tutorial_Spatial_Programs_MERFISH.ipynb)

## Get started

```bash
git clone https://github.com/yjs193/scPISCES.git
cd scPISCES
conda create -n scpisces python=3.11 -y
conda activate scpisces
```

**Visualizations only** (no GPU or model download):

```bash
python -m pip install -r tutorials/requirements.txt
python scripts/run_tutorials.py --notebook annotation --html
# Or: jupyter lab tutorials/
```

Use `--all --html` to rerun every example. Executed copies, HTML previews and
figure exports are saved under `tutorials/outputs/`. The original notebooks
are not overwritten. Example-data integrity is checked before execution.

**Fine-tuning and prediction:** install with `python -m pip install -e .`, then
follow the [model quick start](docs/quickstart.md#annotate-new-cells).
Model weights and vocabularies are not bundled with the visualization examples.
FlashAttention is optional; PyTorch scaled dot-product attention is the fallback.

## Repository layout

```text
scpisces/     Model architecture, data handling and training utilities
scripts/      Fine-tuning, prediction and tutorial execution entrypoints
tutorials/    Executed notebooks, compact example tables and preview gallery
docs/         Quick start, input requirements and common questions
```

This release does not include pretraining or full benchmark pipelines. Tutorial
data are extracts from archived study results, not a full training dataset.
See [data provenance](tutorials/data/README.md) for definitions and source details.

## License

Code is distributed under the [MIT License](LICENSE). Original datasets retain
their respective access and reuse terms.
