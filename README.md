# scPISCES

**Pathway-informed sparse programs for interpretable single-cell annotation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

scPISCES connects cell annotation with pathway-aligned sparse programs and
associated genes. Explore the saved results below, or use the model package
to fine-tune and annotate cells with your own compatible model assets.


## Tutorials

| Task | Notebook | Main views |
| --- | --- | --- |
| Cell annotation | [MERFISH](tutorials/Tutorial_Annotation_MERFISH.ipynb) | Model UMAP and spatial subtype recovery |
| Functional programs | [Pan-GI pathways](tutorials/Tutorial_Pathway_Programs.ipynb) | Program alignment and pathway activity atlas |
| Gene prioritization | [Markers and candidates](tutorials/Tutorial_Marker_Genes.ipynb) | Gene networks, expression maps and dot plots |
| Spatial interpretation | [MERFISH programs](tutorials/Tutorial_Spatial_Programs_MERFISH.ipynb) | Cell identity, programs, pathways, genes and neighborhoods |
| Tumor microenvironment | [Open-ST](tutorials/Tutorial_Tumor_Microenvironment_OpenST.ipynb) | Tissue compartments and cell-label correspondence |


## Get started

```bash
git clone https://github.com/yjs193/scPISCES.git
cd scPISCES
conda create -n scpisces python=3.11 -y
conda activate scpisces
```


```bash
python -m pip install -r tutorials/requirements.txt

```


## License

Code is distributed under the [MIT License](LICENSE). Original datasets retain
their respective access and reuse terms.
