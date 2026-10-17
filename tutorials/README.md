# Tutorials

Open a notebook on GitHub to see the saved figures and tables. Each example
includes the small data extracts needed to rerun its analysis on a CPU.

| Notebook | Example |
| --- | --- |
| [Cell annotation](Tutorial_Annotation_MERFISH.ipynb) | MERFISH predictions, model-embedding UMAPs and spatial subtype maps |
| [Pathway programs](Tutorial_Pathway_Programs.ipynb) | Pan-GI guided programs and their aligned Reactome activities |
| [Marker prioritization](Tutorial_Marker_Genes.ipynb) | Known markers, candidate genes and expression localization |
| [Program ablation](Tutorial_Program_Ablation.ipynb) | Dose response and matched controls for internal program removal |

## Run locally

From the repository root:

```bash
pip install -r tutorials/requirements.txt
jupyter lab tutorials/
```

Run each notebook from top to bottom. Figures are also exported to
`tutorials/outputs/` as PNG and SVG with editable text. Arial is used when
installed; otherwise a local sans-serif fallback is used.

The annotation notebook includes an optional command for applying a fine-tuned
scPISCES checkpoint to a new query. That step additionally requires the main
package, a compatible checkpoint, its gene vocabulary and an expression file.
These model assets are not part of the small example bundle.

## Example data

The files in [data/](data/) are compact extracts from the study's archived
numerical results, rather than simulated observations. MERFISH examples use
6,602 query cells; Pan-GI cell-level examples use 31,000 cells. Population-level
program summaries use curated annotations. Details and source identifiers are
listed in [data/README.md](data/README.md).
