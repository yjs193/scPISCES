# Tutorials

Open a notebook on GitHub to see the saved figures and tables. Each example
includes the small data extracts needed to rerun its analysis on a CPU.

| Notebook | Example |
| --- | --- |
| [Cell annotation](Tutorial_Annotation_MERFISH.ipynb) | MERFISH predictions, model-embedding UMAPs and spatial subtype maps |
| [Pathway programs](Tutorial_Pathway_Programs.ipynb) | Pan-GI guided programs and their aligned Reactome activities |
| [Marker prioritization](Tutorial_Marker_Genes.ipynb) | Program-gene networks, expression UMAPs and marker dot plots |
| [Spatial programs](Tutorial_Spatial_Programs_MERFISH.ipynb) | Cell identity, sparse programs, pathways and genes in tissue; spatial neighborhood networks |

## Preview

| MERFISH annotation | Pan-GI pathway atlas |
| --- | --- |
| [![MERFISH annotations](preview/annotation.png)](Tutorial_Annotation_MERFISH.ipynb) | [![Pathway activity atlas](preview/pathways.png)](Tutorial_Pathway_Programs.ipynb) |
| **Marker and candidate genes** | **Spatial program localization** |
| [![Marker expression](preview/markers.png)](Tutorial_Marker_Genes.ipynb) | [![Spatial programs](preview/spatial_programs.png)](Tutorial_Spatial_Programs_MERFISH.ipynb) |

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
