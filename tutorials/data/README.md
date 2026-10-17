# Example data

These files contain compact numerical extracts from completed scPISCES analyses.
They allow the notebooks to execute without downloading full atlases or model
weights. No synthetic cells or fabricated results are included.

| File | Contents |
| --- | --- |
| `merfish_cells.csv.gz` | 6,602 cells: cell ID, curated and predicted identity, spatial position and model-embedding UMAP coordinates |
| `merfish_expected_metrics.json` | Accuracy and macro-F1 recomputed from the archived MERFISH prediction table |
| `merfish_program_cells.csv.gz` | Per-cell activities for SE82, SE419 and SE1453, aligned pathway scores and six associated genes |
| `merfish_programs.csv` | The three spatial examples, aligned pathways and known/candidate gene choices |
| `pangi_cells.csv.gz` | 31,000 cells: curated identity, model UMAP, four guided sparse activations, their aligned pathway activities, two rectified pre-Top-K guided activations and eight gene-expression columns |
| `pangi_programs.csv` | Guided feature IDs, representative populations and exact Reactome pathway identifiers |
| `pangi_gene_rankings.csv` | Archived gene rankings for SE88 and SE1551, including known/candidate status |
| `pangi_pathway_atlas.csv.gz` | Mean pathway activity and positive-score fractions across all 31 Pan-GI populations |
| `provenance.json` | Source artifact identifiers, processing definitions, SHA-256 checksums and numerical consistency checks |

## Interpretation

MERFISH predictions are the archived sparse-view annotations for the mouse
spinal-cord example with 17 reference sections. UMAP coordinates visualize the
corresponding cached model representations. The annotation notebook recomputes
its metrics from the prediction table, not from a separate inference cache.
Spatial program activities come from the archived interpretation cache and are
joined to these annotations by exact cell ID. Neighborhood enrichment is
recomputed in the notebook from the complete eight-nearest-neighbor graph,
using whole-section frequencies for its independent-mixing expectation.

Pan-GI population summaries use curated annotations. Columns named
`sparse_SE...` contain positive post-Top-K activations. Columns named
`guided_SE...` contain rectified pre-Top-K activations used in the archived
marker analysis. Pathway scores are min-max normalized per pathway across the
31,000 example cells. Gene-expression columns contain log1p-transformed prepared
expression values, matching the archived marker figures. Archived association
scores use an epsilon-stabilized standardized matrix product, so they can differ
slightly from a direct Pearson correlation.

The guided feature index is zero-based and local to its fitted model. A Reactome
name is a training alignment target, not a post-hoc enrichment result. Feature
IDs should not be transferred between models without checking pathway order.

Known/candidate status reproduces the study's population-specific CellMarker
and PanglaoDB lookup. A candidate designation does not establish a new marker.
Network edges in the gene tutorial represent program-gene associations;
spatial-network edges represent cell-type neighborhood enrichment. Neither
network is a gene-regulatory or causal interaction graph.

## Data sources

- **Pan-GI:** Oliver et al., *Single-cell integration reveals metaplasia in
  inflammatory gut diseases*, Nature (2024).
- **Mouse spinal cord MERFISH:** Wang et al., *Spatial transcriptomics reveals
  organizational properties of mouse spinal cord and alterations in neuropathic
  pain* (2026), as cited in the scPISCES manuscript.
- **Pathway identifiers:** Reactome. Marker annotations: CellMarker and PanglaoDB.

These extracts accompany the scPISCES tutorial examples; consult the original
dataset sources for full data access and reuse terms.
