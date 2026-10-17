# Common questions

### Do the tutorials run the model?

The default cells redraw archived numerical results using small bundled tables.
They are CPU-only and do not train or download a model. The MERFISH annotation
notebook has an optional command for inference with your own compatible assets.

### Where are the weights and full datasets?

This release contains the model implementation, fine-tuning/prediction scripts
and visualization examples. Checkpoints, vocabularies and full expression files
are not bundled. Do not use the example CSVs as a training dataset.

### Can I change the displayed populations or genes?

Yes. Edit the population lists near each plotting cell. Gene and program choices
must be present in the bundled table; displaying additional genes requires
exporting their expression from your own data first. The examples deliberately
avoid downloading large atlases in a setup cell.

### Why does the gene-expression background remain visible?

The expression palette starts at light gray rather than white. Zero-expression
cells therefore retain the embedding or tissue outline. Legends describe the
displayed scaling, which may differ between panels.

### Are a guided feature and its pathway score the same quantity?

No. A guided sparse activation is a learned model feature. A pathway score is
an expression-derived alignment target. Their agreement is descriptive evidence
of alignment, not an independent pathway-enrichment test or causal validation.

### Which identities are used for population summaries?

Pan-GI program and gene summaries use curated cell identities. MERFISH and
Open-ST comparison maps show both curated and predicted identities explicitly.
The MERFISH neighborhood tutorial constructs each network from its stated label
column and uses the whole spatial graph for expected counts.

### Are candidate genes validated markers?

No. Known/candidate status reproduces the study's population-specific marker
lookup. Correlation prioritizes genes for follow-up, not functional validation.
Likewise, neighborhood and gene-association networks are not regulatory graphs.

### Why do fonts or outputs differ slightly on my machine?

Arial must be installed locally; otherwise a sans-serif fallback is used.
SVG text remains editable. Library versions and available fonts can change
layout slightly; the runner checks numerical assertions and data integrity,
not pixel-for-pixel equality.

### What should I include in a bug report?

Use the repository's [issue tracker](https://github.com/yjs193/scPISCES/issues)
with your Python/package versions, command, input dimensions and error message.
For tutorial issues, attach `tutorials/outputs/executed/run_summary.json`.
Do not post patient-level data, private credentials or checkpoint files.
