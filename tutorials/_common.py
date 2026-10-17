"""Small presentation helpers shared by the example notebooks."""

from pathlib import Path
import re
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


def configure_plots():
    # Register Arial when installed outside Matplotlib's cached font catalogue.
    for candidate in (
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.exists():
            font_manager.fontManager.addfont(candidate)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def expression_cmap():
    return LinearSegmentedColormap.from_list(
        "expression", ["#e6e8eb", "#f8c6d0", "#e86f96", "#992756"]
    )


def program_cmap():
    return LinearSegmentedColormap.from_list(
        "program", ["#e8ecef", "#bde6df", "#55b4b0", "#007d93", "#254b83"]
    )


def pathway_cmap():
    return LinearSegmentedColormap.from_list(
        "pathway", ["#eceef0", "#fff0c2", "#fbb777", "#ed744e", "#d93b3b"]
    )


def dot_atlas(values, fractions, color_label, fraction_label="Cells with activity > 0 (%)"):
    """Plot row-by-column activity with horizontal legends above the matrix."""
    if not values.index.equals(fractions.index) or not values.columns.equals(fractions.columns):
        raise ValueError("Activity and fraction matrices must have identical axes")
    row_labels = [textwrap.fill(pretty(x), 48) for x in values.index]
    longest = max(label.count("\n") + 1 for label in row_labels)
    height = max(6.0, len(row_labels) * max(0.46, 0.17 * longest + 0.12) + 2.1)
    fig = plt.figure(figsize=(12.5, height), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.35, height - 1.4], hspace=0.10)
    sizes_ax = fig.add_subplot(grid[0, 0])
    color_ax = fig.add_subplot(grid[0, 1])
    ax = fig.add_subplot(grid[1, :])
    sizes_ax.set_axis_off()
    sizes_ax.text(0.0, 0.78, fraction_label, transform=sizes_ax.transAxes, fontsize=10)
    for x, fraction in zip([0.48, 0.66, 0.84], [0.2, 0.5, 0.8]):
        sizes_ax.scatter([x], [0.25], s=12 + 190 * fraction, color="#81909a", linewidths=0)
        sizes_ax.text(x + 0.045, 0.25, f"{fraction:.0%}", va="center", fontsize=9)
    sizes_ax.set_xlim(0, 1)
    sizes_ax.set_ylim(0, 1)
    xx, yy = np.meshgrid(np.arange(values.shape[1]), np.arange(values.shape[0]))
    image = ax.scatter(
        xx.ravel(), yy.ravel(), c=values.to_numpy().ravel(),
        s=12 + 190 * fractions.to_numpy().ravel(), cmap=program_cmap(),
        vmin=0, vmax=1, edgecolors="#8c9ba5", linewidths=0.25,
    )
    colorbar = fig.colorbar(image, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_label(color_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_xticks(range(values.shape[1]), [pretty(x) for x in values.columns], rotation=40, ha="right")
    ax.set_yticks(range(values.shape[0]), row_labels)
    ax.set_xlim(-0.7, values.shape[1] - 0.3)
    ax.set_ylim(values.shape[0] - 0.4, -0.7)
    ax.tick_params(length=0, labelsize=10)
    ax.grid(axis="x", color="#eff1f3", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def save_figure(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            output_dir / f"{name}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )


def clean_map(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def pretty(name):
    return str(name).replace("_", " ")


def pathway_label(identifier):
    value = str(identifier).removeprefix("REACTOME_").replace("_", " ").lower()
    for acronym in ("BCR", "GIP", "FLT3", "SRC", "TLR", "ECM", "UPR", "ATF6", "HDL", "RNA", "DNA", "PD"):
        value = re.sub(r"\b" + acronym.lower() + r"\b", acronym, value)
    value = re.sub(r"\bb cell\b", "B cell", value)
    value = re.sub(r"\btype i\b", "type I", value)
    return value[0].upper() + value[1:] if value else value
