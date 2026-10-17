"""Small presentation helpers shared by the example notebooks."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
