"""Shared figure style for every GapTrust figure.

All figures are drawn at final print size, so no rescaling is needed:
double-column width 174 mm, single-column width 84 mm (check against the
journal's current artwork guide before submission).
Each figure is exported as a vector PDF (fonts embedded) and a 600 dpi
LZW-compressed TIFF.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1.0 / 25.4
DOUBLE_COL = 174 * MM
SINGLE_COL = 84 * MM

# Colour-blind-safe palette (Okabe & Ito, 2008).
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}

# Folder layout: <project>/Codes (this repository) and <project>/Figures.
CODE_ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = CODE_ROOT.parent / "Figures"


def apply():
    """Set journal-ready matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
    })


def panel_label(ax, label, x=-0.02, y=1.02):
    """Bold panel letter, e.g. '(a)', placed above the top-left corner."""
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="right", va="bottom")


def save(fig, name, subdir="main"):
    """Export a figure as PDF and 600 dpi TIFF into Figures/<subdir>/."""
    out = FIG_ROOT / subdir
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}.pdf"
    tif = out / f"{name}.tif"
    fig.savefig(pdf)
    fig.savefig(tif, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return pdf, tif
