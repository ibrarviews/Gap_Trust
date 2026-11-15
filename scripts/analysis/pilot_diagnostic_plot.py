"""Diagnostic plot of the pilot (not a manuscript figure): R against gap width
and against G for each case and reconstruction method.

Run from the Codes folder:  python scripts/pilot_diagnostic_plot.py
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import style  # noqa: E402

rows = list(csv.DictReader(open(style.CODE_ROOT / "results" / "data" / "pilot_R.csv")))
style.apply()
C = style.COLORS
cases = [("250MHz", "0.75", C["blue"], "-"), ("250MHz", "2.5", C["blue"], "--"),
         ("400MHz", "0.75", C["vermillion"], "-"), ("400MHz", "2.5", C["vermillion"], "--")]
fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE_COL, 100 * style.MM),
                         constrained_layout=True, sharey=True)
for j, m in enumerate(("zero", "linear", "pocs")):
    for f, d, c, ls in cases:
        s = [r for r in rows if r["freq"] == f and r["roof_m"] == d and r["method"] == m]
        W = [float(r["W_m"]) for r in s]
        G = [float(r["G"]) for r in s]
        R = [float(r["R"]) for r in s]
        lab = f"{f[:3]} MHz, {float(d):.2f} m"
        axes[0, j].plot(W, R, color=c, ls=ls, label=lab)
        axes[1, j].plot(G, R, color=c, ls=ls, label=lab)
    for i in range(2):
        axes[i, j].axhline(0.5, color="0.5", lw=0.6, ls=":")
        axes[i, j].set_ylim(0, 1.2)
    axes[0, j].set_title(m)
    axes[0, j].set_xlabel("Gap width W (m)")
    axes[1, j].set_xlabel("Gap number G")
    axes[1, j].set_xlim(0, 3)
axes[0, 0].set_ylabel("R")
axes[1, 0].set_ylabel("R")
axes[0, 0].legend(frameon=False, fontsize=6)
out = style.CODE_ROOT / "results" / "diagnostics" / "pilot_R_curves.png"
fig.savefig(out, dpi=200)
print(out)
