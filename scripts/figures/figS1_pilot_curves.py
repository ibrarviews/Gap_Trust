"""Supplementary Figure S1 - Pilot target-recovery curves.

Target-recovery ratio R of the pilot cases (air-filled void, r = 0.40 m,
roof depths 0.75 and 2.50 m, 250 and 400 MHz) against gap width W (top row)
and gap number G = W/D (bottom row), for zero fill, linear interpolation and
f-k POCS. The dotted line marks the failure threshold R = 0.5 (-6 dB).

Input : results/data/pilot_R.csv  (scripts/pilot_analysis.py)
Output: Figures/supplementary/FigS1_pilot_curves.pdf/.tif
        results/logs/figS1_pilot_curves.json

Run from the Codes folder:  python scripts/figS1_pilot_curves.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, style  # noqa: E402

SRC = style.CODE_ROOT / "results" / "data" / "pilot_R.csv"
R_FAIL = 0.5
METHODS = (("zero", "Zero fill"), ("linear", "Linear interpolation"), ("pocs", "f-k POCS"))

with open(SRC) as fh:
    rows = list(csv.DictReader(fh))

style.apply()
C = style.COLORS
CASES = [("250MHz", 0.75, C["blue"], "-"), ("250MHz", 2.50, C["blue"], "--"),
         ("400MHz", 0.75, C["vermillion"], "-"), ("400MHz", 2.50, C["vermillion"], "--")]

fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE_COL, 110 * style.MM),
                         constrained_layout=True, sharey=True)
for j, (key, title) in enumerate(METHODS):
    for freq, roof, color, ls in CASES:
        sub = [r for r in rows if r["freq"] == freq and r["method"] == key
               and abs(float(r["roof_m"]) - roof) < 1e-6]
        W = [float(r["W_m"]) for r in sub]
        G = [float(r["G"]) for r in sub]
        R = [float(r["R"]) for r in sub]
        label = f"{freq[:3]} MHz, roof {roof:.2f} m"
        axes[0, j].plot(W, R, color=color, ls=ls, lw=1.0, label=label)
        axes[1, j].plot(G, R, color=color, ls=ls, lw=1.0, label=label)
    for i in range(2):
        axes[i, j].axhline(R_FAIL, color="0.45", lw=0.6, ls=":")
        axes[i, j].set_ylim(0, 1.25)
    axes[0, j].set_title(title, fontsize=8)
    axes[0, j].set_xlim(0, 2.5)
    axes[1, j].set_xlim(0, 3.0)
    axes[0, j].set_xlabel("Gap width W (m)")
    axes[1, j].set_xlabel("Gap number G")
axes[0, 0].set_ylabel("Recovery ratio R")
axes[1, 0].set_ylabel("Recovery ratio R")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
for ax, lab in zip(axes.ravel(), "abcdef"):
    style.panel_label(ax, lab)

pdf, tif = style.save(fig, "FigS1_pilot_curves", subdir="supplementary")

log = {"figure": "FigS1_pilot_curves", "script": "scripts/figS1_pilot_curves.py",
       "input": "results/data/pilot_R.csv", "gaptrust_version": __version__,
       "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "R_fail": R_FAIL,
       "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                   str(tif.relative_to(style.FIG_ROOT.parent))]}
(style.CODE_ROOT / "results" / "logs" / "figS1_pilot_curves.json").write_text(json.dumps(log, indent=2))
print(pdf, tif)
