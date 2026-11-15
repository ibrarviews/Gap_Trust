"""Supplementary Figure S2 - Effect of void size on the recovery curves.

R against gap number G at 250 MHz for air-filled voids of radius 0.40 m
(pilot) and 0.15 m (test b), roof depths 0.75 and 2.50 m, after zero fill,
linear interpolation and f-k POCS. Dotted line: R = 0.5.

Inputs : results/data/pilot_R.csv, results/data/test_size_R.csv
Output : Figures/supplementary/FigS2_size_test.pdf/.tif
         results/logs/figS2_size_test.json

Run from the Codes folder:  python scripts/figS2_size_test.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, style  # noqa: E402

DATA = style.CODE_ROOT / "results" / "data"
SOURCES = {0.40: DATA / "pilot_R.csv", 0.15: DATA / "test_size_R.csv"}
METHODS = (("zero", "Zero fill"), ("linear", "Linear interpolation"), ("pocs", "f-k POCS"))
R_FAIL = 0.5

style.apply()
C = style.COLORS
CASES = [(0.40, 0.75, C["blue"], "-"), (0.40, 2.50, C["blue"], "--"),
         (0.15, 0.75, C["green"], "-"), (0.15, 2.50, C["green"], "--")]
rows = {}
for r, path in SOURCES.items():
    with open(path) as fh:
        rows[r] = [x for x in csv.DictReader(fh) if x["freq"] == "250MHz"]

fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL, 62 * style.MM),
                         constrained_layout=True, sharey=True)
for ax, (key, title) in zip(axes, METHODS):
    for radius, roof, color, ls in CASES:
        sub = [x for x in rows[radius] if x["method"] == key
               and abs(float(x["roof_m"]) - roof) < 1e-6]
        ax.plot([float(x["G"]) for x in sub], [float(x["R"]) for x in sub],
                color=color, ls=ls, lw=1.0, label=f"r = {radius:.2f} m, roof {roof:.2f} m")
    ax.axhline(R_FAIL, color="0.45", lw=0.6, ls=":")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.4)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Gap number G")
axes[0].set_ylabel("Recovery ratio R")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="outside upper center", ncol=4, frameon=False)
for ax, p in zip(axes, "abc"):
    style.panel_label(ax, p)
pdf, tif = style.save(fig, "FigS2_size_test", subdir="supplementary")
log = {"figure": "FigS2_size_test", "script": "scripts/figS2_size_test.py",
       "inputs": [str(p.relative_to(style.CODE_ROOT)) for p in SOURCES.values()],
       "frequency": "250MHz", "gaptrust_version": __version__,
       "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                   str(tif.relative_to(style.FIG_ROOT.parent))]}
(style.CODE_ROOT / "results" / "logs" / "figS2_size_test.json").write_text(json.dumps(log, indent=2))
print(pdf)
