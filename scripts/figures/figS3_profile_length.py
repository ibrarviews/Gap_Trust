"""Supplementary Figure S3 - Effect of profile length on the recovery curves.

R against gap number G for the shallow air-filled void (roof 0.75 m,
r = 0.40 m) on the full 4.0 m profile and on a profile cut to the angles the
full profile subtends at the deep (2.50 m) void, at 250 and 400 MHz.
Processing as in test (a): void minus no-void difference, no mean removal.

Input : results/data/test_profile_length.csv
Output: Figures/supplementary/FigS3_profile_length.pdf/.tif
        results/logs/figS3_profile_length.json

Run from the Codes folder:  python scripts/figS3_profile_length.py
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, style  # noqa: E402

SRC = style.CODE_ROOT / "results" / "data" / "test_profile_length.csv"
METHODS = (("zero", "Zero fill"), ("linear", "Linear interpolation"), ("pocs", "f-k POCS"))
R_FAIL = 0.5

style.apply()
C = style.COLORS
CASES = [("250MHz", "full", C["blue"], "-", "250 MHz, full profile"),
         ("250MHz", "cut", C["blue"], "--", "250 MHz, cut profile"),
         ("400MHz", "full", C["vermillion"], "-", "400 MHz, full profile"),
         ("400MHz", "cut", C["vermillion"], "--", "400 MHz, cut profile")]
with open(SRC) as fh:
    rows = list(csv.DictReader(fh))

fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL, 62 * style.MM),
                         constrained_layout=True, sharey=True)
for ax, (key, title) in zip(axes, METHODS):
    for freq, prof, color, ls, label in CASES:
        sub = [x for x in rows if x["freq"] == freq and x["profile"] == prof
               and x["method"] == key]
        ax.plot([float(x["G"]) for x in sub], [float(x["R"]) for x in sub],
                color=color, ls=ls, lw=1.0, label=label)
    ax.axhline(R_FAIL, color="0.45", lw=0.6, ls=":")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Gap number G")
axes[0].set_ylabel("Recovery ratio R")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="outside upper center", ncol=4, frameon=False)
for ax, p in zip(axes, "abc"):
    style.panel_label(ax, p)
pdf, tif = style.save(fig, "FigS3_profile_length", subdir="supplementary")
log = {"figure": "FigS3_profile_length", "script": "scripts/figS3_profile_length.py",
       "input": str(SRC.relative_to(style.CODE_ROOT)), "gaptrust_version": __version__,
       "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                   str(tif.relative_to(style.FIG_ROOT.parent))]}
(style.CODE_ROOT / "results" / "logs" / "figS3_profile_length.json").write_text(json.dumps(log, indent=2))
print(pdf)
