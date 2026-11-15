"""Supplementary Figure S4 (rebuilt) - Recovery against gap number, full model set.

The original figS4_full_curves.py is not on disk. This script rebuilds Fig. S4
from the recovered full-set table, following the ledger caption for Fig. S4:
panels (a, b) linear interpolation and (c, d) f-k POCS, with void radius
(a, c) 0.15 m and (b, d) 0.40 m. Colour gives frequency, line style roof depth,
and air- and clay-filled voids share a style.

Input  : results/data/full_R_recovered.csv
Output : Figures/supplementary/FigS4_full_curves_recovered.pdf/.tif
         results/logs/figS4_full_curves_recovered.json

Run from the Codes folder:  python scripts/figS4_full_curves_recovered.py
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, style  # noqa: E402

SRC = style.CODE_ROOT / "results" / "data" / "full_R_recovered.csv"
LOG = style.CODE_ROOT / "results" / "logs" / "figS4_full_curves_recovered.json"
R_FAIL = 0.5
PANELS = (("linear", 0.15, "a", "Linear interpolation"),
          ("linear", 0.40, "b", "Linear interpolation"),
          ("pocs", 0.15, "c", "f-k POCS"),
          ("pocs", 0.40, "d", "f-k POCS"))
ROOFS = ((0.75, "-"), (1.50, "--"), (2.50, "-."))

style.apply()
C = style.COLORS
FREQ_STYLE = (("250MHz", C["blue"]), ("400MHz", C["vermillion"]))

with open(SRC, encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL, 110 * style.MM),
                         constrained_layout=True, sharex=True, sharey=True)

for ax, (method, radius, label, title) in zip(axes.ravel(), PANELS):
    for freq, color in FREQ_STYLE:
        for roof, ls in ROOFS:
            for fill in ("air", "clay"):
                sub = [r for r in rows if r["freq"] == freq and r["method"] == method
                       and abs(float(r["radius_m"]) - radius) < 1e-9
                       and abs(float(r["roof_m"]) - roof) < 1e-9 and r["fill"] == fill]
                ax.plot([float(r["G"]) for r in sub], [float(r["R"]) for r in sub],
                        color=color, ls=ls, lw=0.9)
    ax.axhline(R_FAIL, color="0.45", lw=0.6, ls=":")
    ax.axvline(1.0, color="0.45", lw=0.6, ls=":")
    ax.set_xlim(0, 3.9)
    ax.set_ylim(0, 1.4)
    ax.set_title(f"{title}, r = {radius:.2f} m", fontsize=8)
    style.panel_label(ax, label)

for ax in axes[1, :]:
    ax.set_xlabel("Gap number G")
for ax in axes[:, 0]:
    ax.set_ylabel("Recovery ratio R")

handles = [Line2D([], [], color=c, lw=1.0, label=f"{f[:3]} MHz") for f, c in FREQ_STYLE]
handles += [Line2D([], [], color="0.25", lw=1.0, ls=ls, label=f"roof {r:.2f} m")
            for r, ls in ROOFS]
fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False,
           title="Colour: frequency; line style: roof depth")

pdf, tif = style.save(fig, "FigS4_full_curves_recovered", subdir="supplementary")

log = {
    "figure": "FigS4_full_curves_recovered",
    "script": "scripts/figS4_full_curves_recovered.py",
    "status": "rebuilt from the recovered full-set analysis; original script absent",
    "input": str(SRC.relative_to(style.CODE_ROOT)),
    "input_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "input_rows": len(rows),
    "panels": [{"panel": p[2], "method": p[0], "radius_m": p[1]} for p in PANELS],
    "curves_per_panel": 2 * len(ROOFS) * 2,
    "G_max_in_data": round(max(float(r["G"]) for r in rows), 4),
    "R_fail": R_FAIL,
    "gaptrust_version": __version__,
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                str(tif.relative_to(style.FIG_ROOT.parent))],
}
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG.write_text(json.dumps(log, indent=2))
print(pdf)
print(tif)