"""Main Figure 3 - failure against gap width, all 24 models.

Fig. 3 is the companion of Fig. 4: the same models and the same curves, read
against gap width W in metres instead of gap number G. Fig. 3 shows where
recovery fails in metres; Fig. 4 reads the same failure points in G. Plotted
together they are the paper's argument: the spread exists in W, and G compresses
it without collapsing it.

Panels follow the layout of Fig. S4: (a, b) linear interpolation and (c, d)
f-k POCS, columns for void radius 0.15 and 0.40 m. Colour gives frequency,
line style roof depth, and air- and clay-filled voids share a style. A marker
on the R = 0.5 line marks the failure width of each curve, filled for air and
open for clay. Curves that never cross R = 0.5 within 2.43 m carry no marker;
those are the censored cases of ledger section 12.

Input  : results/data/full_R_recovered.csv
Output : Figures/main/Fig03_failure_vs_width.pdf/.tif
         results/logs/fig03_failure_vs_width.json

Run from the Codes folder with the gprmax environment interpreter, which writes
LZW TIFF; the base interpreter aborts on LZW TIFF (ledger 14h):

  C:/Users/IBRAR IQBAL/anaconda3/envs/gprmax/python.exe scripts/fig03_failure_vs_width.py
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
LOG = style.CODE_ROOT / "results" / "logs" / "fig03_failure_vs_width.json"
R_FAIL = 0.5
PANELS = (("linear", 0.15, "a", "Linear interpolation"),
          ("linear", 0.40, "b", "Linear interpolation"),
          ("pocs", 0.15, "c", "f-k POCS"),
          ("pocs", 0.40, "d", "f-k POCS"))
ROOFS = ((0.75, "-"), (1.50, "--"), (2.50, "-."))
FILL_MARKER = {"air": "filled", "clay": "open"}

style.apply()
C = style.COLORS
FREQ_STYLE = (("250MHz", C["blue"]), ("400MHz", C["vermillion"]))

with open(SRC, encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL, 110 * style.MM),
                         constrained_layout=True, sharex=True, sharey=True)

w50_all = []
censored = []
for ax, (method, radius, label, title) in zip(axes.ravel(), PANELS):
    for freq, color in FREQ_STYLE:
        for roof, ls in ROOFS:
            for fill in ("air", "clay"):
                sub = [r for r in rows if r["freq"] == freq and r["method"] == method
                       and abs(float(r["radius_m"]) - radius) < 1e-9
                       and abs(float(r["roof_m"]) - roof) < 1e-9 and r["fill"] == fill]
                sub.sort(key=lambda r: float(r["W_m"]))
                ax.plot([float(r["W_m"]) for r in sub], [float(r["R"]) for r in sub],
                        color=color, ls=ls, lw=0.9)
                fails = [r for r in sub if float(r["R"]) < R_FAIL]
                if fails:
                    w50 = float(fails[0]["W_m"])
                    w50_all.append(w50)
                    ax.plot([w50], [R_FAIL], marker="o", ms=3.2, mew=0.6,
                            color=color, ls="none",
                            mfc=color if fill == "air" else "none")
                else:
                    censored.append({"panel": label, "freq": freq,
                                     "roof_m": roof, "radius_m": radius,
                                     "fill": fill, "method": method})
    ax.axhline(R_FAIL, color="0.45", lw=0.6, ls=":")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.4)
    ax.set_title("%s, r = %.2f m" % (title, radius), fontsize=8)
    style.panel_label(ax, label)

for ax in axes[1, :]:
    ax.set_xlabel("Gap width W (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("Recovery ratio R")

handles = [Line2D([], [], color=c, lw=1.0, label="%s MHz" % f[:3]) for f, c in FREQ_STYLE]
handles += [Line2D([], [], color="0.25", lw=1.0, ls=ls, label="roof %.2f m" % r)
            for r, ls in ROOFS]
handles += [Line2D([], [], color="0.45", marker="o", ms=3.2, ls="none",
                   mfc="0.45", label="air, W50"),
            Line2D([], [], color="0.45", marker="o", ms=3.2, ls="none",
                   mfc="none", label="clay, W50")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
           title="Colour: frequency; line style: roof depth; marker: fill at R = 0.5")

pdf, tif = style.save(fig, "Fig03_failure_vs_width", subdir="main")

log = {
    "figure": "Fig03_failure_vs_width",
    "role": "main Fig. 3; width-domain companion of Fig. 4",
    "script": "scripts/fig03_failure_vs_width.py",
    "input": str(SRC.relative_to(style.CODE_ROOT)),
    "input_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "input_rows": len(rows),
    "panels": [{"panel": p[2], "method": p[0], "radius_m": p[1]} for p in PANELS],
    "curves_per_panel": 2 * len(ROOFS) * 2,
    "x_axis": "gap width W_m",
    "y_axis": "recovery ratio R",
    "xlim": [0, 2.5],
    "ylim": [0, 1.4],
    "W_range_in_data": [min(float(r["W_m"]) for r in rows),
                        max(float(r["W_m"]) for r in rows)],
    "R_range_in_data": [min(float(r["R"]) for r in rows),
                        max(float(r["R"]) for r in rows)],
    "W50_range": [min(w50_all), max(w50_all)] if w50_all else None,
    "n_W50_markers": len(w50_all),
    "censored_curves": censored,
    "R_fail": R_FAIL,
    "gaptrust_version": __version__,
    "interpreter": sys.executable,
    "python": sys.version.split()[0],
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                str(tif.relative_to(style.FIG_ROOT.parent))],
}
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG.write_text(json.dumps(log, indent=2))
print(pdf)
print(tif)
print("W50 markers:", len(w50_all), "range", log["W50_range"])
print("censored curves:", len(censored))