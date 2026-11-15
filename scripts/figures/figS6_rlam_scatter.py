"""Exploratory figure S6: failure width and gap number against r/lambda.

W50 and G50 for all 18 air-filled 250 MHz models: the twelve new models of the
r/lambda test (radii 0.10, 0.21, 0.28, 0.34 m) and the six already-simulated
models of section 12 (radii 0.15, 0.40 m), all at roof depths 0.75, 1.50 and
2.50 m. New and reused models are marked differently.

This figure is exploratory (ledger 14e). It does not enter the decision rule of
ledger 14d; the rule is applied to the twelve new models only, in
scripts/rlam_analysis.py.

Outputs: Figures/supplementary/FigS6_rlam_scatter.pdf and .tif (600 dpi, LZW),
         results/logs/figS6_rlam_scatter.json

Run from the Codes folder (interpreter with a working Pillow LZW codec):
    python scripts/figS6_rlam_scatter.py
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

DATA = style.CODE_ROOT / "results" / "data"
NEW_SUMMARY = DATA / "rlam_summary.json"
REUSED = DATA / "failure_points_recovered.csv"
LOG = style.CODE_ROOT / "results" / "logs" / "figS6_rlam_scatter.json"
R_FAIL = 0.5
LAMBDA_M = 0.106 / 0.25
DEPTHS = ((0.75, "0.75 m"), (1.50, "1.50 m"), (2.50, "2.50 m"))
NEW_RADII = (0.10, 0.21, 0.28, 0.34)
REUSED_RADII = (0.15, 0.40)
PANELS = (("W50_m", "linear", "a"), ("W50_m", "pocs", "b"),
          ("G50", "linear", "c"), ("G50", "pocs", "d"))

summary = json.loads(NEW_SUMMARY.read_text(encoding="utf-8"))
new_pts = {}
for name, case in summary["cases"].items():
    for method in ("linear", "pocs"):
        new_pts[(case["radius_m"], case["roof_m"], method)] = (
            case["W50_m"][method], case["G50"][method])

with open(REUSED, encoding="utf-8", newline="") as fh:
    table = list(csv.DictReader(fh))
reused_pts = {}
for row in table:
    if row["freq"] != "250MHz" or row["fill"] != "air":
        continue
    radius = float(row["radius_m"])
    if abs(radius - 0.15) > 1e-9 and abs(radius - 0.40) > 1e-9:
        continue
    if row["censored"] == "yes" or row["method"] not in ("linear", "pocs"):
        continue
    reused_pts[(radius, float(row["roof_m"]), row["method"])] = (
        float(row["W50_m"]), float(row["G50"]))

if len(new_pts) != 24 or len(reused_pts) != 12:
    raise SystemExit("unexpected point counts: %d new, %d reused" % (len(new_pts), len(reused_pts)))

style.apply()
C = style.COLORS
DEPTH_COLOR = ((0.75, C["black"]), (1.50, C["blue"]), (2.50, C["vermillion"]))
MODE = {("W50_m", "linear"): "W50 (m)", ("W50_m", "pocs"): "W50 (m)",
        ("G50", "linear"): "G50", ("G50", "pocs"): "G50"}
TITLE = {("W50_m", "linear"): "W50, linear", ("W50_m", "pocs"): "W50, f-k POCS",
         ("G50", "linear"): "G50, linear", ("G50", "pocs"): "G50, f-k POCS"}

fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL, 110 * style.MM),
                         constrained_layout=True)
for ax, (field, method, label) in zip(axes.ravel(), PANELS):
    for roof, color in DEPTH_COLOR:
        nx, ny, rx, ry = [], [], [], []
        for radius in NEW_RADII:
            val = new_pts.get((radius, roof, method))
            if val is not None and val[0 if field == "W50_m" else 1] is not None:
                nx.append(radius / LAMBDA_M)
                ny.append(val[0 if field == "W50_m" else 1])
        for radius in REUSED_RADII:
            val = reused_pts.get((radius, roof, method))
            if val is not None:
                rx.append(radius / LAMBDA_M)
                ry.append(val[0 if field == "W50_m" else 1])
        ax.plot(nx, ny, ls="none", marker="o", ms=4.2, mfc=color, mec="0.25", mew=0.5)
        ax.plot(rx, ry, ls="none", marker="o", ms=4.2, mfc="none", mec=color, mew=1.0)
    ax.axvline(0.5, color="0.45", lw=0.6, ls=":")
    if field == "G50":
        ax.axhline(1.0, color="0.45", lw=0.6, ls=":")
    ax.set_title(TITLE[(field, method)], fontsize=8)
    ax.set_ylabel(MODE[(field, method)])
    style.panel_label(ax, label)
    if field == "G50":
        ax.set_ylim(0.7, 1.5)
    else:
        ax.set_ylim(0.5, 2.2)
    ax.set_xlim(0.15, 1.05)

for ax in axes[1]:
    ax.set_xlabel("r/\u03bb")
axes[0, 0].text(0.52, 0.03, "r/\u03bb = 0.5", transform=axes[0, 0].transAxes,
                fontsize=6.3, color="0.45", ha="left", va="bottom")

handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=color, mec="0.25", mew=0.5,
                  label=label)
           for (_, color), (_, label) in zip(DEPTH_COLOR, DEPTHS)]
handles += [Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none", mec="0.55", mew=1.0,
                   label="section 12 models")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
           title="Colour: roof depth; filled: new models; open: reused models")

pdf, tif = style.save(fig, "FigS6_rlam_scatter", subdir="supplementary")

log = {
    "figure": "FigS6_rlam_scatter",
    "script": "scripts/figS6_rlam_scatter.py",
    "status": "exploratory; does not enter the decision rule of ledger 14d",
    "inputs": {str(NEW_SUMMARY.relative_to(style.CODE_ROOT)): hashlib.sha256(NEW_SUMMARY.read_bytes()).hexdigest(),
               str(REUSED.relative_to(style.CODE_ROOT)): hashlib.sha256(REUSED.read_bytes()).hexdigest()},
    "lambda_m": LAMBDA_M,
    "R_fail": R_FAIL,
    "points": {"new_models": len(new_pts), "reused_models": len(reused_pts),
               "new_radii_m": list(NEW_RADII), "reused_radii_m": list(REUSED_RADII),
               "roof_depths_m": [d for d, _ in DEPTHS]},
    "panels": [{"metric": f, "method": m, "panel": p} for f, m, p in PANELS],
    "gaptrust_version": __version__,
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                str(tif.relative_to(style.FIG_ROOT.parent))],
}
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
print(pdf)
print(tif)