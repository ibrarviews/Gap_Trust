"""Supplementary Figure S5 (rebuilt) - Scatter of the failure point.

The original figS5_failure_scatter.py is not on disk. This script rebuilds
Fig. S5 from the recovered full-set table. Failure points (W50, G50) are
recomputed from full_R_recovered.csv with the frozen rule of
pilot_analysis.first_failure (smallest gap width with R < 0.5, in trace order)
and cross-checked against the recovered failure-point table before anything is
plotted. Both axes are divided by their mean over the uncensored models of that
radius, as stated in the ledger caption.

Inputs : results/data/full_R_recovered.csv
         results/data/failure_points_recovered.csv   (cross-check only)
Output : Figures/supplementary/FigS5_failure_scatter_recovered.pdf/.tif
         results/logs/figS5_failure_scatter_recovered.json

Run from the Codes folder:  python scripts/figS5_failure_scatter_recovered.py
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
SRC = DATA / "full_R_recovered.csv"
CHECK = DATA / "failure_points_recovered.csv"
LOG = style.CODE_ROOT / "results" / "logs" / "figS5_failure_scatter_recovered.json"
R_FAIL = 0.5
METHODS = (("zero", "Zero fill", "o"), ("linear", "Linear interpolation", "s"),
           ("pocs", "f-k POCS", "^"))
RADII = ((0.15, "a"), (0.40, "b"))


def cv(values):
    """Population coefficient of variation, exactly as pilot_analysis.cv."""
    n = len(values)
    if n < 2:
        return float("nan")
    m = sum(values) / n
    return (sum((v - m) ** 2 for v in values) / n) ** 0.5 / m


with open(SRC, encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

points = {}
for freq in ("250MHz", "400MHz"):
    for roof in (0.75, 1.50, 2.50):
        for radius in (0.15, 0.40):
            for fill in ("air", "clay"):
                for method, _, _ in METHODS:
                    sub = sorted(
                        (r for r in rows if r["freq"] == freq and r["method"] == method
                         and abs(float(r["roof_m"]) - roof) < 1e-9
                         and abs(float(r["radius_m"]) - radius) < 1e-9 and r["fill"] == fill),
                        key=lambda r: int(r["gap_traces"]))
                    W50 = G50 = None
                    for r in sub:
                        if float(r["R"]) < R_FAIL:
                            W50, G50 = float(r["W_m"]), float(r["G"])
                            break
                    points[(freq, roof, radius, fill, method)] = (W50, G50)

with open(CHECK, encoding="utf-8", newline="") as fh:
    table = list(csv.DictReader(fh))
mismatch = []
for row in table:
    key = (row["freq"], float(row["roof_m"]), float(row["radius_m"]), row["fill"], row["method"])
    W50, G50 = points[key]
    if row["censored"] == "yes":
        if W50 is not None:
            mismatch.append(key)
    elif (W50 is None or abs(W50 - float(row["W50_m"])) > 1e-9
          or abs(G50 - float(row["G50"])) > 1e-9):
        mismatch.append(key)
if mismatch:
    raise SystemExit(f"recomputed failure points disagree with {CHECK.name}: {mismatch[:4]}")

style.apply()
C = style.COLORS
FREQ_STYLE = (("250MHz", C["blue"]), ("400MHz", C["vermillion"]))
fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL, 62 * style.MM),
                         constrained_layout=True, sharex=True, sharey=True)

summary = {}
for ax, (radius, label) in zip(axes, RADII):
    unc = {k: v for k, v in points.items() if abs(k[2] - radius) < 1e-9 and v[0] is not None}
    mw = sum(v[0] for v in unc.values()) / len(unc)
    mg = sum(v[1] for v in unc.values()) / len(unc)
    for freq, color in FREQ_STYLE:
        for method, _, marker in METHODS:
            sub = [(p[0] / mw, p[1] / mg) for k, p in unc.items()
                   if k[0] == freq and k[4] == method]
            ax.plot([s[0] for s in sub], [s[1] for s in sub], ls="none", marker=marker,
                    ms=4.2, mfc=color, mec="0.25", mew=0.5)
    ax.axhline(1.0, color="0.45", lw=0.6, ls=":")
    ax.axvline(1.0, color="0.45", lw=0.6, ls=":")
    ax.set_title(f"r = {radius:.2f} m", fontsize=8)
    ax.set_xlabel("W50 / mean W50")
    style.panel_label(ax, label)

    txt, cvs = "", {}
    for method, name, _ in METHODS:
        vals = [v for k, v in unc.items() if k[4] == method]
        cw, cg = cv([v[0] for v in vals]), cv([v[1] for v in vals])
        cvs[method] = (cw, cg, len(vals))
        txt += f"{name} {cw:.2f}/{cg:.2f}\n"
    ax.text(0.03, 0.97, "CV(W50)/CV(G50)\n" + txt.strip(), transform=ax.transAxes,
            fontsize=6.3, va="top", ha="left", linespacing=1.35,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    summary[f"r{radius:.2f}"] = {"n_uncensored": len(unc), "mean_W50": mw, "mean_G50": mg,
                                 "cv": {k: {"CV_W50": v[0], "CV_G50": v[1], "n": v[2]}
                                        for k, v in cvs.items()}}

axes[0].set_ylabel("G50 / mean G50")

handles = [Line2D([], [], ls="none", marker=m, ms=4.2, mfc="0.55", mec="0.25",
                  mew=0.5, label=n) for _, n, m in METHODS]
handles += [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=c, mec="0.25",
                   mew=0.5, label=f) for f, c in FREQ_STYLE]
fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False,
           title="Marker: method; colour: frequency")

pdf, tif = style.save(fig, "FigS5_failure_scatter_recovered", subdir="supplementary")

log = {
    "figure": "FigS5_failure_scatter_recovered",
    "script": "scripts/figS5_failure_scatter_recovered.py",
    "status": "rebuilt from the recovered full-set analysis; original script absent",
    "inputs": {str(SRC.relative_to(style.CODE_ROOT)): hashlib.sha256(SRC.read_bytes()).hexdigest(),
               str(CHECK.relative_to(style.CODE_ROOT)): hashlib.sha256(CHECK.read_bytes()).hexdigest()},
    "R_fail": R_FAIL,
    "cross_check": "recomputed W50/G50 agree with failure_points_recovered.csv, 0 mismatches",
    "normalisation": summary,
    "gaptrust_version": __version__,
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                str(tif.relative_to(style.FIG_ROOT.parent))],
}
LOG.parent.mkdir(parents=True, exist_ok=True)
LOG.write_text(json.dumps(log, indent=2))
print(pdf)
print(tif)
print("r0.40 CV", json.dumps(summary["r0.40"]["cv"]))