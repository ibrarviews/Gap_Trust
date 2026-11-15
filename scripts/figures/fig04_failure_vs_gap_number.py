"""Main Figure 4 - the failure points read in gap number, and the H1 test.

Fig. 3 shows where recovery fails in metres; Fig. 4 reads the same 24 models in
gap numbers and states the H1 verdict.

Panel (a) gives the failure gap number G50 of every model, grouped by void radius
and reconstruction method, coloured by frequency, filled for air-filled and open
for clay-filled voids, against G = 1.
Panel (b) is the H1 criterion itself, group by group: the coefficient of
variation of W50 (open square), the value CV(G50) had to fall to in order to
pass (half of it, open circle), and the value it actually reaches (filled
circle). A group passes only if the filled circle sits on or below the open
circle.

Every quantity is recomputed here from the recovered full-set table and checked
against the recorded H1 statistics; the script aborts if they disagree.

Input  : results/data/full_R_recovered.csv, results/data/full_summary_recovered.json
Output : Figures/main/Fig04_failure_vs_gap_number.pdf/.tif
         results/logs/fig04_failure_vs_gap_number.json

Run from the Codes folder with the gprmax environment interpreter (LZW TIFF;
the base interpreter aborts, ledger 14h):

  C:/Users/IBRAR IQBAL/anaconda3/envs/gprmax/python.exe scripts/fig04_failure_vs_gap_number.py
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, style  # noqa: E402

CODE = style.CODE_ROOT
SRC = CODE / "results" / "data" / "full_R_recovered.csv"
SUM = CODE / "results" / "data" / "full_summary_recovered.json"
LOG = CODE / "results" / "logs" / "fig04_failure_vs_gap_number.json"
R_FAIL = 0.5
METHODS = (("zero", "Zero fill"), ("linear", "Linear"), ("pocs", "f-k POCS"))
RADII = (0.15, 0.40)
FREQS = ("250MHz", "400MHz")
FILLS = ("air", "clay")
SHORT = {"zero": "zero", "linear": "linear", "pocs": "POCS"}

style.apply()
C = style.COLORS
FREQ_COLOR = {"250MHz": C["blue"], "400MHz": C["vermillion"]}
OFFSET = {("250MHz", "air"): -0.27, ("250MHz", "clay"): -0.09,
          ("400MHz", "air"): 0.09, ("400MHz", "clay"): 0.27}


def cv(values):
    a = np.asarray([v for v in values if v is not None], float)
    return float(a.std() / a.mean()), len(a)


with open(SRC, encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

groups = []
for radius in RADII:
    for method, title in METHODS:
        w50s, g50s = [], []
        for freq in FREQS:
            for roof in (0.75, 1.50, 2.50):
                for fill in FILLS:
                    sub = [r for r in rows if r["freq"] == freq and r["method"] == method
                           and abs(float(r["radius_m"]) - radius) < 1e-9
                           and abs(float(r["roof_m"]) - roof) < 1e-9 and r["fill"] == fill]
                    sub.sort(key=lambda r: float(r["W_m"]))
                    fails = [r for r in sub if float(r["R"]) < R_FAIL]
                    w50s.append(float(fails[0]["W_m"]) if fails else None)
                    g50s.append(float(fails[0]["G"]) if fails else None)
        groups.append({"radius_m": radius, "method": method, "title": title,
                       "W50": w50s, "G50": g50s})

cvw, cvg = [], []
for g in groups:
    w, nw = cv(g["W50"])
    v, nv = cv(g["G50"])
    g["CV_W50"], g["n_W50"] = w, nw
    g["CV_G50"], g["n_G50"] = v, nv
    g["ratio"] = v / w
    g["censored"] = sum(1 for x in g["W50"] if x is None)
    cvw.append(w)
    cvg.append(v)

# cross-check against the recorded H1 statistics (frozen analysis)
summary = json.loads(SUM.read_text(encoding="utf-8"))
bad = []
for g in groups:
    key = "r%.2f_%s" % (g["radius_m"], g["method"])
    rec = summary["tests"][key]
    if abs(rec["CV_W50"] - g["CV_W50"]) > 1e-9 or abs(rec["CV_G50"] - g["CV_G50"]) > 1e-9:
        bad.append((key, rec["CV_W50"], g["CV_W50"], rec["CV_G50"], g["CV_G50"]))
if bad:
    raise SystemExit("cross-check failed: %s" % bad)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(style.DOUBLE_COL, 85 * style.MM),
                               constrained_layout=True)
xpos = np.arange(len(groups))

for i, g in enumerate(groups):
    # points, ordered exactly as the group arrays were built
    j = 0
    for freq in FREQS:
        for roof in (0.75, 1.50, 2.50):
            for fill in FILLS:
                g50 = g["G50"][j]
                if g50 is not None:
                    axa.plot(i + OFFSET[(freq, fill)], g50, marker="o", ms=3.4,
                             mew=0.6, ls="none", color=FREQ_COLOR[freq],
                             mfc=FREQ_COLOR[freq] if fill == "air" else "none")
                j += 1
    axa.text(i, 0.30, "n = %d" % g["n_W50"], ha="center", va="bottom",
             fontsize=6.0, color="0.25")

axa.axhline(1.0, color="0.45", lw=0.6, ls=":")
axa.set_xlim(-0.5, len(groups) - 0.5)
axa.set_ylim(0.25, 1.95)
axa.set_xticks(xpos)
axa.set_xticklabels(["%.2f m\n%s" % (g["radius_m"], SHORT[g["method"]]) for g in groups],
                    fontsize=7)
axa.set_ylabel("Failure gap number G50")
axa.set_title("Failure points read in G", fontsize=8)
style.panel_label(axa, "a", x=-0.02)

for i, g in enumerate(groups):
    axb.plot([i, i], [0.5 * g["CV_W50"], g["CV_W50"]], color="0.6", lw=0.6, zorder=1)
    axb.plot(i, g["CV_W50"], marker="s", ms=4.2, mfc="none", mew=0.8,
             color="0.35", ls="none", zorder=3)
    axb.plot(i, 0.5 * g["CV_W50"], marker="o", ms=4.2, mfc="none", mew=0.8,
             color=C["green"], ls="none", zorder=3)
    axb.plot(i, g["CV_G50"], marker="o", ms=4.2, mfc="black", mew=0.8,
             color="black", ls="none", zorder=4)
    axb.text(i, 0.012, "%.2f" % g["ratio"], ha="center", va="bottom", fontsize=6.5,
             color="0.25")

axb.set_xlim(-0.5, len(groups) - 0.5)
axb.set_ylim(0, 0.40)
axb.set_xticks(xpos)
axb.set_xticklabels(["%.2f m\n%s" % (g["radius_m"], SHORT[g["method"]]) for g in groups],
                    fontsize=7)
axb.set_ylabel("Coefficient of variation")
axb.set_title("The H1 criterion, group by group", fontsize=8)
style.panel_label(axb, "b", x=-0.02)
axb.legend(handles=[
    Line2D([], [], marker="s", ms=4.2, mfc="none", mew=0.8, color="0.35", ls="none",
           label="CV(W50), measured"),
    Line2D([], [], marker="o", ms=4.2, mfc="none", mew=0.8, color=C["green"],
           ls="none", label="half of it (pass line)"),
    Line2D([], [], marker="o", ms=4.2, mfc="black", color="black", ls="none",
           label="CV(G50), measured")],
    loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.4, borderaxespad=0.2)

handles = [Line2D([], [], marker="o", ms=3.4, ls="none", color=FREQ_COLOR[f],
                  label="%s MHz" % f[:3]) for f in FREQS]
handles += [Line2D([], [], marker="o", ms=3.4, ls="none", color="0.45", mfc="0.45",
                   label="air"),
            Line2D([], [], marker="o", ms=3.4, ls="none", color="0.45", mfc="none",
                   label="clay")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
           title="(a) Colour: frequency; marker fill: void fill")

for ax in (axa, axb):
    ax.axvline(2.5, color="0.82", lw=0.6, zorder=0)

pdf, tif = style.save(fig, "Fig04_failure_vs_gap_number", subdir="main")

log = {
    "figure": "Fig04_failure_vs_gap_number",
    "role": "main Fig. 4; gap-number reading of the failure points and the H1 verdict",
    "script": "scripts/fig04_failure_vs_gap_number.py",
    "input": str(SRC.relative_to(CODE)),
    "input_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "summary_sha256": hashlib.sha256(SUM.read_bytes()).hexdigest(),
    "cross_check": "recomputed CVs match full_summary_recovered.json exactly",
    "groups": [{k: g[k] for k in ("radius_m", "method", "n_W50", "censored",
                                  "CV_W50", "CV_G50", "ratio")} for g in groups],
    "CV_G50_range": [min(cvg), max(cvg)],
    "CV_W50_range": [min(cvw), max(cvw)],
    "ratio_range": [min(g["ratio"] for g in groups), max(g["ratio"] for g in groups)],
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
for g in groups:
    print("r%.2f %-6s n=%2d censored=%d CV_W=%.4f CV_G=%.4f ratio=%.3f"
          % (g["radius_m"], g["method"], g["n_W50"], g["censored"],
             g["CV_W50"], g["CV_G50"], g["ratio"]))