#!/usr/bin/env python
"""Fig. 7 - off-target gap control (ledger 14v).

(a) Great cave line 14 with the target box and the off-target gap band.
(b) Zero-fill recovery ratio against gap offset; the on-target value at offset 0 is
    the W50 crossing of Fig. 6.
(c) The same for linear interpolation.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from gaptrust import processing, style  # noqa: E402

KARST = {"V": 0.106, "DX": 0.05, "L": 8.0, "DT": 110.86392974853516 / 1023.0}
R_FAIL = 0.5

CSV = CODE / "results" / "data" / "field_R_offtarget.csv"
LOG = CODE / "results" / "logs" / "field_control_offtarget.json"
LOG_KARST = CODE / "results" / "logs" / "field_analysis_karst_guarded.json"
CACHE = CODE / "results" / "data" / "field_cache"
OUTLOG = CODE / "results" / "logs" / "fig07_gap_control.json"

PROFILES = (
    ("great_l14", "karst", "o"),
    ("small_l6", "karst", "^"),
    ("dugout_l2", "karst", "s"),
    ("aquileia_A", "aqu", "o"),
    ("aquileia_B", "aqu", "^"),
    ("aquileia_C", "aqu", "s"),
)


def karst_image():
    label = "great_l14"
    arr = np.load(CACHE / (label + ".npz"))["data"]
    data = np.ascontiguousarray(arr.T, dtype=float)
    x_old = np.linspace(0.0, KARST["L"], data.shape[1])
    x = np.arange(0.0, KARST["L"] + 1e-9, KARST["DX"])
    data = np.stack([np.interp(x, x_old, data[k]) for k in range(data.shape[0])])
    data, _ = processing.time_zero(data, KARST["DT"])
    data = processing.remove_mean(data)
    z = np.arange(0.10, 3.40, 0.01)
    return processing.envelope_z(processing.migrate(data, KARST["DT"], x,
                                                    KARST["V"], z)), z, x


def main():
    style.apply()
    rows = []
    with open(CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["offset_m"] = float(r["offset_m"])
            r["R"] = float(r["R"])
            rows.append(r)
    ctl = json.loads(LOG.read_text(encoding="utf-8"))["predictions"]
    li = json.loads(LOG_KARST.read_text(encoding="utf-8"))["lines"]["great_l14"]

    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL, 62 * style.MM),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0],
                                          "wspace": 0.32})

    ax = axes[0]
    env, z, x = karst_image()
    zi = z <= 2.20
    env = env[zi] / np.percentile(env[(z >= 0.45) & (z <= 2.20)], 98)
    ax.imshow(env, extent=[x[0], x[-1], z[zi][-1], z[zi][0]], cmap="Greys",
              vmin=0.0, vmax=1.0, aspect="auto", interpolation="bilinear")
    xc, zc = li["target_x_m"], li["target_z_m"]
    W50 = li["W50_m"]["zero"]
    ax.add_patch(plt.Rectangle((xc - 0.25, zc - 0.15), 0.50, 0.30, fill=False,
                               ec=style.COLORS["vermillion"], lw=0.9, zorder=3))
    xoff = 1.5
    ax.axvspan(xc + xoff - W50 / 2.0, xc + xoff + W50 / 2.0, 0.0, 1.0, color="none",
               ec=style.COLORS["blue"], lw=0.9, ls="--", zorder=3)
    ax.annotate("on target", xy=(xc, zc - 0.16), xytext=(xc - 1.15, 1.50),
                fontsize=6.2, color=style.COLORS["vermillion"], zorder=4,
                arrowprops=dict(arrowstyle="-", color=style.COLORS["vermillion"], lw=0.7))
    ax.annotate("off target, 1.5 m", xy=(xc + xoff, 0.50),
                xytext=(xc + xoff + 0.10, 1.50), fontsize=6.2,
                color=style.COLORS["blue"], zorder=4,
                arrowprops=dict(arrowstyle="-", color=style.COLORS["blue"], lw=0.7))
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Great cave, line 14")
    ax.tick_params(labelsize=6.4)
    style.panel_label(ax, "a", x=-0.08)

    for k, (m, mtitle) in enumerate([("zero", "Zero fill"), ("linear", "Linear")]):
        ax = axes[k + 1]
        for label, site, marker in PROFILES:
            col = style.COLORS["blue"] if site == "karst" else style.COLORS["vermillion"]
            ls = "-" if site == "karst" else "--"
            sub = sorted([r for r in rows if r["line"] == label
                          and r["method"] == m and r["offset_m"] != 0.0],
                         key=lambda r: r["offset_m"])
            if not sub:
                continue
            ax.plot([r["offset_m"] for r in sub], [r["R"] for r in sub], ls,
                    marker=marker, ms=3.0, lw=0.7, color=col,
                    mfc=col if site == "karst" else "none", mew=0.8,
                    label=("Karst" if site == "karst" else "Aquileia") if label in
                    ("great_l14", "aquileia_A") else None)
        ax.axhline(R_FAIL, color=style.COLORS["black"], lw=0.7, ls="--")
        ax.set_xlim(-2.25, 2.25)
        ax.set_ylim(0.0, 1.35)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.text(0.03, 0.045, "same gap at the target: R < 0.5", transform=ax.transAxes,
                fontsize=6.2, color=style.COLORS["black"], va="bottom", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        ax.set_title(mtitle)
        ax.set_xlabel("Gap offset from target (m)")
        if k == 0:
            ax.set_ylabel("Recovery ratio R")
            ax.legend(loc="center right", frameon=False, handlelength=1.8,
                      fontsize=6.2, borderpad=0.2, labelspacing=0.25)
        ax.tick_params(labelsize=6.4)
        style.panel_label(ax, "bc"[k], x=-0.09)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.175)
    pdf, tif = style.save(fig, "Fig07_gap_control", subdir="main")

    OUTLOG.write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": str(CSV.relative_to(CODE)),
        "predictions": ctl,
        "outputs": [str(pdf), str(tif)],
        "caption": ("Off-target gap control. (a) Great cave line 14 with the target "
                    "box and the band removed in the control, 1.5 m to the side. "
                    "(b), (c) Recovery ratio R against the offset of a gap of the "
                    "same width as the W50 of each profile, for zero fill and linear "
                    "interpolation; the value at zero offset is the W50 crossing of "
                    "Fig. 6. Aperture loss at the target is what drives the failure, "
                    "since an equal gap away from it leaves R near unity."),
    }, indent=2), encoding="utf-8")
    print("written:", pdf)
    print("written:", tif)


if __name__ == "__main__":
    main()