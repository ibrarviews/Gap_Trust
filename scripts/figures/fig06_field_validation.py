#!/usr/bin/env python
"""Fig. 6 - field validation of GapTrust, six measured profiles.

Rows (a)-(c): three karst lines (Great cave L14, Small cave L6, Dugout L2).
Rows (d)-(f): three Aquileia profiles (PROF3 in 250EW, PROF10 in 2, prof10 in 3).
Rows (g)-(i): recovery ratio R against gap number G for zero fill, linear and
POCS. Karst lines are drawn filled in blue, Aquileia profiles open and dashed in
vermillion; marker shape identifies the profile within each site.

Curves come from field_R_guarded.csv and field_R_aquileia.csv; sections are
recomputed here with the frozen chain (time_zero, remove_mean, migrate,
envelope_z). Display gain only, no change to any measured value.
"""

import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from gaptrust import processing, style  # noqa: E402

KARST = {"V": 0.106, "DX": 0.05, "L": 8.0, "DT": 110.86392974853516 / 1023.0,
         "Z_TOP": 2.20}
AQU = {"V": 0.10, "DX": 0.05, "Z_TOP": 2.80}
R_FAIL = 0.5
Y_TOP = 2.2
METHODS = [("zero", "Zero fill"), ("linear", "Linear"), ("pocs", "POCS")]

PROFILES = [
    ("karst", "great_l14", "Great cave, line 14", "o"),
    ("karst", "small_l6", "Small cave, line 6", "^"),
    ("karst", "dugout_l2", "Dugout, line 2", "s"),
    ("aqu", "aquileia_A", "Aquileia, 250EW PROF3", "o"),
    ("aqu", "aquileia_B", "Aquileia, PROF10 (2)", "^"),
    ("aqu", "aquileia_C", "Aquileia, prof10 (3)", "s"),
]

KARST_CACHE = CODE / "results" / "data" / "field_cache"
CSV_KARST = CODE / "results" / "data" / "field_R_guarded.csv"
CSV_AQU = CODE / "results" / "data" / "field_R_aquileia_full.csv"
LOG_KARST = CODE / "results" / "logs" / "field_analysis_karst_guarded.json"
LOG_AQU = CODE / "results" / "logs" / "field_analysis_aquileia_full.json"
OUTLOG = CODE / "results" / "logs" / "fig06_field_validation.json"
ROOT_AQU = CODE.parent / "Aquileia Data"


def sgy_reader():
    spec = importlib.util.spec_from_file_location(
        "field_inventory", CODE / "scripts" / "field_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sgy_read


def karst_image(label):
    arr = np.load(KARST_CACHE / ("%s.npz" % label))["data"]
    data = np.ascontiguousarray(arr.T, dtype=float)
    x_old = np.linspace(0.0, KARST["L"], data.shape[1])
    x = np.arange(0.0, KARST["L"] + 1e-9, KARST["DX"])
    data = np.stack([np.interp(x, x_old, data[k]) for k in range(data.shape[0])])
    data, _ = processing.time_zero(data, KARST["DT"])
    data = processing.remove_mean(data)
    z = np.arange(0.10, 3.40, 0.01)
    return processing.envelope_z(processing.migrate(data, KARST["DT"], x,
                                                    KARST["V"], z)), z, x


def aqu_image(label, sgy_read):
    folder, name = {"aquileia_A": ("1/250EW", "PROF3.SGY"),
                    "aquileia_B": ("2", "PROF10.sgy"),
                    "aquileia_C": ("3", "prof10.sgy")}[label]
    hit = [p for p in ROOT_AQU.joinpath(*folder.split("/")).rglob("*")
           if p.is_file() and p.name.lower() == name.lower()][0]
    h = sgy_read(hit)
    data = h["traces"].T.astype(float)
    dt = h["dt"] / 1000.0
    x = np.arange(data.shape[1]) * AQU["DX"]
    data, _ = processing.time_zero(data, dt)
    data = processing.remove_mean(data)
    z = np.arange(0.10, 3.40, 0.01)
    return processing.envelope_z(processing.migrate(data, dt, x, AQU["V"], z)), z, x


def box_peak(env, x, z, xc, zc, hx, hz):
    bx = np.abs(x - xc) <= hx
    bz = np.abs(z - zc) <= hz
    return env[np.ix_(bz, bx)].max()


def main():
    style.apply()
    sgy_read = sgy_reader()
    infos = {"karst": json.loads(LOG_KARST.read_text(encoding="utf-8"))["lines"],
             "aqu": json.loads(LOG_AQU.read_text(encoding="utf-8"))["lines"]}
    rows = []
    for path in (CSV_KARST, CSV_AQU):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["gap_traces"] = int(r["gap_traces"])
                for k in ("W_m", "G", "R"):
                    r[k] = float(r[k])
                rows.append(r)

    fig, axes = plt.subplots(3, 3, figsize=(style.DOUBLE_COL, 190 * style.MM),
                             gridspec_kw={"height_ratios": [1.05, 1.05, 1.0],
                                          "hspace": 0.42, "wspace": 0.30})

    im = None
    for j, (site, label, title, marker) in enumerate(PROFILES):
        row = 0 if site == "karst" else 1
        ax = axes[row, j % 3]
        if site == "karst":
            env, z, x = karst_image(label)
            z_top = KARST["Z_TOP"]
        else:
            env, z, x = aqu_image(label, sgy_read)
            z_top = AQU["Z_TOP"]
        zi = z <= z_top
        below = env[(z >= 0.45) & (z <= z_top)]
        env = env[zi] / np.percentile(below, 98)
        ax.imshow(env, extent=[x[0], x[-1], z[zi][-1], z[zi][0]], cmap="Greys",
                  vmin=0.0, vmax=1.0, aspect="auto", interpolation="bilinear")

        li = infos[site][label]
        xc, zc, W50, G50 = li["target_x_m"], li["target_z_m"], \
            li["W50_m"]["zero"], li["G50"]["zero"]
        ax.add_patch(plt.Rectangle((xc - 0.25, zc - 0.15), 0.50, 0.30, fill=False,
                                   ec=style.COLORS["vermillion"], lw=0.9, zorder=3))
        yb = min(zc + 0.28, z_top - 0.05)
        ax.annotate("", xy=(xc - W50 / 2.0, yb), xytext=(xc + W50 / 2.0, yb),
                    arrowprops=dict(arrowstyle="<|-|>", color=style.COLORS["vermillion"],
                                    lw=1.0, shrinkA=0, shrinkB=0), zorder=3)
        txt = "W$_{50}$ = %.2f m, G$_{50}$ = %.2f" % (W50, G50)
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=6.2,
                color=style.COLORS["vermillion"], va="bottom", ha="left", zorder=4,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        ax.set_title(title, pad=3)
        ax.set_xlabel("Distance (m)")
        if j % 3 == 0:
            ax.set_ylabel("Depth (m)")
        ax.tick_params(labelsize=6.4)
        style.panel_label(ax, "abcdef"[j], x=-0.06)

    for j, (m, mtitle) in enumerate(METHODS):
        ax = axes[2, j]
        for site, label, title, marker in PROFILES:
            col = style.COLORS["blue"] if site == "karst" else style.COLORS["vermillion"]
            ls = "-" if site == "karst" else "--"
            sub = [r for r in rows if r["line"] == label and r["method"] == m]
            ax.plot([r["G"] for r in sub], [r["R"] for r in sub], ls, marker=marker,
                    ms=2.6, lw=0.8, color=col,
                    mfc=col if site == "karst" else "none", mew=0.8,
                    label=("Karst" if site == "karst" else "Aquileia") if label in
                    ("great_l14", "aquileia_A") else None)
            g50 = infos[site][label]["G50"][m]
            if g50 is not None:
                ax.plot([g50], [R_FAIL], marker, ms=4.2, mfc="none",
                        mec=col, mew=0.9)
        ax.axhline(R_FAIL, color=style.COLORS["black"], lw=0.7, ls="--")
        ax.axvspan(0.70, 1.70, color=style.COLORS["grey"], alpha=0.14, lw=0)
        ax.set_xlim(0.0, 1.8)
        ax.set_ylim(0.0, Y_TOP)
        mx = max(r["R"] for r in rows if r["method"] == m)
        if mx > Y_TOP:
            ax.text(0.03, 0.97, "off scale: max R = %.1f" % mx, transform=ax.transAxes,
                    fontsize=6.2, va="top", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        ax.set_title(mtitle, pad=3)
        ax.set_xlabel("Gap number G = W / D")
        if j == 0:
            ax.set_ylabel("Recovery ratio R")
            ax.legend(loc="upper right", frameon=False, handlelength=1.8, fontsize=6.2,
                      borderpad=0.2, labelspacing=0.25)
        ax.tick_params(labelsize=6.4)
        style.panel_label(ax, "ghi"[j], x=-0.06)

    fig.subplots_adjust(left=0.062, right=0.985, top=0.955, bottom=0.065)
    pdf, tif = style.save(fig, "Fig06_field_validation", subdir="main")

    OUTLOG.write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": [str(CSV_KARST.relative_to(CODE)), str(CSV_AQU.relative_to(CODE))],
        "targets": {k: {kk: {"x": v[kk]["target_x_m"], "z": v[kk]["target_z_m"],
                            "D": v[kk]["D_m"], "W50": v[kk]["W50_m"]["zero"],
                            "G50": v[kk]["G50"]["zero"]} for kk in v}
                    for k, v in infos.items()},
        "outputs": [str(pdf), str(tif)],
        "caption": ("Field validation of GapTrust on six measured profiles. "
                    "(a)-(c) Karst lines of the Guilin sites, 400 MHz antenna; "
                    "(d)-(f) Aquileia profiles, 250 MHz (Zhao et al., 2013). Boxes "
                    "mark the target (+/-0.25 m, +/-0.15 m), arrows the gap width at "
                    "W50; display gain only. (g)-(i) Recovery ratio R versus gap "
                    "number G for zero fill, linear interpolation and POCS; dashed "
                    "line R = 0.5, shaded band the G50 range of the models "
                    "(0.70-1.70), open symbols the G50 of each method. POCS raises R "
                    "above unity, so panels are annotated rather than rescaled."),
    }, indent=2), encoding="utf-8")
    print("written:", pdf)
    print("written:", tif)


if __name__ == "__main__":
    main()