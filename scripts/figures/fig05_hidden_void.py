#!/usr/bin/env python
"""Main Fig. 5: the void the gap hides, and whether that case is representative.

Case and width were fixed in ledger section 14o before this ran: the first 250 MHz
model of models.void_set() with r = 0.40 m (roof 0.75 m, air filled), at its
zero-fill W50 of 35 traces = 1.05 m.

Panel (a): complete section and the three reconstructions at that width, one
common scale normalised to the complete section, saturated above VMAX.
Panel (b): R interpolated at G = 1 for all 24 models and three methods.

Run from the Codes folder with the interpreter that can write LZW TIFF:

    C:/Users/IBRAR IQBAL/anaconda3/envs/gprmax/python.exe scripts/fig05_hidden_void.py
"""

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import models, physics, processing, style, __version__ as GT_VERSION
import pilot_analysis as pa

FREQ = "250MHz"
DEPTH_M = 0.75
RADIUS_M = 0.40
FILL = "air"
W_TRACES = 35
METHODS = pa.METHODS
R_FAIL = pa.R_FAIL
VMAX = 0.55

COL = {"250MHz": "#0072B2", "400MHz": "#D55E00"}
LABEL = {"zero": "zero fill", "linear": "linear", "pocs": "f-k POCS"}
TITLE = {"complete": "complete section", "zero": "zero fill",
         "linear": "linear", "pocs": "f-k POCS"}
OFFS = {("250MHz", "air"): -0.14, ("250MHz", "clay"): -0.05,
        ("400MHz", "air"): 0.05, ("400MHz", "clay"): 0.14}

RESULTS = style.CODE_ROOT / "results"
FULL_R = RESULTS / "data" / "full_R_recovered.csv"
LOG = RESULTS / "logs" / "fig05_hidden_void.json"
FIG = style.FIG_ROOT / "main" / "Fig05_hidden_void"
TOL = 1e-4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def environment():
    return {"python": platform.python_version(), "interpreter": sys.executable,
            "numpy": np.__version__, "matplotlib": pkg("matplotlib"),
            "pillow": pkg("pillow"), "scipy": pkg("scipy"),
            "gaptrust": GT_VERSION,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def render_case():
    name = models.model_name(models.VoidSpec(DEPTH_M, RADIUS_M, FILL))
    data, dt, x, spec = pa.load(FREQ, name)
    data, i0 = processing.time_zero(data, dt)
    roof = float(spec["void"]["roof_depth_m"])
    f_ghz = float(spec["sim"]["frequency_GHz"])
    z = np.arange(max(roof - 0.45, 0.05), roof + 0.45, 0.01)
    ntr = data.shape[1]
    centre = int(np.argmin(np.abs(x - float(spec["void"]["x_m"]))))
    dxs = float(x[1] - x[0])

    full = processing.envelope_z(processing.migrate(
        processing.remove_mean(data), dt, x, pa.V_MIG, z))
    sel_x = np.abs(x - float(spec["void"]["x_m"])) <= 0.30
    sel_z = np.abs(z - roof) <= 0.30
    sub = full[np.ix_(sel_z, sel_x)]
    iz, ix = np.unravel_index(np.argmax(sub), sub.shape)
    zc, xc = float(z[sel_z][iz]), float(x[sel_x][ix])
    ref = float(pa.box_peak(full, x, z, xc, zc, 0.25, 0.15))

    miss = processing.centred_gap(ntr, centre, W_TRACES)
    base = processing.remove_mean(data, miss)
    imgs = {"complete": full}
    R = {}
    for m in METHODS:
        filled = processing.FILLERS[m](base, miss)
        env = processing.envelope_z(processing.migrate(filled, dt, x, pa.V_MIG, z))
        imgs[m] = env
        R[m] = float(pa.box_peak(env, x, z, xc, zc, 0.25, 0.15) / ref)

    idx = np.flatnonzero(miss)
    edges = (float(x[idx[0]] - dxs / 2), float(x[idx[-1]] + dxs / 2))
    D = float(physics.aperture(roof, pa.V_MIG, f_ghz))
    info = {"model": name, "freq": FREQ, "roof_m": roof, "radius_m": RADIUS_M,
            "fill": FILL, "time_zero_samples": int(i0), "trace_spacing_m": dxs,
            "ntraces": int(ntr), "gap_traces": W_TRACES, "gap_m": W_TRACES * dxs,
            "gap_edges_m": edges, "target_x_m": xc, "target_z_m": zc, "ref": ref,
            "D_m": D, "G_at_width": W_TRACES * dxs / D}
    return imgs, R, info, x, z


def series_at_g1():
    with open(FULL_R, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    key = lambda r: (r["freq"], float(r["roof_m"]), float(r["radius_m"]),
                     r["fill"], r["method"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append((float(r["G"]), float(r["R"])))
    out = []
    for k, pts in sorted(groups.items()):
        g = [p[0] for p in pts]
        v = [p[1] for p in pts]
        if min(g) > 1.0 or max(g) < 1.0:
            raise SystemExit("series does not bracket G = 1: %s" % (k,))
        out.append({"freq": k[0], "roof_m": k[1], "radius_m": k[2], "fill": k[3],
                    "method": k[4], "R_at_G1": float(np.interp(1.0, g, v))})
    return out


def cross_check(R, info):
    with open(FULL_R, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = []
    for m in METHODS:
        hit = [r for r in rows if r["freq"] == FREQ
               and abs(float(r["roof_m"]) - info["roof_m"]) < 1e-9
               and abs(float(r["radius_m"]) - info["radius_m"]) < 1e-9
               and r["fill"] == FILL and r["method"] == m
               and int(r["gap_traces"]) == W_TRACES]
        if len(hit) != 1:
            raise SystemExit("expected one logged row, found %d for %s" % (len(hit), m))
        d = abs(float(hit[0]["R"]) - R[m])
        if d > TOL:
            bad.append((m, float(hit[0]["R"]), R[m], d))
    return bad


def main():
    for p in (LOG, Path(str(FIG) + ".pdf"), Path(str(FIG) + ".tif")):
        if p.exists():
            p.unlink()

    imgs, R, info, x, z = render_case()
    series = series_at_g1()
    bad = cross_check(R, info)
    if bad:
        raise SystemExit("cross-check failed against full_R_recovered.csv: %s" % (bad,))

    n_below = sum(1 for s in series if s["R_at_G1"] < R_FAIL)
    n_series = len(series)

    style.apply()
    fig = plt.figure(figsize=(174 * style.MM, 140 * style.MM))

    ix = (x >= info["target_x_m"] - 1.0) & (x <= info["target_x_m"] + 1.0)
    xr = (float(x[ix].min()), float(x[ix].max()))
    pos = {"complete": (0.070, 0.690), "zero": (0.490, 0.690),
           "linear": (0.070, 0.435), "pocs": (0.490, 0.435)}
    w_ax, h_ax = 0.380, 0.205

    img_last = None
    for k, (px, py) in pos.items():
        ax = fig.add_axes([px, py, w_ax, h_ax])
        im = ax.imshow(imgs[k][:, ix] / info["ref"],
                       extent=[xr[0], xr[1], float(z.max()), float(z.min())],
                       cmap="gray", vmin=0.0, vmax=VMAX, aspect="auto",
                       interpolation="nearest")
        img_last = im
        ax.set_title(TITLE[k], fontsize=7, pad=3)
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
        ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
        ax.tick_params(labelsize=6.5, length=2.0)
        if k in ("complete", "linear"):
            ax.set_ylabel("depth (m)", fontsize=7)
        else:
            ax.set_yticklabels([])
        if k == "complete":
            ax.text(0.03, 0.93, "R = 1.00", transform=ax.transAxes, ha="left",
                    va="top", fontsize=7, color="w")
            style.panel_label(ax, "a", x=-0.10, y=1.16)
        else:
            for e in info["gap_edges_m"]:
                ax.axvline(e, color="w", lw=0.6, ls=(0, (2.5, 1.5)), alpha=0.9)
            ax.text(0.03, 0.93, "R = %.2f" % R[k], transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color="w")
    fig.text(0.5, 0.912, "gap W = %.2f m (%d traces), G = %.2f"
             % (info["gap_m"], W_TRACES, info["G_at_width"]),
             ha="center", va="bottom", fontsize=7)

    cax = fig.add_axes([0.888, 0.435, 0.011, 0.460])
    cb = fig.colorbar(img_last, cax=cax, orientation="vertical")
    cb.set_label("relative envelope amplitude (saturated above %.2f)" % VMAX,
                 fontsize=6.5)
    cax.tick_params(labelsize=6, length=2.0)
    cb.set_ticks([0.0, 0.2, 0.4, VMAX])

    axb = fig.add_axes([0.105, 0.072, 0.860, 0.240])
    y_of = {"zero": 0, "linear": 1, "pocs": 2}
    for s in series:
        axb.plot(s["R_at_G1"], y_of[s["method"]] + OFFS[(s["freq"], s["fill"])],
                 marker="o", ms=2.6, lw=0,
                 mfc=COL[s["freq"]] if s["fill"] == "air" else "none",
                 mec=COL[s["freq"]], mew=0.5)
    ill = [s for s in series if s["freq"] == FREQ
           and abs(s["roof_m"] - info["roof_m"]) < 1e-9
           and abs(s["radius_m"] - info["radius_m"]) < 1e-9
           and s["fill"] == FILL and s["method"] == "zero"][0]
    axb.plot(ill["R_at_G1"], y_of["zero"] + OFFS[(FREQ, FILL)], marker="s", ms=7.0,
             mfc="none", mec="0.10", mew=0.9, zorder=6)
    axb.annotate("illustrated in (a)",
                 xy=(ill["R_at_G1"], y_of["zero"] + OFFS[(FREQ, FILL)]),
                 xytext=(ill["R_at_G1"] + 0.055, -0.42), fontsize=6.5, color="0.2",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
    axb.axvline(R_FAIL, color="0.45", lw=0.6, ls=":")
    axb.set_xlim(0.30, 0.95)
    axb.set_ylim(-0.55, 2.45)
    axb.set_yticks([0, 1, 2])
    axb.set_yticklabels([LABEL["zero"], LABEL["linear"], LABEL["pocs"]], fontsize=7)
    axb.set_xlabel("R at G = 1 (gap width equal to one aperture D)", fontsize=7)
    axb.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axb.tick_params(labelsize=6.5, length=2.5)
    axb.text(0.015, 0.045, "%d of %d series below R = 0.5 (dotted line)"
             % (n_below, n_series), transform=axb.transAxes, fontsize=6.5, color="0.2")
    style.panel_label(axb, "b", x=-0.075, y=1.19)

    handles = [Line2D([], [], marker="o", ls="", ms=3.2, mfc=COL[f], mec=COL[f],
                      label=f) for f in ("250MHz", "400MHz")]
    handles += [Line2D([], [], marker="o", ls="", ms=3.2, mfc="0.35", mec="0.35",
                       label="air"),
                Line2D([], [], marker="o", ls="", ms=3.2, mfc="none", mec="0.35",
                       label="clay")]
    axb.legend(handles=handles, ncol=4, loc="lower center", fontsize=6.5,
               bbox_to_anchor=(0.5, 1.005), frameon=False, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0.0)

    pdf, tif = style.save(fig, "Fig05_hidden_void", subdir="main")
    plt.close(fig)

    rec = {"figure": str(Path(pdf)), "tiff": str(Path(tif)),
           "script": str(Path(__file__).resolve()),
           "case": info, "R_at_width": R, "display_vmax": VMAX,
           "cross_check": {"file": str(FULL_R), "sha256": sha256_file(FULL_R),
                           "gap_traces": W_TRACES, "mismatches": 0, "tolerance": TOL},
           "series_at_G1": series, "n_series": n_series, "n_below_R_fail": n_below,
           "prediction_as_registered": "at least 60 of 72 below R = 0.5",
           "prediction_met": bool(n_below >= 60),
           "R_summary": {"min": min(s["R_at_G1"] for s in series),
                         "max": max(s["R_at_G1"] for s in series),
                         "median": float(np.median([s["R_at_G1"] for s in series]))},
           "frozen": {"V_MIG": pa.V_MIG, "R_FAIL": R_FAIL, "box_m": [0.25, 0.15],
                      "methods": list(METHODS)},
           "environment": environment(),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "outputs": {str(Path(p)): {"bytes": Path(p).stat().st_size,
                                      "sha256": sha256_file(p)} for p in (pdf, tif)}}
    LOG.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print("case            %s" % info["model"])
    print("gap             %.2f m, %d traces, G = %.2f"
          % (info["gap_m"], W_TRACES, info["G_at_width"]))
    for m in METHODS:
        print("R %-6s        %.4f" % (m, R[m]))
    print("cross-check     %d mismatches" % len(bad))
    print("R at G = 1      median %.3f, min %.3f, max %.3f"
          % (rec["R_summary"]["median"], rec["R_summary"]["min"],
             rec["R_summary"]["max"]))
    print("below R = 0.5   %d of %d  (registered prediction met: %s)"
          % (n_below, n_series, rec["prediction_met"]))
    print("pdf %s %d B" % (pdf, Path(pdf).stat().st_size))
    print("tif %s %d B" % (tif, Path(tif).stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())