#!/usr/bin/env python
"""Field validation of GapTrust, karst lines - runs as registered in ledger 14q.

Reads the cached trace arrays produced by readgssi (results/data/field_cache),
resamples onto a 0.05 m grid, and applies the frozen chain: time_zero,
remove_mean, migrate (V, z), envelope_z. The target is the strongest envelope
amplitude in the 10-60 ns window of the complete image. Gaps are centred on that
target and reconstructed by zero, linear and POCS.

Outputs results/data/field_R.csv and results/logs/field_analysis_karst.json.
Nothing is overwritten; both paths are new.
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from gaptrust import physics, processing, style  # noqa: E402

V_MIG = 0.106                # m/ns, adopted for the karst lines (ledger 14q)
L_LINE = 8.0                 # m, approximate line length (bracket 7-10 m)
DX_TARGET = 0.05             # m, resampling grid
R_FAIL = 0.5
G_MAX = 1.7
T_WIN_NS = (10.0, 60.0)
METHODS = ("zero", "linear", "pocs")

CACHE = CODE / "results" / "data" / "field_cache"
import argparse

OUT_CSV = CODE / "results" / "data" / "field_R.csv"
OUT_LOG = CODE / "results" / "logs" / "field_analysis_karst.json"
# ledger 14r amendment: a target must have one full aperture of lateral
# support inside the profile and its averaging box must lie inside the
# processed time window, otherwise the peak is an edge artefact.
GUARD_TRACES = 10
GUARD_DZ = 0.10

LINES = {
    "great_l14": {"peak_GHz": 0.480, "nominal_GHz": 0.400, "traces": 9799},
    "small_l6": {"peak_GHz": 0.620, "nominal_GHz": 0.400, "traces": 16189},
    "dugout_l2": {"peak_GHz": 0.460, "nominal_GHz": 0.400, "traces": 14193},
}


def box_peak(env, x, z, xc, zc, hx, hz):
    bx = np.abs(x - xc) <= hx
    bz = np.abs(z - zc) <= hz
    return env[np.ix_(bz, bx)].max()


def resample(data, length):
    """Linear interpolation of traces onto a uniform grid over [0, length]."""
    x_old = np.linspace(0.0, length, data.shape[1])
    x_new = np.arange(0.0, length + 1e-9, DX_TARGET)
    out = np.empty((data.shape[0], len(x_new)))
    for k in range(data.shape[0]):
        out[k] = np.interp(x_new, x_old, data[k])
    return out, x_new


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(label, cfg, guard=False):
    arr = np.load(CACHE / ("%s.npz" % label))["data"]           # traces x samples
    data = np.ascontiguousarray(arr.T, dtype=float)             # samples x traces
    dt = 110.86392974853516 / 1023.0                            # ns per sample
    n_old = data.shape[1]
    s_old = L_LINE / n_old

    data, x = resample(data, L_LINE)
    data, t0 = processing.time_zero(data, dt)
    data = processing.remove_mean(data)

    z = np.arange(0.10, 3.40, 0.01)
    img = processing.envelope_z(processing.migrate(data, dt, x, V_MIG, z))

    zsel = (z >= V_MIG * T_WIN_NS[0] / 2.0) & (z <= V_MIG * T_WIN_NS[1] / 2.0)
    if guard:
        gx = np.abs(x - x.mean()) <= (x.max() - x.min()) / 2.0 - GUARD_TRACES * DX_TARGET
        gz = (z >= z[zsel].min() + GUARD_DZ) & (z <= z[zsel].max() - GUARD_DZ)
        zsel = zsel & gz
        xcol = gx
    else:
        xcol = np.ones(len(x), bool)
    sub = img[np.ix_(zsel, xcol)]
    iz, ix = np.unravel_index(np.argmax(sub), sub.shape)
    zc = z[zsel][iz]
    xc = x[xcol][ix]
    ref = box_peak(img, x, z, xc, zc, 0.25, 0.15)

    D = physics.aperture(zc, V_MIG, cfg["peak_GHz"])
    n_max = int(G_MAX * D / DX_TARGET)
    centre = int(np.argmin(np.abs(x - xc)))

    rows = []
    for n in range(1, n_max + 1, 2):
        miss = processing.centred_gap(len(x), centre, n)
        base = processing.remove_mean(data, miss)
        for m in METHODS:
            filled = processing.FILLERS[m](base, miss)
            env = processing.envelope_z(processing.migrate(filled, dt, x, V_MIG, z))
            R = box_peak(env, x, z, xc, zc, 0.25, 0.15) / ref
            rows.append({"line": label, "method": m, "gap_traces": n,
                         "W_m": round(n * DX_TARGET, 4), "G": round(n * DX_TARGET / D, 4),
                         "R": round(float(R), 4)})

    firsts = {}
    for m in METHODS:
        sub = sorted([r for r in rows if r["method"] == m], key=lambda r: r["gap_traces"])
        firsts[m] = next((r for r in sub if r["R"] < R_FAIL), None)
    info = {"target_x_m": round(float(xc), 3), "target_z_m": round(float(zc), 3),
            "reference_amp": float(ref), "D_m": round(float(D), 4),
            "n_traces_resampled": int(len(x)), "spacing_orig_m": round(s_old, 6),
            "n_gap_widths": int(len(range(1, n_max + 1, 2))), "time_zero_ns": float(t0),
            "D_sensitivity": {
                "v0.080_f_peak": round(float(physics.aperture(zc, 0.080, cfg["peak_GHz"])), 4),
                "v0.122_f_peak": round(float(physics.aperture(zc, 0.122, cfg["peak_GHz"])), 4),
                "v0.106_f_400": round(float(physics.aperture(zc, 0.106, cfg["nominal_GHz"])), 4)}}
    rs = {m: [r["R"] for r in rows if r["method"] == m] for m in METHODS}
    info["spearman_R_vs_gap"] = {m: round(spearman([r["gap_traces"] for r in rows
                                                    if r["method"] == m], rs[m]), 3)
                                 for m in METHODS}
    info["W50_m"] = {m: (None if firsts[m] is None else firsts[m]["W_m"]) for m in METHODS}
    info["G50"] = {m: (None if firsts[m] is None else firsts[m]["G"]) for m in METHODS}
    return rows, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-guard", action="store_true",
                    help="apply the ledger 14r target-support rule")
    args = ap.parse_args()
    guard = args.edge_guard
    all_rows, infos = [], {}
    for label, cfg in LINES.items():
        rows, info = analyse(label, cfg, guard)
        all_rows += rows
        infos[label] = info
        print(label, json.dumps(info), flush=True)

    tag = "_guarded" if guard else ""
    out_csv = OUT_CSV.with_name(OUT_CSV.stem + tag + OUT_CSV.suffix)
    out_log = OUT_LOG.with_name(OUT_LOG.stem + tag + OUT_LOG.suffix)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    pred = {"P1_any_line_reaches_W50": any(i["W50_m"]["zero"] is not None
                                          for i in infos.values()),
            "P2_all_spearman_negative": all(v < 0 for i in infos.values()
                                            for v in i["spearman_R_vs_gap"].values())}
    out_log.write_text(json.dumps(
        {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "edge_guard": guard,
         "constants": {"V_MIG": V_MIG, "L_LINE": L_LINE, "DX_TARGET": DX_TARGET,
                       "R_FAIL": R_FAIL, "G_MAX": G_MAX, "T_WIN_NS": T_WIN_NS},
         "lines": infos, "predictions": pred}, indent=2), encoding="utf-8")
    print("predictions", pred)
    print("written:", out_csv)
    print("written:", out_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())