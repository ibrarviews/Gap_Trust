"""Pilot diagnostic: is the deep-void R plateau caused by host/flank energy?

Same pilot cases, gaps and processing as scripts/pilot_analysis.py, but R is
measured on the void response alone: the migrated image of the reconstructed
void model minus the migrated image of the reconstructed no-void twin (same
gap, same method, processed separately). Gap widths are sampled every
4 traces (0.12 m) to halve the run time.

Output: results/data/pilot_R_difference.csv

Run from the Codes folder:  python scripts/pilot_diagnostic_difference.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import physics, processing, style  # noqa: E402
from pilot_analysis import (DEPTHS, FREQS, MAX_GAP_M, METHODS, OUT,  # noqa: E402
                            V_MIG, box_peak, load)

STEP = 4


def image(data, dt, x, z, miss, method):
    base = processing.remove_mean(data, miss)
    filled = processing.FILLERS[method](base, miss)
    return processing.migrate(filled, dt, x, V_MIG, z)


def analyse_case(freq, depth):
    name = f"d{depth:.2f}_r0.40_air".replace(".", "p")
    void, dt, x, spec = load(freq, name)
    host, _, _, _ = load(freq, "novoid")
    vd = spec["void"]
    n_t0 = int(10.0 / dt)
    i0 = int(np.argmax(np.abs(void[:n_t0].mean(axis=1))))
    void, host = void[i0:], host[i0:]
    roof, x0 = vd["roof_depth_m"], vd["x_m"]
    z = np.arange(max(roof - 0.45, 0.05), roof + 0.45, 0.01)
    ntr = void.shape[1]
    centre = int(np.argmin(np.abs(x - x0)))
    dxs = float(x[1] - x[0])
    none = np.zeros(ntr, bool)
    full = processing.envelope_z(image(void, dt, x, z, none, "zero")
                                 - image(host, dt, x, z, none, "zero"))
    sx, sz = np.abs(x - x0) <= 0.30, np.abs(z - roof) <= 0.30
    sub = full[np.ix_(sz, sx)]
    iz, ix = np.unravel_index(np.argmax(sub), sub.shape)
    zc, xc = z[sz][iz], x[sx][ix]
    ref = box_peak(full, x, z, xc, zc, 0.25, 0.15)
    D = physics.aperture(roof, V_MIG, spec["sim"]["frequency_GHz"])
    rows = []
    for n in range(1, int(MAX_GAP_M / dxs) + 2, STEP):
        miss = processing.centred_gap(ntr, centre, n)
        for m in METHODS:
            env = processing.envelope_z(image(void, dt, x, z, miss, m)
                                        - image(host, dt, x, z, miss, m))
            rows.append({"freq": freq, "roof_m": roof, "method": m, "gap_traces": n,
                         "W_m": round(n * dxs, 4), "D_m": round(D, 4),
                         "G": round(n * dxs / D, 4),
                         "R": round(float(box_peak(env, x, z, xc, zc, 0.25, 0.15) / ref), 4)})
    print(freq, depth, "target", round(float(xc), 3), round(float(zc), 3), flush=True)
    return rows


def main():
    rows = []
    for f in FREQS:
        for d in DEPTHS:
            rows += analyse_case(f, d)
    with open(OUT / "pilot_R_difference.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
