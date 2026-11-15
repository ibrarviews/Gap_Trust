"""Pilot analysis: does failure collapse against G? (pre-registered test)

For each pilot case (250/400 MHz x void roof 0.75/2.50 m, air, r = 0.40 m):
  1. time-zero the complete B-scan;
  2. for gap widths of 1, 3, 5, ... traces centred on the void apex:
     remove the mean recorded trace, fill the gap (zero / linear / POCS),
     migrate (constant v), take the depth envelope;
  3. R = peak envelope inside the target box / same for the complete data.
Target box: centred on the complete-data envelope peak found within +-0.3 m
of the true roof position; half-widths 0.25 m (lateral) and 0.15 m (depth).

Outputs: results/data/pilot_R.csv, results/data/pilot_summary.json

Run from the Codes folder:
    python scripts/pilot_analysis.py                 (pilot, r = 0.40 m)
    python scripts/pilot_analysis.py --size-test     (test b, r = 0.15 m, 250 MHz)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import models, physics, processing, style  # noqa: E402

V_MIG = 0.106          # m/ns, limestone eps_r ~ 8
DEPTHS = (0.75, 2.50)
FREQS = ("250MHz", "400MHz")
METHODS = ("zero", "linear", "pocs")
MAX_GAP_M = 2.4
R_FAIL = 0.5
SIM_ROOT = style.CODE_ROOT / "results" / "sims"
OUT = style.CODE_ROOT / "results" / "data"


def load(freq, name):
    f = np.load(SIM_ROOT / freq / f"{name}.npz")
    return f["data"].astype(float), float(f["dt_ns"]), f["x_m"], json.loads(str(f["spec"]))


def box_peak(env, x, z, xc, zc, hx, hz):
    bx = np.abs(x - xc) <= hx
    bz = np.abs(z - zc) <= hz
    return env[np.ix_(bz, bx)].max()


def analyse_case(freq, depth, radius=0.40, fill="air"):
    name = models.model_name(models.VoidSpec(depth, radius, fill))
    data, dt, x, spec = load(freq, name)
    void = spec["void"]
    f_ghz = spec["sim"]["frequency_GHz"]
    data, _ = processing.time_zero(data, dt)
    x0 = void["x_m"]
    roof = void["roof_depth_m"]
    z = np.arange(max(roof - 0.45, 0.05), roof + 0.45, 0.01)
    ntr = data.shape[1]
    centre = int(np.argmin(np.abs(x - x0)))
    dxs = float(x[1] - x[0])

    full = processing.envelope_z(processing.migrate(
        processing.remove_mean(data), dt, x, V_MIG, z))
    # locate target in the complete image
    sel_x = np.abs(x - x0) <= 0.30
    sel_z = np.abs(z - roof) <= 0.30
    sub = full[np.ix_(sel_z, sel_x)]
    iz, ix = np.unravel_index(np.argmax(sub), sub.shape)
    zc, xc = z[sel_z][iz], x[sel_x][ix]
    ref = box_peak(full, x, z, xc, zc, 0.25, 0.15)

    D = physics.aperture(roof, V_MIG, f_ghz)
    rows = []
    for n in range(1, int(MAX_GAP_M / dxs) + 2, 2):
        miss = processing.centred_gap(ntr, centre, n)
        base = processing.remove_mean(data, miss)
        for m in METHODS:
            filled = processing.FILLERS[m](base, miss)
            env = processing.envelope_z(processing.migrate(filled, dt, x, V_MIG, z))
            R = box_peak(env, x, z, xc, zc, 0.25, 0.15) / ref
            rows.append({"freq": freq, "roof_m": roof, "method": m, "gap_traces": n,
                         "W_m": round(n * dxs, 4), "D_m": round(D, 4),
                         "G": round(n * dxs / D, 4), "R": round(float(R), 4)})
    info = {"target_x_m": float(xc), "target_z_m": float(zc), "D_m": float(D),
            "trace_spacing_m": dxs, "f_GHz": f_ghz}
    return rows, info


def first_failure(rows):
    """Smallest gap width (m) and G with R < R_FAIL; None if never fails."""
    for r in sorted(rows, key=lambda r: r["gap_traces"]):
        if r["R"] < R_FAIL:
            return r["W_m"], r["G"]
    return None, None


def cv(a):
    a = np.asarray([v for v in a if v is not None], float)
    return float(a.std() / a.mean()) if len(a) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-test", action="store_true")
    args = ap.parse_args()
    freqs, radius, tag = FREQS, 0.40, "pilot"
    if args.size_test:
        freqs, radius, tag = ("250MHz",), 0.15, "test_size"
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, infos = [], {}
    for freq in freqs:
        for depth in DEPTHS:
            rows, info = analyse_case(freq, depth, radius)
            all_rows += rows
            infos[f"{freq}_{depth:.2f}"] = info
            print(freq, depth, info, flush=True)

    with open(OUT / f"{tag}_R.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    summary = {"R_fail": R_FAIL, "v_mig_m_per_ns": V_MIG, "cases": infos, "methods": {}}
    for m in METHODS:
        W50, G50 = {}, {}
        for key in infos:
            freq, depth = key.split("_")
            sub = [r for r in all_rows if r["method"] == m and r["freq"] == freq
                   and abs(r["roof_m"] - float(depth)) < 1e-6]
            W50[key], G50[key] = first_failure(sub)
        summary["methods"][m] = {"W50_m": W50, "G50": G50,
                                 "CV_W50": cv(W50.values()), "CV_G50": cv(G50.values())}
    summary["radius_m"] = radius
    if not args.size_test:
        for m in ("linear", "pocs"):
            s = summary["methods"][m]
            s["pass"] = bool(s["CV_G50"] <= 0.5 * s["CV_W50"])
        summary["pilot_pass"] = all(summary["methods"][m]["pass"] for m in ("linear", "pocs"))
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["methods"], indent=2))
    if not args.size_test:
        print("PILOT PASS:", summary["pilot_pass"])


if __name__ == "__main__":
    main()
