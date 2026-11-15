"""Test (a): does a short profile produce the deep-void plateau?

Pre-registered in Paper_Ledger.md, section 10b. The shallow pilot cases
(roof 0.75 m) are analysed on the full 4.0 m profile (control) and on a
profile cut so that it subtends the same angles at the target as the full
profile does for the deep case (test). Processing: void - no-void
difference B-scan, no mean removal, gap fill, constant-v migration, R in the
target box relative to the complete profile of the same length.

Outputs: results/data/test_profile_length.csv, test_profile_length.json

Run from the Codes folder:  python scripts/test_profile_length.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import physics, processing  # noqa: E402
from pilot_analysis import FREQS, METHODS, OUT, V_MIG, box_peak, load  # noqa: E402

# Deep-case geometry: target at x = 2.09 m, z = 2.62 m; profile 0.20-4.20 m
TAN_LEFT = (2.09 - 0.20) / 2.62
TAN_RIGHT = (4.20 - 2.09) / 2.62
SHALLOW = 0.75


def difference_bscan(freq):
    name = f"d{SHALLOW:.2f}_r0.40_air".replace(".", "p")
    void, dt, x, spec = load(freq, name)
    host, _, _, _ = load(freq, "novoid")
    i0 = int(np.argmax(np.abs(void[:int(10.0 / dt)].mean(axis=1))))
    return void[i0:] - host[i0:], dt, x, spec


def locate(img_env, x, z, x0, roof):
    sx, sz = np.abs(x - x0) <= 0.30, np.abs(z - roof) <= 0.30
    sub = img_env[np.ix_(sz, sx)]
    iz, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return float(x[sx][ix]), float(z[sz][iz])


def curve(data, dt, x, z, x0, roof, D, label, freq):
    ntr = data.shape[1]
    none = np.zeros(ntr, bool)
    full = processing.envelope_z(processing.migrate(data, dt, x, V_MIG, z))
    xc, zc = locate(full, x, z, x0, roof)
    ref = box_peak(full, x, z, xc, zc, 0.25, 0.15)
    centre = int(np.argmin(np.abs(x - x0)))
    dxs = float(x[1] - x[0])
    max_n = min(2 * centre + 1, 2 * (ntr - 1 - centre) + 1) - 4
    rows = []
    for n in range(1, max_n + 1, 2):
        miss = processing.centred_gap(ntr, centre, n)
        for m in METHODS:
            filled = processing.FILLERS[m](data, miss) if n else data
            env = processing.envelope_z(processing.migrate(filled, dt, x, V_MIG, z))
            rows.append({"freq": freq, "profile": label, "method": m, "gap_traces": n,
                         "W_m": round(n * dxs, 4), "G": round(n * dxs / D, 4),
                         "R": round(float(box_peak(env, x, z, xc, zc, 0.25, 0.15) / ref), 4)})
    info = {"x_from_m": float(x[0]), "x_to_m": float(x[-1]), "n_traces": ntr,
            "target_x_m": xc, "target_z_m": zc, "max_G": round(max_n * dxs / D, 3)}
    return rows, info


def g50(rows, freq, label, m):
    for r in rows:
        if r["freq"] == freq and r["profile"] == label and r["method"] == m and r["R"] < 0.5:
            return r["G"]
    return None


def main():
    rows, infos = [], {}
    for freq in FREQS:
        data, dt, x, spec = difference_bscan(freq)
        x0 = spec["void"]["x_m"]
        roof = spec["void"]["roof_depth_m"]
        D = physics.aperture(roof, V_MIG, spec["sim"]["frequency_GHz"])
        z = np.arange(0.30, 1.20, 0.01)
        r_full, i_full = curve(data, dt, x, z, x0, roof, D, "full", freq)
        xt, zt = i_full["target_x_m"], i_full["target_z_m"]
        keep = (x >= xt - TAN_LEFT * zt - 1e-9) & (x <= xt + TAN_RIGHT * zt + 1e-9)
        r_cut, i_cut = curve(data[:, keep], dt, x[keep], z, x0, roof, D, "cut", freq)
        rows += r_full + r_cut
        infos[freq] = {"D_m": D, "full": i_full, "cut": i_cut}
        print(freq, json.dumps(infos[freq]), flush=True)

    with open(OUT / "test_profile_length.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    verdict = {}
    for freq in FREQS:
        for m in METHODS:
            a, b = g50(rows, freq, "full", m), g50(rows, freq, "cut", m)
            verdict[f"{freq}_{m}"] = {"G50_full": a, "G50_cut": b,
                                      "shift": None if (a is None or b is None) else round(b - a, 3)}
    out = {"cases": infos, "G50": verdict}
    (OUT / "test_profile_length.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
