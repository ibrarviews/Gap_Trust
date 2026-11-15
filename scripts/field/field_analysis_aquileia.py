#!/usr/bin/env python
"""Aquileia field validation - runs as registered in ledger 14t.

Reads three SEG-Y profiles with the verified IBM-float reader, applies the frozen
chain (time_zero, remove_mean, migrate, envelope_z) and measures R against gap
width. Site constants come from Zhao et al. (2013): v = 0.10 m/ns, f = 250 MHz,
trace spacing 0.05 m, sampling 0.398 ns (dt fields stored in picoseconds).
"""

import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from gaptrust import physics, processing, style  # noqa: E402

V = 0.10
F_GHZ = 0.250
DX = 0.05
R_FAIL = 0.5
G_MAX = 1.7
T_WIN_NS = (10.0, 60.0)
GUARD_TRACES = 10
METHODS = ("zero", "linear", "pocs")

ROOT = CODE.parent / "Aquileia Data"
OUT_CSV = CODE / "results" / "data" / "field_R_aquileia_full.csv"
OUT_LOG = CODE / "results" / "logs" / "field_analysis_aquileia_full.json"

FILES = {
    "aquileia_A": ("1/250EW", "PROF3.SGY"),
    "aquileia_B": ("2", "PROF10.sgy"),
    "aquileia_C": ("3", "prof10.sgy"),
}


def reader():
    spec = importlib.util.spec_from_file_location(
        "field_inventory", CODE / "scripts" / "field_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sgy_read


def box_peak(env, x, z, xc, zc, hx, hz):
    bx = np.abs(x - xc) <= hx
    bz = np.abs(z - zc) <= hz
    return env[np.ix_(bz, bx)].max()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def find_file(folder, name):
    base = ROOT.joinpath(*folder.split("/"))
    hits = [p for p in base.rglob("*") if p.is_file() and p.name.lower() == name.lower()]
    if not hits:
        raise FileNotFoundError(base / name)
    return hits[0]


def analyse(label, sgy_read):
    folder, name = FILES[label]
    h = sgy_read(find_file(folder, name))
    data = h["traces"].T.astype(float)                # samples x traces
    dt = h["dt"] / 1000.0                             # ps -> ns
    x = np.arange(data.shape[1]) * DX

    data, t0 = processing.time_zero(data, dt)
    data = processing.remove_mean(data)

    z = np.arange(0.10, 3.40, 0.01)
    img = processing.envelope_z(processing.migrate(data, dt, x, V, z))

    zsel = (z >= V * T_WIN_NS[0] / 2.0) & (z <= V * T_WIN_NS[1] / 2.0)
    gx = np.abs(x - x.mean()) <= (x.max() - x.min()) / 2.0 - GUARD_TRACES * DX
    gz = (z >= z[zsel].min() + 0.10) & (z <= z[zsel].max() - 0.10)
    zsel = zsel & gz
    sub = img[np.ix_(zsel, gx)]
    iz, ix = np.unravel_index(np.argmax(sub), sub.shape)
    zc, xc = z[zsel][iz], x[gx][ix]
    # reconstructions are evaluated only in a window around the target: the
    # Kirchhoff image is computed depth sample by depth sample, so this is
    # identical inside the box and avoids migrating the full 3.4 m grid 48 times
    z_loc = z
    ref = box_peak(img, x, z_loc, xc, zc, 0.25, 0.15)

    D = physics.aperture(zc, V, F_GHZ)
    n_max = int(G_MAX * D / DX)
    centre = int(np.argmin(np.abs(x - xc)))

    rows = []
    for n in range(1, n_max + 1, 2):
        miss = processing.centred_gap(len(x), centre, n)
        base = processing.remove_mean(data, miss)
        for m in METHODS:
            filled = processing.FILLERS[m](base, miss)
            env = processing.envelope_z(processing.migrate(filled, dt, x, V, z_loc))
            R = box_peak(env, x, z_loc, xc, zc, 0.25, 0.15) / ref
            rows.append({"line": label, "method": m, "gap_traces": n,
                         "W_m": round(n * DX, 4), "G": round(n * DX / D, 4),
                         "R": round(float(R), 4)})

    firsts = {}
    for m in METHODS:
        sub_m = sorted([r for r in rows if r["method"] == m], key=lambda r: r["gap_traces"])
        firsts[m] = next((r for r in sub_m if r["R"] < R_FAIL), None)

    info = {"file": find_file(folder, name).name, "ns": int(h["ns"]), "dt_ns": dt,
            "ntraces": int(h["ntraces"]), "format_code": int(h["format_code"]),
            "target_x_m": round(float(xc), 3), "target_z_m": round(float(zc), 3),
            "reference_amp": float(ref), "D_m": round(float(D), 4),
            "time_zero_ns": float(t0), "n_gap_widths": len(range(1, n_max + 1, 2)),
            "W50_m": {m: (None if firsts[m] is None else firsts[m]["W_m"]) for m in METHODS},
            "G50": {m: (None if firsts[m] is None else firsts[m]["G"]) for m in METHODS},
            "spearman_R_vs_gap": {m: round(spearman(
                [r["gap_traces"] for r in rows if r["method"] == m],
                [r["R"] for r in rows if r["method"] == m]), 3) for m in METHODS}}
    return rows, info


def main():
    sgy_read = reader()
    all_rows, infos = [], {}
    for label in FILES:
        rows, info = analyse(label, sgy_read)
        all_rows += rows
        infos[label] = info
        print(label, json.dumps(info), flush=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    pred = {"P3_any_profile_reaches_W50": any(i["W50_m"]["zero"] is not None
                                             for i in infos.values()),
            "P4_zero_fails_before_linear": all(
                (i["G50"]["linear"] is None) or
                (i["G50"]["zero"] is not None and i["G50"]["zero"] < i["G50"]["linear"])
                for i in infos.values())}
    OUT_LOG.write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constants": {"V": V, "F_GHz": F_GHZ, "DX": DX, "R_FAIL": R_FAIL,
                      "G_MAX": G_MAX, "T_WIN_NS": T_WIN_NS, "GUARD_TRACES": GUARD_TRACES},
        "lines": infos, "predictions": pred}, indent=2), encoding="utf-8")
    print("predictions", pred)
    print("written:", OUT_CSV)
    print("written:", OUT_LOG)


if __name__ == "__main__":
    main()