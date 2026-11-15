#!/usr/bin/env python
"""Off-target gap control for GapTrust, pre-registered as ledger 14v.

A gap of each profile's own zero-fill W50 is placed at 1.0, 1.5 and 2.0 m from the
target, on both sides, and R is measured in the same averaging box. Everything
else is the frozen Fig. 6 chain.
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
from gaptrust import processing  # noqa: E402

KARST = {"V": 0.106, "DX": 0.05, "L": 8.0, "DT": 110.86392974853516 / 1023.0}
AQU = {"V": 0.10, "DX": 0.05}
DX = 0.05
R_FAIL = 0.5
GUARD_TRACES = 10
OFFSETS = (1.0, 1.5, 2.0)
METHODS = ("zero", "linear")

CACHE = CODE / "results" / "data" / "field_cache"
LOG_KARST = CODE / "results" / "logs" / "field_analysis_karst_guarded.json"
LOG_AQU = CODE / "results" / "logs" / "field_analysis_aquileia_full.json"
OUT_CSV = CODE / "results" / "data" / "field_R_offtarget.csv"
OUT_LOG = CODE / "results" / "logs" / "field_control_offtarget.json"
ROOT_AQU = CODE.parent / "Aquileia Data"

PROFILES = (
    ("great_l14", "karst"),
    ("small_l6", "karst"),
    ("dugout_l2", "karst"),
    ("aquileia_A", "aqu"),
    ("aquileia_B", "aqu"),
    ("aquileia_C", "aqu"),
)


def sgy_reader():
    spec = importlib.util.spec_from_file_location(
        "field_inventory", CODE / "scripts" / "field_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sgy_read


def load(site, label, sgy_read):
    if site == "karst":
        arr = np.load(CACHE / (label + ".npz"))["data"]
        data = np.ascontiguousarray(arr.T, dtype=float)
        x_old = np.linspace(0.0, KARST["L"], data.shape[1])
        x = np.arange(0.0, KARST["L"] + 1e-9, KARST["DX"])
        data = np.stack([np.interp(x, x_old, data[k]) for k in range(data.shape[0])])
        dt, v = KARST["DT"], KARST["V"]
    else:
        folder, name = {"aquileia_A": ("1/250EW", "PROF3.SGY"),
                        "aquileia_B": ("2", "PROF10.sgy"),
                        "aquileia_C": ("3", "prof10.sgy")}[label]
        hit = [p for p in ROOT_AQU.joinpath(*folder.split("/")).rglob("*")
               if p.is_file() and p.name.lower() == name.lower()][0]
        h = sgy_read(hit)
        data = h["traces"].T.astype(float)
        dt = h["dt"] / 1000.0
        x = np.arange(data.shape[1]) * AQU["DX"]
        v = AQU["V"]
    data, _ = processing.time_zero(data, dt)
    data = processing.remove_mean(data)
    z = np.arange(0.10, 3.40, 0.01)
    img = processing.envelope_z(processing.migrate(data, dt, x, v, z))
    return data, dt, x, z, img, v


def table_R(label, method, n):
    hit = [r["R"] for r in TABLE
           if r["line"] == label and r["method"] == method and r["gap_traces"] == n]
    return None if not hit else round(hit[0], 4)


def box_peak(env, x, z, xc, zc, hx=0.25, hz=0.15):
    bx = np.abs(x - xc) <= hx
    bz = np.abs(z - zc) <= hz
    return env[np.ix_(bz, bx)].max()


def load_table():
    out = []
    for path in (CODE / "results" / "data" / "field_R_guarded.csv",
                 CODE / "results" / "data" / "field_R_aquileia_full.csv"):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append({"line": r["line"], "method": r["method"],
                            "gap_traces": int(r["gap_traces"]),
                            "R": float(r["R"])})
    return out


def main():
    global TABLE
    TABLE = load_table()
    sgy_read = sgy_reader()
    tab = {"karst": json.loads(LOG_KARST.read_text(encoding="utf-8"))["lines"],
           "aqu": json.loads(LOG_AQU.read_text(encoding="utf-8"))["lines"]}

    rows, checks, skips = [], [], []
    for label, site in PROFILES:
        li = tab[site][label]
        xc, zc = li["target_x_m"], li["target_z_m"]
        W50 = li["W50_m"]["zero"]
        n = int(round(W50 / DX))
        data, dt, x, z, img, v = load(site, label, sgy_read)
        # reference and reconstructions must share one depth grid: the Hilbert
        # envelope is not local, so a windowed image would not reproduce the
        # tabulated R (verified: full grid reproduces it exactly)
        z_loc, img_loc = z, img
        ref = box_peak(img_loc, x, z_loc, xc, zc)
        centre = int(np.argmin(np.abs(x - xc)))

        for method in METHODS:
            # the on-target value is taken from the frozen tables, not recomputed
            # here: a second code path did not reproduce the frozen chain exactly
            # for the linear reconstruction on one profile (ledger 14x, 14y)
            R_on = table_R(label, method, n)
            if R_on is None:
                print("  note:", label, method, "censored in the table", flush=True)
            else:
                rows.append({"line": label, "method": method, "offset_m": 0.0,
                             "gap_traces": n, "R": round(float(R_on), 4)})
            checks.append({"line": label, "method": method,
                           "gap_traces": n, "R_on_table": R_on})

            for off in OFFSETS:
                for sign in (-1.0, 1.0):
                    c_off = centre + int(round(sign * off / DX))
                    half = (n - 1) // 2
                    reason = None
                    if (c_off - half < GUARD_TRACES
                            or c_off + half + 1 > len(x) - GUARD_TRACES):
                        reason = "end guard"
                    elif abs(c_off * DX - xc) < 0.25 + n * DX / 2.0:
                        reason = "overlaps target box"
                    if reason:
                        skips.append({"line": label, "method": method,
                                      "offset_m": sign * off,
                                      "trace": c_off, "reason": reason})
                        continue
                    miss_off = processing.centred_gap(len(x), c_off, n)
                    idx = np.flatnonzero(miss_off)
                    assert idx.size == n
                    base_off = processing.remove_mean(data, miss_off)
                    env_off = processing.envelope_z(processing.migrate(
                        processing.FILLERS[method](base_off, miss_off),
                        dt, x, v, z_loc))
                    R_off = box_peak(env_off, x, z_loc, xc, zc) / ref
                    rows.append({"line": label, "method": method,
                                 "offset_m": sign * off, "gap_traces": n,
                                 "R": round(float(R_off), 4)})
            print(label, method, "done", flush=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["line", "method", "offset_m",
                                           "gap_traces", "R"])
        w.writeheader()
        w.writerows(rows)

    def val(label, method, off):
        hit = [r for r in rows if r["line"] == label and r["method"] == method
               and abs(r["offset_m"] - off) < 1e-9]
        return None if not hit else hit[0]["R"]

    p5 = all((val(r["line"], "zero", r["offset_m"]) is not None)
             for r in rows if r["method"] == "zero" and abs(abs(r["offset_m"]) - 1.0) < 1e-9)
    p5_detail = {}
    for label, site in PROFILES:
        r_on = table_R(label, "zero",
                       int(round(tab[site][label]["W50_m"]["zero"] / DX)))
        cand = [val(label, "zero", -1.0), val(label, "zero", 1.0)]
        cand = [c for c in cand if c is not None]
        p5_detail[label] = {"R_on": r_on, "R_off_1m": cand,
                            "ok": bool(cand) and all(c >= R_FAIL and c - r_on >= 0.15
                                                     for c in cand)}
    p5 = all(v["ok"] for v in p5_detail.values())

    mean1 = [val(l, "zero", s * 1.0) for l, _ in PROFILES for s in (-1.0, 1.0)]
    mean2 = [val(l, "zero", s * 2.0) for l, _ in PROFILES for s in (-1.0, 1.0)]
    mean1 = [m for m in mean1 if m is not None]
    mean2 = [m for m in mean2 if m is not None]
    p6 = (float(np.mean(mean2)) > float(np.mean(mean1))) if mean1 and mean2 else None

    p7 = all((val(l, "linear", s * 1.0) is None) or (val(l, "linear", s * 1.0) >= R_FAIL)
             for l, _ in PROFILES for s in (-1.0, 1.0))

    OUT_LOG.write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "constants": {"DX": DX, "R_FAIL": R_FAIL, "GUARD_TRACES": GUARD_TRACES,
                      "OFFSETS": OFFSETS, "METHODS": METHODS,
                      "KARST": KARST, "AQU": AQU},
        "inputs": [str(OUT_CSV.relative_to(CODE))],
        "on_target_from_table": checks,
        "skips": skips,
        "predictions": {"P5": p5, "P5_detail": p5_detail,
                        "P6": p6, "P6_mean_1m": float(np.mean(mean1)) if mean1 else None,
                        "P6_mean_2m": float(np.mean(mean2)) if mean2 else None,
                        "P7": p7},
    }, indent=2), encoding="utf-8")
    print(json.dumps({"P5": p5, "P6": p6, "P7": p7, "skips": len(skips)}))
    print("written:", OUT_CSV)
    print("written:", OUT_LOG)


if __name__ == "__main__":
    main()