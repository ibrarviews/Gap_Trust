"""Full simulation set: recovery curves, G50/W50, collapse tests, thresholds.

Pre-registered in Paper_Ledger.md, section 11 (before the full run).
Every void model of gaptrust.models.void_set() at 250 and 400 MHz is processed
with exactly the pilot pipeline (scripts/pilot_analysis.analyse_case):
time-zero, mean-trace removal from recorded traces, zero / linear / POCS fill,
constant-v Kirchhoff migration (0.106 m/ns), R in a +-0.25 m x +-0.15 m box
around the complete-data peak.

Outputs: results/data/full_R.csv, results/data/full_summary.json

Run from the Codes folder:  python scripts/full_analysis.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import models  # noqa: E402
from pilot_analysis import FREQS, METHODS, OUT, R_FAIL, V_MIG, analyse_case, cv  # noqa: E402

R_SAFE = 0.8          # -2 dB: upper limit of the data-constrained class
MAX_CENSORED = 1      # cases allowed without failure inside the tested gaps


def first_below(rows, level):
    for r in sorted(rows, key=lambda r: r["gap_traces"]):
        if r["R"] < level:
            return r["W_m"], r["G"]
    return None, None


def collapse_test(cases):
    """cases: {key: (W50, G50)} -> CV statistics, censoring and verdict."""
    done = {k: v for k, v in cases.items() if v[0] is not None}
    cen = sorted(k for k in cases if cases[k][0] is None)
    cvw = cv([v[0] for v in done.values()])
    cvg = cv([v[1] for v in done.values()])
    ok = bool(len(cen) <= MAX_CENSORED and cvg <= 0.5 * cvw)
    return {"n_cases": len(cases), "censored": cen, "CV_W50": cvw, "CV_G50": cvg,
            "criterion_met": ok}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, infos = [], {}
    for freq in FREQS:
        for v in models.void_set():
            rows, info = analyse_case(freq, v.roof_depth_m, v.radius_m, v.fill)
            key = f"{freq}_{models.model_name(v)}"
            for r in rows:
                r.update({"radius_m": v.radius_m, "fill": v.fill})
            all_rows += rows
            infos[key] = dict(info, radius_m=v.radius_m, fill=v.fill, roof_m=v.roof_depth_m,
                              freq=freq)
            print(key, json.dumps(info), flush=True)

    with open(OUT / "full_R.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    per_case = {}
    for key, info in infos.items():
        per_case[key] = {}
        for m in METHODS:
            sub = [r for r in all_rows if r["method"] == m and r["freq"] == info["freq"]
                   and r["radius_m"] == info["radius_m"] and r["fill"] == info["fill"]
                   and abs(r["roof_m"] - info["roof_m"]) < 1e-6]
            W50, G50 = first_below(sub, R_FAIL)
            W80, G80 = first_below(sub, R_SAFE)
            per_case[key][m] = {"W50": W50, "G50": G50, "W80": W80, "G80": G80,
                                "G_max_tested": max(r["G"] for r in sub)}

    tests = {}
    for radius in models.VOID_RADII:
        for m in METHODS:
            cases = {k: (per_case[k][m]["W50"], per_case[k][m]["G50"])
                     for k, i in infos.items() if i["radius_m"] == radius}
            tests[f"r{radius:.2f}_{m}"] = collapse_test(cases)
    h1 = all(tests[f"r0.15_{m}"]["criterion_met"] for m in ("linear", "pocs"))
    h2 = not all(tests[f"r0.40_{m}"]["criterion_met"] for m in ("linear", "pocs"))

    # Classification thresholds from the compact voids (rule fixed in ledger s.11)
    g80 = [per_case[k][m]["G80"] for k, i in infos.items() if i["radius_m"] == 0.15
           for m in ("linear", "pocs") if per_case[k][m]["G80"] is not None]
    g50 = [per_case[k][m]["G50"] for k, i in infos.items() if i["radius_m"] == 0.15
           for m in ("linear", "pocs") if per_case[k][m]["G50"] is not None]
    thresholds = {"G_constrained_below": float(np.min(g80)) if g80 else None,
                  "G_unsupported_from": float(np.median(g50)) if g50 else None,
                  "n_G80": len(g80), "n_G50": len(g50)}

    summary = {"R_fail": R_FAIL, "R_safe": R_SAFE, "v_mig_m_per_ns": V_MIG,
               "max_censored": MAX_CENSORED, "cases": infos, "per_case": per_case,
               "tests": tests, "H1_compact_collapse": h1, "H2_large_no_collapse": h2,
               "thresholds": thresholds}
    (OUT / "full_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"tests": tests, "H1": h1, "H2": h2, "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
