"""r/lambda test analysis (ledger sections 14b-14d).

Twelve new 250 MHz air-filled models (radii 0.10, 0.21, 0.28, 0.34 m at roof
depths 0.75, 1.50, 2.50 m) are analysed with the frozen pipeline of
pilot_analysis.analyse_case: time zero, mean-trace removal from the recorded
traces only, zero / linear / f-k POCS fill, Kirchhoff migration at 0.106 m/ns,
depth envelope, R in the target box (+-0.25 m lateral, +-0.15 m depth).

Nothing in the pipeline, the thresholds or the metric is changed; this script
only supplies the new model list and the group statistics of section 14d.

Outputs (new names only):
    results/data/rlam_R.csv          per-gap R for every model and method
    results/data/rlam_summary.json   W50 / G50 per model, group CVs, H3a / H3b verdict
    results/logs/rlam_analysis.json  parameters, versions, input and output checksums

Run from the Codes folder:
    python scripts/rlam_analysis.py --check    (verify the 12 inputs, write nothing)
    python scripts/rlam_analysis.py            (analyse and write the outputs)
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from gaptrust import models  # noqa: E402
import pilot_analysis as pa  # noqa: E402

FREQ = "250MHz"
FILL = "air"
RADII = (0.10, 0.21, 0.28, 0.34)
DEPTHS = (0.75, 1.50, 2.50)
COMPACT = (0.10, 0.21)
EXTENDED = (0.28, 0.34)
METHODS = ("linear", "pocs")
LAMBDA_M = pa.V_MIG / 0.25          # 0.424 m at 250 MHz, v = 0.106 m/ns
MIN_UNCENSORED = 5
DATA = CODE_ROOT / "results" / "data"
LOGS = CODE_ROOT / "results" / "logs"
OUT_CSV = DATA / "rlam_R.csv"
OUT_JSON = DATA / "rlam_summary.json"
OUT_LOG = LOGS / "rlam_analysis.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cases():
    for radius in RADII:
        for depth in DEPTHS:
            name = models.model_name(models.VoidSpec(depth, radius, FILL))
            yield radius, depth, name, pa.SIM_ROOT / FREQ / (name + ".npz")


def group_stats(case_summary, radii):
    out = {}
    for m in METHODS:
        keys = [k for k, v in case_summary.items() if v["radius_m"] in radii]
        w50 = [case_summary[k]["W50_m"][m] for k in keys]
        g50 = [case_summary[k]["G50"][m] for k in keys]
        n_unc = sum(1 for v in g50 if v is not None)
        cens = len(keys) - n_unc
        cv_w = pa.cv(w50)
        cv_g = pa.cv(g50)
        tested = n_unc >= MIN_UNCENSORED
        ratio = (cv_g / cv_w) if cv_w and cv_w == cv_w else float("nan")
        met = bool(tested and cens <= 1 and ratio == ratio and ratio <= 0.5)
        out[m] = {"n_models": len(keys), "n_uncensored": n_unc, "n_censored": cens,
                  "CV_W50": cv_w, "CV_G50": cv_g, "ratio_CV_G50_over_CV_W50": ratio,
                  "tested": tested, "criterion_met": met}
    return out


def atomic_write_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the 12 input files and the output destinations, write nothing")
    args = ap.parse_args()

    inputs = []
    for radius, depth, name, path in cases():
        inputs.append({"radius_m": radius, "roof_m": depth, "name": name,
                       "path": str(path), "exists": path.is_file(),
                       "bytes": path.stat().st_size if path.is_file() else 0,
                       "sha256": sha256(path) if path.is_file() else ""})

    missing = [r["name"] for r in inputs if not r["exists"]]
    if missing:
        print("MISSING INPUT(S):")
        for name in missing:
            print("  " + name)
        return 2

    if args.check:
        for r in inputs:
            print("%s  %10d B  %s" % (r["name"], r["bytes"], r["sha256"][:16]))
        for p in (OUT_CSV, OUT_JSON, OUT_LOG):
            print(("EXISTS  " if p.exists() else "new     ") + str(p))
        print("%d inputs present" % len(inputs))
        return 0

    clash = [p for p in (OUT_CSV, OUT_JSON, OUT_LOG) if p.exists()]
    if clash:
        print("REFUSING TO OVERWRITE EXISTING OUTPUT(S):")
        for p in clash:
            print("  " + str(p))
        return 3

    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_rows, case_summary = [], {}
    for radius, depth, name, path in cases():
        rows, info = pa.analyse_case(FREQ, depth, radius, FILL)
        for row in rows:
            row["radius_m"] = radius
            row["r_over_lambda"] = round(radius / LAMBDA_M, 3)
            row["fill"] = FILL
        all_rows += rows
        entry = {"radius_m": radius, "roof_m": depth,
                 "r_over_lambda": round(radius / LAMBDA_M, 3),
                 "D_m": round(float(info["D_m"]), 4),
                 "target_x_m": round(float(info["target_x_m"]), 4),
                 "target_z_m": round(float(info["target_z_m"]), 4),
                 "W50_m": {}, "G50": {}}
        for m in METHODS:
            sub = [r for r in rows if r["method"] == m]
            w50, g50 = pa.first_failure(sub)
            entry["W50_m"][m] = w50
            entry["G50"][m] = g50
        case_summary[name] = entry
        print("%-22s r=%.2f m  r/lambda=%.3f  W50 %s  G50 %s"
              % (name, radius, entry["r_over_lambda"],
                 [entry["W50_m"][m] for m in METHODS], [entry["G50"][m] for m in METHODS]))
    runtime = time.time() - t0

    summary = {"test": "r/lambda on unseen models (ledger 14b-14d)",
               "frequency": FREQ, "fill": FILL, "lambda_m": LAMBDA_M,
               "lambda_definition": "lambda = V_MIG / f = 0.106 / 0.25 m = 0.424 m",
               "v_mig_m_per_ns": pa.V_MIG, "R_fail": pa.R_FAIL,
               "R_safe": getattr(pa, "R_SAFE", None), "max_gap_m": pa.MAX_GAP_M,
               "radii_m": list(RADII), "roof_depths_m": list(DEPTHS),
               "groups_definition": {"compact": list(COMPACT), "extended": list(EXTENDED)},
               "cases": case_summary,
               "groups": {"compact (r/lambda 0.236 and 0.495)": group_stats(case_summary, COMPACT),
                          "extended (r/lambda 0.660 and 0.802)": group_stats(case_summary, EXTENDED)}}

    comp = summary["groups"]["compact (r/lambda 0.236 and 0.495)"]
    ext = summary["groups"]["extended (r/lambda 0.660 and 0.802)"]
    h3a = all(comp[m]["criterion_met"] for m in METHODS)
    h3b = not all(ext[m]["criterion_met"] for m in METHODS)
    if h3a and h3b:
        verdict = "supported"
    elif h3a and not h3b:
        verdict = "partly supported"
    else:
        verdict = "not supported"
    summary["decision"] = {"H3a_met": bool(h3a), "H3b_met": bool(h3b), "outcome": verdict,
                           "rule": "H3a: CV(G50) <= 0.5 CV(W50) for linear AND POCS, "
                                   "at most 1 censored model per method; "
                                   "H3b: H3a criterion NOT met for at least one method"}

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    atomic_write_text(OUT_JSON, json.dumps(summary, indent=2))

    log = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "script": str(Path(__file__).relative_to(CODE_ROOT)),
           "python": sys.version.split()[0], "executable": sys.executable,
           "platform": platform.platform(),
           "numpy": __import__("numpy").__version__,
           "runtime_s": round(runtime, 1),
           "inputs": inputs,
           "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                       for p in (OUT_CSV, OUT_JSON)},
           "parameters": {"frequency": FREQ, "fill": FILL, "radii_m": list(RADII),
                          "roof_depths_m": list(DEPTHS), "lambda_m": LAMBDA_M,
                          "v_mig": pa.V_MIG, "R_fail": pa.R_FAIL, "max_gap_m": pa.MAX_GAP_M,
                          "methods": list(METHODS), "min_uncensored": MIN_UNCENSORED,
                          "pipeline": "pilot_analysis.analyse_case (unchanged)"},
           "decision": summary["decision"]}
    atomic_write_text(OUT_LOG, json.dumps(log, indent=2))

    print("")
    for gname, g in summary["groups"].items():
        for m in METHODS:
            s = g[m]
            print("%-38s %-6s n=%d unc=%d cens=%d CV_W50=%.3f CV_G50=%.3f ratio=%.3f met=%s"
                  % (gname, m, s["n_models"], s["n_uncensored"], s["n_censored"],
                     s["CV_W50"], s["CV_G50"], s["ratio_CV_G50_over_CV_W50"],
                     s["criterion_met"]))
    print("")
    print("H3a met: %s ; H3b met: %s ; outcome: %s" % (h3a, h3b, verdict))
    print("runtime %.1f s" % runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())