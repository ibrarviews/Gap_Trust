#!/usr/bin/env python
"""Independent recovery of the full-set GapTrust analysis (ledger section 12).

The original full_analysis.py, pilot_analysis.py and the gaptrust modules are
imported / left untouched. This script re-applies the frozen pipeline through
pilot_analysis.analyse_case over the 24 void models of models.void_set() and
writes results under NEW names only:

    results/data/full_R_recovered.csv
    results/data/full_summary_recovered.json
    results/data/failure_points_recovered.csv
    results/logs/full_analysis_recovered.log
    results/recovery/full_recovery_manifest.json
    results/recovery/cache/<freq>_<model>.json          (per-case cache)

Usage, from the Codes folder:

    python scripts/full_analysis_recovery.py --list
    python scripts/full_analysis_recovery.py --case 400MHz:d0p75_r0p15_clay
    python scripts/full_analysis_recovery.py --run          (all 24 cases)

--list performs no computation. --case runs exactly one case and writes only
its cache file. --run writes the final recovered outputs.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__ as GAPTRUST_VERSION  # noqa: E402
from gaptrust import models, physics, style  # noqa: E402
from pilot_analysis import (  # noqa: E402
    FREQS, MAX_GAP_M, METHODS, OUT, R_FAIL, SIM_ROOT, V_MIG,
    analyse_case, cv,
)

# ----------------------------------------------------------------------------
# Frozen constants (identical to full_analysis.py / pilot_analysis.py)
# ----------------------------------------------------------------------------
R_SAFE = 0.8             # -2 dB
MAX_CENSORED = 1         # cases allowed without failure inside the tested gaps
BOX_HX, BOX_HZ = 0.25, 0.15     # target box, implemented in analyse_case
SEARCH = 0.30                   # box-centring search radius, analyse_case

RESULTS = style.CODE_ROOT / "results"
RECOVERY = RESULTS / "recovery"
CACHE_DIR = RECOVERY / "cache"
FULL_R = OUT / "full_R_recovered.csv"
FULL_S = OUT / "full_summary_recovered.json"
FAILPTS = OUT / "failure_points_recovered.csv"
LOG = RESULTS / "logs" / "full_analysis_recovered.log"
MANIFEST = RECOVERY / "full_recovery_manifest.json"
SEMANTIC = (FULL_R, FULL_S, FAILPTS, LOG, MANIFEST)

FROZEN_MODULES = (
    "gaptrust/physics.py",
    "gaptrust/models.py",
    "gaptrust/processing.py",
    "scripts/pilot_analysis.py",
    "scripts/full_analysis.py",
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def sha256_file(path):
    """SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def environment():
    return {
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "numpy": np.__version__,
        "scipy": pkg_version("scipy"),
        "matplotlib": pkg_version("matplotlib"),
        "h5py": pkg_version("h5py"),
        "gprMax": pkg_version("gprMax"),
        "gaptrust": GAPTRUST_VERSION,
        "platform": platform.platform(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def cases():
    """The 24 analysis cases: every void model of void_set() at both frequencies."""
    return [(f, v) for f in FREQS for v in models.void_set()]


def case_key(freq, void):
    return f"{freq}_{models.model_name(void)}"


def input_path(freq, void):
    return SIM_ROOT / freq / f"{models.model_name(void)}.npz"


def cache_path(freq, void):
    return CACHE_DIR / f"{case_key(freq, void)}.json"


def widths_for(freq):
    """Gap schedule length implied by the frozen trace spacing (41)."""
    dx = models.SIMS[freq].trace_spacing_m
    return list(range(1, int(MAX_GAP_M / dx) + 2, 2))


def atomic_write_text(path, text, overwrite=False):
    """Write via a .tmp sibling and os.replace; refuse to clobber by default."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)          # atomic on the same filesystem


def write_csv(path, rows, fieldnames):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def first_below(rows, level):
    """Smallest gap width / G with R below the level; (None, None) if censored."""
    for r in sorted(rows, key=lambda r: r["gap_traces"]):
        if r["R"] < level:
            return r["W_m"], r["G"]
    return None, None


def collapse_test(case_map):
    """CV statistics, censoring and the frozen criterion (as full_analysis.py)."""
    done = {k: v for k, v in case_map.items() if v[0] is not None}
    cen = sorted(k for k in case_map if case_map[k][0] is None)
    cvw = cv([v[0] for v in done.values()])
    cvg = cv([v[1] for v in done.values()])
    ok = bool(len(cen) <= MAX_CENSORED and cvg <= 0.5 * cvw)
    return {"n_cases": len(case_map), "censored": cen, "CV_W50": cvw,
            "CV_G50": cvg, "criterion_met": ok}


# ----------------------------------------------------------------------------
# Stage 1: dry listing -- no computation, no writes
# ----------------------------------------------------------------------------
def show_list():
    env = environment()
    print("GapTrust full-set recovery - dry listing (no computation)")
    print("\nEnvironment")
    for k, v in env.items():
        print(f"  {k:12s} {v}")

    print("\nFrozen modules (SHA-256, to prove they are unmodified)")
    for rel in FROZEN_MODULES:
        p = style.CODE_ROOT / rel
        print(f"  {rel:28s} {'missing' if not p.exists() else sha256_file(p)}")

    print("\nFrozen scientific constants")
    print(f"  V_MIG           = {V_MIG} m/ns")
    print(f"  R_FAIL          = {R_FAIL} (-6.02 dB)")
    print(f"  R_SAFE          = {R_SAFE} (-1.94 dB)")
    print(f"  MAX_CENSORED    = {MAX_CENSORED} per method")
    print(f"  target box      = +-{BOX_HX} m lateral, +-{BOX_HZ} m depth")
    print(f"  search window   = +-{SEARCH} m around the true roof")
    print(f"  METHODS         = {METHODS}")
    print(f"  FREQS           = {FREQS}")
    print(f"  MAX_GAP_M       = {MAX_GAP_M} m")
    for f in FREQS:
        print(f"  {f:8s} gap widths = {len(widths_for(f))} "
              f"({widths_for(f)[0]}..{widths_for(f)[-1]} traces)")
    print(f"  analyse_case    = {analyse_case.__module__}.{analyse_case.__qualname__}")
    print(f"  cv              = {cv.__module__}.{cv.__qualname__}")

    print("\nThe 24 analysis inputs")
    keys, missing = [], []
    for freq, v in cases():
        p = input_path(freq, v)
        key = case_key(freq, v)
        keys.append(key)
        if not p.exists():
            missing.append(str(p))
            print(f"  {key:32s} MISSING {p}")
            continue
        print(f"  {key:32s} {p.stat().st_size/1e6:6.2f} MB  {sha256_file(p)}")

    print(f"\n  cases listed     {len(keys)}   duplicates "
          f"{len(keys)-len(set(keys))}   missing inputs {len(missing)}")
    total_expected = sum(len(widths_for(f)) * len(METHODS) for f, v in cases())
    print(f"  rows per case    {len(widths_for('250MHz'))*len(METHODS)} "
          f"(41 widths x 3 methods)")
    print(f"  total rows       {total_expected}")

    print("\nProposed outputs (must all be new)")
    for p in SEMANTIC + (CACHE_DIR,):
        print(f"  {'EXISTS' if p.exists() else 'new   '}  {p}")

    if missing:
        print("\nWARNING: missing inputs - the run would fail.")
        return 1
    if any(p.exists() for p in SEMANTIC):
        print("\nWARNING: an output already exists - refusing to proceed.")
        return 1
    return 0


# ----------------------------------------------------------------------------
# Stage 2: one case, cache only
# ----------------------------------------------------------------------------
def find_case(spec):
    if ":" not in spec:
        raise SystemExit("--case expects <freq>:<model>, e.g. 400MHz:d0p75_r0p15_clay")
    freq, name = spec.split(":", 1)
    for f, v in cases():
        if f == freq and models.model_name(v) == name:
            return f, v
    raise SystemExit(f"unknown case {spec}")


def per_method_points(rows):
    out = {}
    for m in METHODS:
        sub = [r for r in rows if r["method"] == m]
        W50, G50 = first_below(sub, R_FAIL)
        W80, G80 = first_below(sub, R_SAFE)
        out[m] = {"W50": W50, "G50": G50, "W80": W80, "G80": G80,
                  "min_R": min(r["R"] for r in sub),
                  "G_max_tested": max(r["G"] for r in sub)}
    return out


def run_case(spec):
    freq, void = find_case(spec)
    p = input_path(freq, void)
    if not p.exists():
        raise SystemExit(f"missing input {p}")
    for q in SEMANTIC:
        if q.exists():
            raise SystemExit(f"refusing to run: {q} already exists")

    print(f"Case {case_key(freq, void)}")
    print(f"  input   {p}")
    print(f"  sha256  {sha256_file(p)}")
    print(f"  spec    {void}")

    t0 = time.perf_counter()
    rows, info = analyse_case(freq, void.roof_depth_m, void.radius_m, void.fill)
    runtime = time.perf_counter() - t0

    # Repeatability: same function, same input, second evaluation in-process.
    t1 = time.perf_counter()
    rows2, info2 = analyse_case(freq, void.roof_depth_m, void.radius_m, void.fill)
    runtime2 = time.perf_counter() - t1
    identical = rows == rows2
    max_dr = max(abs(a["R"] - b["R"]) for a, b in zip(rows, rows2))

    print(f"\n  rows              {len(rows)} (expected {len(widths_for(freq))*3})")
    print(f"  columns           {list(rows[0])}")
    print(f"  target (x, z) m   {info['target_x_m']:.3f}, {info['target_z_m']:.3f}")
    print(f"  D                 {info['D_m']:.4f} m")
    print(f"  runtime           {runtime:.1f} s  (repeat {runtime2:.1f} s)")
    print(f"  repeatable        {identical}  max |dR| = {max_dr}")

    pts = per_method_points(rows)
    print("\n  method   W50(m)  G50    W80(m)  G80    min R   G_max_tested")
    for m in METHODS:
        d = pts[m]
        fmt = lambda x: "censored" if x is None else f"{x:6.2f}"
        print(f"  {m:8s} {fmt(d['W50']):>8s} {fmt(d['G50']):>7s} "
              f"{fmt(d['W80']):>8s} {fmt(d['G80']):>7s} "
              f"{d['min_R']:7.3f} {d['G_max_tested']:8.3f}")

    print("\n  R at the smallest and largest gap (sanity)")
    for m in METHODS:
        sub = [r for r in rows if r["method"] == m]
        first, last = sub[0], sub[-1]
        print(f"    {m:8s} n={first['gap_traces']:2d} R={first['R']:.3f}   "
              f"n={last['gap_traces']:2d} R={last['R']:.3f}")

    # Intermediate cache, written atomically and verified by re-reading.
    rec = {
        "case": case_key(freq, void),
        "input": str(p),
        "input_sha256": sha256_file(p),
        "input_size": p.stat().st_size,
        "info": info,
        "rows": rows,
        "runtime_s": round(runtime, 2),
        "versions": environment(),
        "frozen": {"V_MIG": V_MIG, "R_FAIL": R_FAIL, "R_SAFE": R_SAFE,
                   "MAX_CENSORED": MAX_CENSORED, "box_m": [BOX_HX, BOX_HZ],
                   "search_m": SEARCH},
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cp = cache_path(freq, void)
    atomic_write_text(cp, json.dumps(rec, indent=2), overwrite=True)
    back = json.loads(cp.read_text(encoding="utf-8"))
    ok = (back["case"] == rec["case"]
          and back["input_sha256"] == rec["input_sha256"]
          and len(back["rows"]) == len(rows)
          and back["rows"] == rows)
    print(f"\n  cache             {cp}")
    print(f"  cache bytes       {cp.stat().st_size}")
    print(f"  cache sha256      {sha256_file(cp)}")
    print(f"  cache round-trip  {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# Full run: remaining cases + final outputs
# ----------------------------------------------------------------------------
def load_or_compute(freq, void, log):
    p = input_path(freq, void)
    sha = sha256_file(p)
    cp = cache_path(freq, void)
    if cp.exists():
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
            if (rec.get("input_sha256") == sha
                    and len(rec.get("rows", [])) == len(widths_for(freq)) * 3):
                log(f"cache {case_key(freq, void)}")
                return rec["rows"], rec["info"], rec.get("runtime_s"), True
        except (json.JSONDecodeError, KeyError):
            pass
    t0 = time.perf_counter()
    rows, info = analyse_case(freq, void.roof_depth_m, void.radius_m, void.fill)
    runtime = time.perf_counter() - t0
    rec = {
        "case": case_key(freq, void),
        "input": str(p),
        "input_sha256": sha,
        "input_size": p.stat().st_size,
        "info": info,
        "rows": rows,
        "runtime_s": round(runtime, 1),
        "versions": environment(),
        "frozen": {"V_MIG": V_MIG, "R_FAIL": R_FAIL, "R_SAFE": R_SAFE,
                   "MAX_CENSORED": MAX_CENSORED, "box_m": [BOX_HX, BOX_HZ],
                   "search_m": SEARCH},
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(cp, json.dumps(rec, indent=2), overwrite=True)
    log(f"done  {case_key(freq, void)}  {runtime:.1f} s")
    return rows, info, rec["runtime_s"], False


def run_all():
    for q in SEMANTIC:
        if q.exists():
            raise SystemExit(f"refusing to run: {q} already exists")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    lines = []

    def log(msg):
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        lines.append(line)
        print(line, flush=True)

    log("full_analysis_recovery.py --run")
    for k, v in environment().items():
        log(f"env {k} = {v}")

    all_rows, infos, runtimes = [], {}, {}
    for freq, void in cases():
        rows, info, rt, cached = load_or_compute(freq, void, log)
        key = case_key(freq, void)
        for r in rows:
            r.update({"radius_m": void.radius_m, "fill": void.fill})
        all_rows += rows
        infos[key] = dict(info, radius_m=void.radius_m, fill=void.fill,
                          roof_m=void.roof_depth_m, freq=freq)
        runtimes[key] = rt

    write_csv(FULL_R, all_rows, list(all_rows[0]))

    per_case = {}
    for key, info in infos.items():
        per_case[key] = {}
        for m in METHODS:
            sub = [r for r in all_rows if r["method"] == m
                   and r["freq"] == info["freq"]
                   and r["radius_m"] == info["radius_m"]
                   and r["fill"] == info["fill"]
                   and abs(r["roof_m"] - info["roof_m"]) < 1e-6]
            W50, G50 = first_below(sub, R_FAIL)
            W80, G80 = first_below(sub, R_SAFE)
            per_case[key][m] = {"W50": W50, "G50": G50, "W80": W80, "G80": G80,
                                "G_max_tested": max(r["G"] for r in sub)}

    tests = {}
    for radius in models.VOID_RADII:
        for m in METHODS:
            cmap = {k: (per_case[k][m]["W50"], per_case[k][m]["G50"])
                    for k, i in infos.items() if i["radius_m"] == radius}
            tests[f"r{radius:.2f}_{m}"] = collapse_test(cmap)
    h1 = all(tests[f"r0.15_{m}"]["criterion_met"] for m in ("linear", "pocs"))
    h2 = not all(tests[f"r0.40_{m}"]["criterion_met"] for m in ("linear", "pocs"))

    g80 = [per_case[k][m]["G80"] for k, i in infos.items() if i["radius_m"] == 0.15
           for m in ("linear", "pocs") if per_case[k][m]["G80"] is not None]
    g50 = [per_case[k][m]["G50"] for k, i in infos.items() if i["radius_m"] == 0.15
           for m in ("linear", "pocs") if per_case[k][m]["G50"] is not None]
    thresholds = {"G_constrained_below": float(np.min(g80)) if g80 else None,
                  "G_unsupported_from": float(np.median(g50)) if g50 else None,
                  "n_G80": len(g80), "n_G50": len(g50)}

    summary = {"R_fail": R_FAIL, "R_safe": R_SAFE, "v_mig_m_per_ns": V_MIG,
               "max_censored": MAX_CENSORED, "cases": infos,
               "per_case": per_case, "tests": tests,
               "H1_compact_collapse": h1, "H2_large_no_collapse": h2,
               "thresholds": thresholds}
    atomic_write_text(FULL_S, json.dumps(summary, indent=2))

    fp_rows = []
    for key, info in infos.items():
        for m in METHODS:
            d = per_case[key][m]
            fp_rows.append({
                "case": key, "freq": info["freq"],
                "model": models.model_name(models.VoidSpec(
                    info["roof_m"], info["radius_m"], info["fill"])),
                "roof_m": info["roof_m"], "radius_m": info["radius_m"],
                "fill": info["fill"], "method": m,
                "W50_m": d["W50"], "G50": d["G50"],
                "W80_m": d["W80"], "G80": d["G80"],
                "G_max_tested": d["G_max_tested"],
                "censored": "yes" if d["W50"] is None else "no",
            })
    write_csv(FAILPTS, fp_rows, list(fp_rows[0]))

    manifest = {
        "environment": environment(),
        "frozen": {"V_MIG": V_MIG, "R_FAIL": R_FAIL, "R_SAFE": R_SAFE,
                   "MAX_CENSORED": MAX_CENSORED, "MAX_GAP_M": MAX_GAP_M,
                   "box_m": [BOX_HX, BOX_HZ], "search_m": SEARCH,
                   "methods": list(METHODS), "freqs": list(FREQS)},
        "frozen_module_sha256": {
            rel: (sha256_file(style.CODE_ROOT / rel)
                  if (style.CODE_ROOT / rel).exists() else "missing")
            for rel in FROZEN_MODULES},
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [{"case": case_key(f, v), "path": str(input_path(f, v)),
                    "bytes": input_path(f, v).stat().st_size,
                    "sha256": sha256_file(input_path(f, v))}
                   for f, v in cases()],
        "runtimes_s": runtimes,
        "rows": {"full_R": len(all_rows), "failure_points": len(fp_rows),
                 "cases": len(infos), "per_case_cells": len(per_case) * len(METHODS)},
        "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in (FULL_R, FULL_S, FAILPTS)},
    }
    atomic_write_text(MANIFEST, json.dumps(manifest, indent=2))

    log(f"rows {len(all_rows)}; cases {len(infos)}; H1 {h1}; H2 {h2}")
    log(f"thresholds {thresholds}")
    for k, v in tests.items():
        log(f"test {k}: n={v['n_cases']} censored={len(v['censored'])} "
            f"CV_W50={v['CV_W50']:.3f} CV_G50={v['CV_G50']:.3f} met={v['criterion_met']}")
    log(f"outputs {[str(p) for p in (FULL_R, FULL_S, FAILPTS, MANIFEST)]}")
    atomic_write_text(LOG, "\n".join(lines) + "\n")

    print(json.dumps({"tests": tests, "H1": h1, "H2": h2,
                      "thresholds": thresholds}, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="dry listing, no computation")
    ap.add_argument("--case", help="single case as <freq>:<model>")
    ap.add_argument("--run", action="store_true", help="all 24 cases")
    args = ap.parse_args()
    if args.list:
        return show_list()
    if args.case:
        return run_case(args.case)
    if args.run:
        return run_all()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())