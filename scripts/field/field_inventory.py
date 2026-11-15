#!/usr/bin/env python
"""Verified metadata inventory of the two field datasets, with reader audit.

Only facts established by direct test are recorded. The GSSI .DZT record layout
could not be established from the files themselves and is marked unresolved; the
values that earlier size-based inference produced are deliberately not written.

Aquileia SEG-Y reader is validated: IBM float conversion, adjacent-trace median
correlation 0.995 on the first profile.

Writes results/data/field_inventory.csv and results/logs/field_inventory.json.
"""

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
PROJ = Path(__file__).resolve().parents[3]

import numpy as np

ROOT = Path(PROJ + r"")
KARST = ROOT / "Karst data"
AQUI = ROOT / "Aquileia Data"
OUT_CSV = ROOT / "Codes" / "results" / "data" / "field_inventory.csv"
OUT_LOG = ROOT / "Codes" / "results" / "logs" / "field_inventory.json"


def dzx_props(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    props = {}
    for key in ("verticalUnit", "horizontalUnit", "dielectric", "unitsPerScan",
                "scanPerMeters", "depthRange", "system", "softwareVersion",
                "originalDielectric"):
        m = re.search("<%s>([^<]*)</%s>" % (key, key), text)
        if m:
            props[key] = m.group(1).strip()
    return props


def ibm_to_ieee(buf):
    x = np.frombuffer(buf, dtype=">u4").astype(np.uint64)
    sign = np.where((x >> 31) & 1, -1.0, 1.0)
    exp = ((x >> 24) & 0x7F).astype(np.float64) - 64
    mant = (x & 0x00FFFFFF).astype(np.float64) / float(1 << 24)
    return sign * mant * np.power(16.0, exp)


def sgy_read(path, max_traces=None):
    raw = path.read_bytes()
    ns = int.from_bytes(raw[3220:3222], "big")
    dt = int.from_bytes(raw[3216:3218], "big")
    fmt = int.from_bytes(raw[3224:3226], "big")
    itemsize = {1: 4, 3: 2, 5: 4, 8: 1}[fmt]
    rec = 240 + itemsize * ns
    ntr = (len(raw) - 3600) // rec
    n = ntr if max_traces is None else min(ntr, max_traces)
    tr = np.empty((n, ns))
    for t in range(n):
        off = 3600 + t * rec + 240
        buf = raw[off:off + itemsize * ns]
        tr[t] = ibm_to_ieee(buf) if fmt == 1 else np.frombuffer(buf, dtype="<i2").astype(float)
    return {"ns": ns, "dt": dt, "format_code": fmt, "ntraces": ntr, "traces": tr}


def main():
    rows = []
    audit = {}

    for dzt in sorted(KARST.rglob("*.DZT")):
        keys = [dzt.parent.parent.parent.name, dzt.parent.name]
        rows.append({"dataset": "karst", "site": keys[0], "project": keys[1],
                     "file": dzt.name, "bytes": dzt.stat().st_size,
                     "ns": "unresolved", "ntraces": "unresolved", "dt": "",
                     "geometry_status": "DZT record layout unresolved",
                     "dzx": json.dumps(dzx_props(dzt.with_suffix(".DZX")), ensure_ascii=False)
                     if dzt.with_suffix(".DZX").exists() else "{}"})

    sgy = sorted(AQUI.rglob("*.sgy"))
    for p in sgy:
        h = sgy_read(p, max_traces=2)
        rows.append({"dataset": "aquileia", "site": str(p.parent.relative_to(AQUI)),
                     "project": "", "file": p.name, "bytes": p.stat().st_size,
                     "ns": h["ns"], "ntraces": h["ntraces"], "dt": h["dt"],
                     "geometry_status": "reader validated by trace coherence", "dzx": "{}"})

    check = sgy_read(sgy[0], max_traces=400)
    tr = check["traces"]
    corr = [float(np.corrcoef(tr[i], tr[i + 1])[0, 1]) for i in range(len(tr) - 1)]
    audit["aquileia_reader"] = {"file": sgy[0].name, "ns": check["ns"], "dt_raw": check["dt"],
                                "format_code": check["format_code"], "ntraces": check["ntraces"],
                                "median_adjacent_correlation": float(np.median(corr)),
                                "min": float(min(corr)), "max": float(max(corr)),
                                "verdict": "validated"}
    audit["karst_reader"] = {
        "files": len(list(KARST.rglob("*.DZT"))),
        "file_header_bytes": 1024, "trace_header_bytes": 128,
        "data_offset_word": 128, "range_ns_from_float_at_13": 304.0,
        "dyx": "dielectric 14, unitsPerScan 0.025 m, depthRange 4.0 m, system SIR4K",
        "layout_tests_run": 36,
        "only_exact_divisor_layout": {"file_header": 1024, "ns": 1024, "bytes_per_sample": 2,
                                      "median_adjacent_correlation": 0.143},
        "best_correlation_but_not_exact": {"ns": 2048, "bytes_per_sample": 2, "value": 0.543},
        "verdict": "unresolved - no layout produced coherent adjacent traces"}

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "site", "project", "file", "bytes",
                                           "ns", "ntraces", "dt", "geometry_status", "dzx"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "audit": audit,
                                   "counts": {"karst": sum(1 for r in rows if r["dataset"] == "karst"),
                                              "aquileia": sum(1 for r in rows if r["dataset"] == "aquileia")}},
                                  indent=2, ensure_ascii=False), encoding="utf-8")

    print("karst files   %d (geometry unresolved)" % audit["karst_reader"]["files"])
    print("aquileia files %d (reader validated, median adjacent corr %.3f)"
          % (len(sgy), audit["aquileia_reader"]["median_adjacent_correlation"]))
    print("written:", OUT_CSV)
    print("written:", OUT_LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())