# -*- coding: utf-8 -*-
"""Read the three GSSI SIR-4000 karst lines with readgssi and cache them as .npz.

Run in the readgssi environment (Python 3.13, readgssi 0.0.22, setuptools<80).
`readdzt()` returns (header, {0: ndarray(samples, traces)}, gps); note that the
array order in its docstring is the reverse, so the array is transposed here.

Each cached file holds traces x samples plus the printable header fields, and the
audit written to results/logs/field_read_audit.json records shape, amplitude range
and the median adjacent-trace correlation (about 1.000 for these lines, which is
why G was counted in traces rather than distance).

Existing caches are kept unless --force is given: the cached arrays are inputs to
published results, and overwriting them silently would break those numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from readgssi import readgssi

HERE = Path(__file__).resolve()
CODE = HERE.parents[2]            # repository root, holds gaptrust and results
ROOT = CODE.parent                # raw data sit beside the repository
KARST = ROOT / "Karst data"
OUT = CODE / "results" / "data" / "field_cache"
AUDIT = CODE / "results" / "logs" / "field_read_audit.json"

TARGETS = {
    "great_l14": "PROJECT700__001.DZT",
    "small_l6": "PROJECD000__001.DZT",
    "dugout_l2": "PROJECG000__001.DZT",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    for label, name in TARGETS.items():
        cache = OUT / ("%s.npz" % label)
        if cache.exists() and not args.force:
            print("%-10s cache exists, kept (%s)" % (label, cache.name))
            continue
        p = sorted(KARST.rglob(name))[0]
        hdr, arr, gps = readgssi.readdzt(str(p))
        a = np.asarray(arr[0], dtype=np.float64)      # samples x traces
        traces = a.T                                   # traces x samples
        c = [np.corrcoef(traces[i], traces[i + 1])[0, 1]
             for i in range(min(300, traces.shape[0]) - 1)
             if traces[i].std() > 0 and traces[i + 1].std() > 0]
        keys = {k: v for k, v in hdr.items() if isinstance(v, (int, float, str))}
        info[label] = {
            "file": p.name, "shape_traces_samples": list(traces.shape),
            "amp_min": float(traces.min()), "amp_max": float(traces.max()),
            "median_adjacent_correlation": float(np.median(c)),
            "antenna": hdr.get("rh_antname"), "antfreq": hdr.get("antfreq"),
            "samples_per_trace": hdr.get("rh_nsamp"), "bits": hdr.get("rh_bits"),
            "time_range_ns": hdr.get("rhf_range"), "spt_ns": hdr.get("rhf_spt"),
            "spm": hdr.get("rhf_spm"), "sps": hdr.get("rhf_sps"),
            "dielectric": hdr.get("rhf_dielectric"),
        }
        np.savez_compressed(cache, data=traces,
                            header=json.dumps({k: str(v) for k, v in keys.items()}))
        print("%-10s %s  shape %s  amp %.0f..%.0f  corr %.3f"
              % (label, p.name, traces.shape, traces.min(), traces.max(), np.median(c)))

    if info:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print("audit:", AUDIT)


if __name__ == "__main__":
    main()
