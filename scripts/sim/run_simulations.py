"""Run gprMax B-scans for a set of GapTrust models (resumable).

Each model is written by gaptrust.models.write_gprmax, run trace by trace with
gprMax, merged into one B-scan and saved as results/sims/<freq>/<name>.npz
together with its specification. Models that already have an .npz are skipped,
so the script can be stopped and restarted.

Usage (from the Codes folder):
    python scripts/run_simulations.py --set pilot
    python scripts/run_simulations.py --set size    (test b: r = 0.15 m, 250 MHz)
    python scripts/run_simulations.py --set full
    python scripts/run_simulations.py --set rlam    (r/lambda test: 12 models, 250 MHz, air)
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, models, style  # noqa: E402

SIM_ROOT = style.CODE_ROOT / "results" / "sims"
WORK_ROOT = style.CODE_ROOT / "results" / "work"

# r/lambda test (ledger section 14b): four radii x three roof depths, 250 MHz, air.
RLAM_RADII = (0.10, 0.21, 0.28, 0.34)
RLAM_DEPTHS = (0.75, 1.50, 2.50)


def model_list(which):
    """Return [(freq_key, void or None), ...] for the requested set."""
    if which == "pilot":
        voids = [models.VoidSpec(d, 0.40, "air") for d in (0.75, 2.50)]
        freqs = ("250MHz", "400MHz")
    elif which == "size":
        voids = [models.VoidSpec(d, 0.15, "air") for d in (0.75, 2.50)]
        return [("250MHz", v) for v in voids]
    elif which == "full":
        voids = models.void_set()
        freqs = ("250MHz", "400MHz")
    elif which == "rlam":
        voids = [models.VoidSpec(d, r, "air")
                 for r in RLAM_RADII for d in RLAM_DEPTHS]
        return [("250MHz", v) for v in voids]
    else:
        raise ValueError(which)
    return [(f, v) for f in freqs for v in [None] + voids]


def run_one(freq_key, void):
    host = models.HostSpec()
    sim = models.SIMS[freq_key]
    name = models.model_name(void)
    out = SIM_ROOT / freq_key / f"{name}.npz"
    if out.exists():
        print(f"skip {freq_key}/{name}", flush=True)
        return
    work = WORK_ROOT / freq_key / name
    if work.exists():
        shutil.rmtree(work)
    infile, n_traces = models.write_gprmax(host, sim, void, work)

    t0 = time.time()
    subprocess.run([sys.executable, "-m", "gprMax", infile.name, "-n", str(n_traces)],
                   cwd=work, check=True, stdout=subprocess.DEVNULL)
    traces, dt = [], None
    for i in range(1, n_traces + 1):
        with h5py.File(work / f"{infile.stem}{i}.h5", "r") as f:
            traces.append(f["rxs/rx1/Ez"][:])
            dt = f.attrs["dt"]
    data = np.stack(traces, axis=1).astype(np.float32)   # (nt, ntraces)
    x = sim.profile_start_m + sim.trace_spacing_m * np.arange(n_traces)
    spec = models.spec_dict(host, sim, void)
    spec["name"] = name
    spec["gprMax_runtime_s"] = round(time.time() - t0, 1)
    spec["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec["gaptrust_version"] = __version__
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=data, dt_ns=dt * 1e9, x_m=x,
                        spec=json.dumps(spec))
    shutil.rmtree(work)
    print(f"done {freq_key}/{name}: {data.shape}, {spec['gprMax_runtime_s']} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pilot", choices=("pilot", "size", "full", "rlam"))
    ap.add_argument("--list", action="store_true",
                    help="list the models of the chosen set and exit without simulating")
    args = ap.parse_args()
    cases = model_list(args.set)
    if args.list:
        for freq_key, void in cases:
            name = models.model_name(void)
            out = SIM_ROOT / freq_key / (name + ".npz")
            status = "exists" if out.exists() else "new"
            print(freq_key + "/" + name + "  " + status)
        print(str(len(cases)) + " cases")
        return
    for freq_key, void in cases:
        run_one(freq_key, void)


if __name__ == "__main__":
    main()
