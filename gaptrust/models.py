"""Synthetic karst models for the gprMax simulations.

One geometry array is built per model and used twice: written to HDF5 for
gprMax (#geometry_objects_read) and plotted in Figure 2, so the figure shows
exactly what was simulated.

Model (2-D, x = distance, z = depth below the ground surface):
  - soil cover over limestone, with an undulating soil/rock interface;
  - limestone permittivity varies as a correlated Gaussian random field,
    quantised into discrete materials (gprMax needs discrete materials);
  - an optional circular void (2-D cross-section of a conduit) filled with
    air or wet clay;
  - one fixed random seed, so every void model shares the same host and the
    no-void model is its exact twin.
"""

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------------
# Material properties: relative permittivity, conductivity (S/m)
# ----------------------------------------------------------------------------
SOIL = (12.0, 0.010)            # red-clay soil cover
LIMESTONE_MEAN = 8.0            # background limestone permittivity
LIMESTONE_STD = 0.8             # standard deviation of the random field
LIMESTONE_SIGMA = 0.002
FILLS = {"air": (1.0, 0.0), "clay": (25.0, 0.050)}


@dataclass
class HostSpec:
    """Host medium shared by all models at one frequency."""
    width_m: float = 4.4
    depth_m: float = 3.6               # ground thickness inside the domain
    air_m: float = 0.3                 # air layer above ground
    soil_thickness_m: float = 0.40
    interface_amp_m: float = 0.10      # undulation amplitude of soil/rock
    interface_wavelength_m: float = 1.6
    corr_length_m: float = 0.20        # correlation length of heterogeneity
    n_eps_bins: int = 9                # discrete limestone materials
    seed: int = 20261106


@dataclass
class VoidSpec:
    roof_depth_m: float
    radius_m: float
    fill: str                          # "air" or "clay"
    x_m: float = 2.2


@dataclass
class SimSpec:
    frequency_GHz: float
    dx_m: float
    time_window_ns: float
    trace_spacing_m: float = 0.03
    profile_start_m: float = 0.20
    profile_end_m: float = 4.20
    antenna_offset_m: float = 0.10     # transmitter-receiver separation
    antenna_height_m: float = 0.02


SIMS = {
    "400MHz": SimSpec(frequency_GHz=0.40, dx_m=0.006, time_window_ns=80.0),
    "250MHz": SimSpec(frequency_GHz=0.25, dx_m=0.010, time_window_ns=110.0),
}

VOID_DEPTHS = (0.75, 1.50, 2.50)
VOID_RADII = (0.15, 0.40)
VOID_FILLS = ("air", "clay")


def void_set():
    """All void configurations (12): depth x radius x fill."""
    return [VoidSpec(d, r, fl) for d, r, fl in product(VOID_DEPTHS, VOID_RADII, VOID_FILLS)]


def model_name(void):
    if void is None:
        return "novoid"
    return f"d{void.roof_depth_m:.2f}_r{void.radius_m:.2f}_{void.fill}".replace(".", "p")


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def material_table(host):
    """Ordered list of (name, eps_r, sigma). Index in list = material index."""
    edges = np.linspace(LIMESTONE_MEAN - 2 * LIMESTONE_STD,
                        LIMESTONE_MEAN + 2 * LIMESTONE_STD, host.n_eps_bins)
    mats = [("soil", *SOIL)]
    mats += [(f"limestone{i}", float(e), LIMESTONE_SIGMA) for i, e in enumerate(edges)]
    mats += [(f"void_{k}", *v) for k, v in FILLS.items()]
    return mats


def build(host, dx, void=None):
    """Return (index_grid, eps_grid, sigma_grid, x, z) for one model.

    Grids have shape (nz, nx) with z = 0 at the ground surface, increasing
    downward. Only the ground is returned; the air layer is gprMax free space.
    """
    nx = int(round(host.width_m / dx))
    nz = int(round(host.depth_m / dx))
    x = (np.arange(nx) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    X, Z = np.meshgrid(x, z)
    mats = material_table(host)
    names = [m[0] for m in mats]

    # Correlated random field, generated on a fixed 1 cm grid so that the
    # same host is obtained at every model resolution, then resampled.
    rng = np.random.default_rng(host.seed)
    ref_dx = 0.01
    nxr, nzr = int(round(host.width_m / ref_dx)), int(round(host.depth_m / ref_dx))
    field = gaussian_filter(rng.standard_normal((nzr, nxr)),
                            host.corr_length_m / ref_dx / 2.0, mode="wrap")
    field /= field.std()
    iz = np.minimum((z / ref_dx).astype(int), nzr - 1)
    ix = np.minimum((x / ref_dx).astype(int), nxr - 1)
    field = field[np.ix_(iz, ix)]

    eps_lime = LIMESTONE_MEAN + LIMESTONE_STD * field
    lime_eps = np.array([m[1] for m in mats if m[0].startswith("limestone")])
    lime_idx = np.argmin(np.abs(eps_lime[..., None] - lime_eps), axis=-1)
    index = (lime_idx + names.index("limestone0")).astype(np.int16)

    # Soil cover with an undulating base.
    phase = rng.uniform(0, 2 * np.pi)
    base = host.soil_thickness_m + host.interface_amp_m * np.sin(
        2 * np.pi * x / host.interface_wavelength_m + phase)
    index[Z < base[None, :]] = names.index("soil")

    if void is not None:
        zc = void.roof_depth_m + void.radius_m
        inside = (X - void.x_m) ** 2 + (Z - zc) ** 2 <= void.radius_m ** 2
        index[inside] = names.index(f"void_{void.fill}")

    eps = np.array([m[1] for m in mats])[index]
    sig = np.array([m[2] for m in mats])[index]
    return index, eps, sig, x, z


# ----------------------------------------------------------------------------
# gprMax input
# ----------------------------------------------------------------------------
def write_gprmax(host, sim, void, out_dir):
    """Write HDF5 geometry, materials file and .in file for one B-scan model.

    Returns the path of the .in file and the number of traces.
    """
    import h5py

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = model_name(void)
    dx = sim.dx_m
    index, *_ = build(host, dx, void)
    mats = material_table(host)

    # gprMax array order is (x, y, z) with y upward; our grid is (z down, x).
    data = np.flipud(index).T[:, :, None].astype(np.int16)
    h5 = out_dir / f"{name}_geom.h5"
    with h5py.File(h5, "w") as f:
        f.attrs["dx_dy_dz"] = (dx, dx, dx)
        f.create_dataset("data", data=data)
    mfile = out_dir / f"{name}_materials.txt"
    mfile.write_text("".join(f"#material: {e:g} {s:g} 1 0 {n}\n" for n, e, s in mats))

    # gprMax 4 reads keyed HDF5 + a JSON material database; convert the pair
    # with gprMax's own tool so the material assignment is done by gprMax.
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "gprMax.toolboxes.MaterialDatabase",
                    "convert-geometry", h5.name, mfile.name],
                   cwd=out_dir, check=True, capture_output=True)
    h5_conv = f"{name}_geom_converted.h5"
    database = f"{name}_geom_materials"

    ground_top = host.depth_m
    y_ant = ground_top + sim.antenna_height_m
    n_traces = int(round((sim.profile_end_m - sim.profile_start_m) / sim.trace_spacing_m)) + 1
    tx = sim.profile_start_m - sim.antenna_offset_m / 2
    rx = sim.profile_start_m + sim.antenna_offset_m / 2
    lines = [
        f"#title: GapTrust {name} {sim.frequency_GHz * 1000:.0f} MHz",
        f"#domain: {host.width_m:g} {host.depth_m + host.air_m:g} {dx:g}",
        f"#dx_dy_dz: {dx:g} {dx:g} {dx:g}",
        f"#time_window: {sim.time_window_ns * 1e-9:g}",
        f"#geometry_objects_read: 0 0 0 {h5_conv} {database}",
        f"#waveform: ricker 1 {sim.frequency_GHz * 1e9:g} src",
        f"#hertzian_dipole: z {tx:.4f} {y_ant:.4f} 0 src",
        f"#rx: {rx:.4f} {y_ant:.4f} 0",
        f"#src_steps: {sim.trace_spacing_m:g} 0 0",
        f"#rx_steps: {sim.trace_spacing_m:g} 0 0",
    ]
    infile = out_dir / f"{name}.in"
    infile.write_text("\n".join(lines) + "\n")
    return infile, n_traces


def spec_dict(host, sim=None, void=None):
    d = {"host": asdict(host)}
    if sim is not None:
        d["sim"] = asdict(sim)
    if void is not None:
        d["void"] = asdict(void)
    d["materials"] = material_table(host)
    return d
