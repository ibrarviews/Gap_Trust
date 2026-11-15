"""Figure 2 - Simulation models.

(a) Relative permittivity of the shared host (400 MHz grid, 6 mm cells) with
    the outlines of all 12 void configurations and the survey profile.
(b) Conductivity of one example model (clay-filled void, roof 1.5 m, r 0.4 m).
(c) Source wavelets (Ricker) at 250 and 400 MHz.
(d) Normalised amplitude spectra of the wavelets.

The arrays plotted here are produced by gaptrust.models.build, the same
function that writes the gprMax geometry, so the figure shows exactly what is
simulated.

Run from the Codes folder:  python scripts/fig02_models.py
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, models, physics, style  # noqa: E402

host = models.HostSpec()
sim = models.SIMS["400MHz"]
example = models.VoidSpec(roof_depth_m=1.50, radius_m=0.40, fill="clay")

_, eps, _, x, z = models.build(host, sim.dx_m, None)
_, _, sig_ex, _, _ = models.build(host, sim.dx_m, example)

style.apply()
C = style.COLORS
fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL, 138 * style.MM),
                         constrained_layout=True, height_ratios=[1.3, 0.85])
ax_a, ax_b, ax_c, ax_d = axes.ravel()
extent = [0, host.width_m, host.depth_m, 0]

# (a) permittivity of the shared host --------------------------------------------
cmap = plt.get_cmap("cividis").copy()
im = ax_a.imshow(eps, cmap=cmap, vmin=5.5, vmax=12.5, extent=extent, aspect="equal",
                 interpolation="none")
for d in models.VOID_DEPTHS:
    for r in models.VOID_RADII:
        ax_a.add_patch(Circle((2.2, d + r), r, fill=False, ec="white", lw=0.7,
                              ls="-" if r > 0.2 else "--"))
    ax_a.text(2.2 + 0.47, d + 0.05, f"{d:.2f} m", color="white", fontsize=6.5, va="top")
ax_a.plot([sim.profile_start_m, sim.profile_end_m], [0.03, 0.03], color=C["orange"], lw=2.0,
          solid_capstyle="butt")
ax_a.text(sim.profile_start_m + 0.02, 0.10, "profile: 134 traces, 0.03 m",
          color="black", fontsize=6.5, va="top")
ax_a.text(0.05, 0.62, "soil", color="black", fontsize=7, va="center",
          bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.8))
ax_a.text(0.05, 3.35, "limestone", color="black", fontsize=7, va="center",
          bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.8))
cb = fig.colorbar(im, ax=ax_a, pad=0.02, shrink=0.85)
cb.set_label(r"Relative permittivity $\varepsilon_r$")
cb.ax.tick_params(width=0.6, length=2.5)
ax_a.set_xlabel("Distance (m)")
ax_a.set_ylabel("Depth (m)")

# (b) conductivity of an example model ------------------------------------------------
im_b = ax_b.imshow(sig_ex * 1000, cmap="magma_r", extent=extent, aspect="equal",
                   interpolation="none", vmin=0, vmax=50)
cb_b = fig.colorbar(im_b, ax=ax_b, pad=0.02, shrink=0.85)
cb_b.set_label(r"$\sigma$ (mS/m)")
cb_b.ax.tick_params(width=0.6, length=2.5)
ax_b.set_xlabel("Distance (m)")
ax_b.set_ylabel("Depth (m)")
ax_b.text(0.05, 3.45, "clay-filled void, roof 1.5 m, r = 0.4 m", fontsize=6.5, va="bottom")

# (c) wavelets and (d) spectra --------------------------------------------------------------
t = np.linspace(-8, 8, 4001)
freq = np.linspace(0, 1.2, 1201)
for key, color in (("250MHz", C["blue"]), ("400MHz", C["vermillion"])):
    fk = models.SIMS[key].frequency_GHz
    w = physics.ricker(t, fk)
    ax_c.plot(t, w, color=color, label=f"{fk * 1000:.0f} MHz")
    # Analytical Ricker amplitude spectrum: |W(f)| ~ (f/fp)^2 exp(-(f/fp)^2)
    spec = (freq / fk) ** 2 * np.exp(-(freq / fk) ** 2)
    ax_d.plot(freq * 1000, spec / spec.max(), color=color, label=f"{fk * 1000:.0f} MHz")
ax_c.set_xlim(-6, 6)
ax_c.set_xlabel("Time (ns)")
ax_c.set_ylabel("Normalised amplitude")
ax_c.legend(frameon=False, loc="upper right")
ax_d.set_xlim(0, 1200)
ax_d.set_ylim(0, 1.05)
ax_d.set_xlabel("Frequency (MHz)")
ax_d.set_ylabel("Normalised amplitude")
ax_d.legend(frameon=False, loc="upper right")

for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
    style.panel_label(ax, label)

pdf, tif = style.save(fig, "Fig02_models")

# ----------------------------------------------------------------------------
# Run log
# ----------------------------------------------------------------------------
log = {
    "figure": "Fig02_models",
    "script": "scripts/fig02_models.py",
    "gaptrust_version": __version__,
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
    "host": models.spec_dict(host),
    "simulations": {k: vars(s) for k, s in models.SIMS.items()},
    "void_depths_m": models.VOID_DEPTHS,
    "void_radii_m": models.VOID_RADII,
    "void_fills": {k: models.FILLS[k] for k in models.VOID_FILLS},
    "n_models_per_frequency": len(models.void_set()) + 1,
    "example_void": vars(example),
    "limestone_eps_stats_below_0p6m": {
        "mean": float(eps[(z > 0.6)[:, None].repeat(len(x), 1)].mean()),
        "std": float(eps[(z > 0.6)[:, None].repeat(len(x), 1)].std())},
    "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                str(tif.relative_to(style.FIG_ROOT.parent))],
}
log_path = style.CODE_ROOT / "results" / "logs" / "fig02_models.json"
log_path.write_text(json.dumps(log, indent=2))
print(json.dumps(log["limestone_eps_stats_below_0p6m"]))
