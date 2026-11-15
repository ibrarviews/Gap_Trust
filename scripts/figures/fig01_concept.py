"""Figure 1 - GapTrust concept.

(a) Survey geometry: a gap of width W in a common-offset profile above a void
    whose diffraction aperture at the surface is D.
(b) Analytical zero-offset B-scan of the void with a narrow gap (G < 1).
(c) The same B-scan with a wide gap (G > 1).
(d) Gap number G = W / D as a function of void depth and gap width.

All panels are analytical illustrations computed from gaptrust.physics; no
field or simulated (FDTD) data are used in this figure.

Run from the Codes folder:  python scripts/fig01_concept.py
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gaptrust import __version__, physics, style  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters (single source of truth for this figure)
# ----------------------------------------------------------------------------
P = {
    "velocity_m_per_ns": 0.10,       # limestone, relative permittivity ~9
    "frequency_GHz": 0.40,           # 400 MHz antenna
    "void_roof_depth_m": 1.50,       # depth to the diffracting void roof
    "void_x_m": 2.00,
    "profile_length_m": 4.00,
    "trace_spacing_m": 0.02,
    "time_window_ns": 60.0,
    "time_step_ns": 0.05,
    "soil_cover_m": 0.40,
    "gap_narrow_m": 0.35,
    "gap_wide_m": 1.40,
    "display_clip": 0.6,
}

v = P["velocity_m_per_ns"]
f = P["frequency_GHz"]
d = P["void_roof_depth_m"]
x0 = P["void_x_m"]
lam = physics.wavelength(v, f)
D = physics.aperture(d, v, f)

x = np.arange(0.0, P["profile_length_m"] + 1e-9, P["trace_spacing_m"])
t = np.arange(0.0, P["time_window_ns"] + 1e-9, P["time_step_ns"])
bscan = physics.diffraction_bscan(x, t, x0, d, v, f)
bscan /= np.abs(bscan).max()


def gap_mask(width):
    """Boolean mask of traces inside a gap of given width centred on the void."""
    return np.abs(x - x0) <= width / 2.0


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
style.apply()
C = style.COLORS
fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL, 128 * style.MM),
                         constrained_layout=True)
ax_a, ax_b, ax_c, ax_d = axes.ravel()

# (a) geometry ---------------------------------------------------------------
W = P["gap_narrow_m"]
zmax = 2.3
ax_a.add_patch(Rectangle((0, 0), x[-1], P["soil_cover_m"], fc="#E8DCC4", ec="none"))
ax_a.add_patch(Rectangle((0, P["soil_cover_m"]), x[-1], zmax, fc="#E3E3E3", ec="none"))
ax_a.add_patch(Ellipse((x0, d + 0.18), 0.55, 0.36, fc="white", ec="black", lw=0.7))
ax_a.text(x0, d + 0.19, "void", ha="center", va="center", fontsize=7)
ax_a.text(0.08, P["soil_cover_m"] / 2, "soil cover", va="center", fontsize=7)
ax_a.text(0.08, P["soil_cover_m"] + 0.18, "limestone", va="center", fontsize=7)

recorded = ~gap_mask(W)
ax_a.vlines(x[recorded][::5], -0.07, 0.0, color="black", lw=0.6)
ax_a.add_patch(Rectangle((x0 - W / 2, -0.09), W, 0.11, fc=C["vermillion"],
                         ec="none", alpha=0.85))
for side in (-1, 1):
    ax_a.plot([x0, x0 + side * D / 2], [d, 0], color=C["blue"], lw=0.8, ls="--")


def bracket(ax, xa, xb, y, text, color):
    ax.annotate("", xy=(xa, y), xytext=(xb, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.8,
                                shrinkA=0, shrinkB=0))
    ax.text((xa + xb) / 2, y - 0.05, text, ha="center", va="bottom",
            color=color, fontsize=8)


bracket(ax_a, x0 - W / 2, x0 + W / 2, -0.18, "$W$", C["vermillion"])
bracket(ax_a, x0 - D / 2, x0 + D / 2, -0.50, "$D$", C["blue"])
ax_a.annotate("", xy=(x0 + 0.62, d), xytext=(x0 + 0.62, 0.0),
              arrowprops=dict(arrowstyle="<->", color="black", lw=0.6,
                              shrinkA=0, shrinkB=0))
ax_a.text(x0 + 0.67, d / 2, "$d$", va="center", fontsize=8)
ax_a.text(3.95, 1.95, r"$D=\sqrt{2\lambda d+\lambda^{2}/4}$" "\n" r"$G=W/D$",
          ha="right", va="center", fontsize=8)
ax_a.set_xlim(0, x[-1])
ax_a.set_ylim(zmax, -0.68)
ax_a.set_xlabel("Distance (m)")
ax_a.set_ylabel("Depth (m)")
ax_a.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

# (b), (c) B-scans --------------------------------------------------------------
clip = P["display_clip"]
extent = [x[0], x[-1], t[-1], t[0]]
t_apex = 2 * d / v
for ax, width in ((ax_b, P["gap_narrow_m"]), (ax_c, P["gap_wide_m"])):
    m = gap_mask(width)
    shown = np.ma.masked_array(bscan, np.broadcast_to(m, bscan.shape))
    ax.imshow(shown, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
              extent=extent, interpolation="none")
    ax.add_patch(Rectangle((x0 - width / 2, t[0]), width, t[-1] - t[0],
                           fc="none", ec=C["vermillion"], hatch="////", lw=0.0))
    for side in (-1, 1):
        ax.axvline(x0 + side * D / 2, color=C["blue"], lw=0.8, ls="--")
    G = width / D
    ax.text(0.03, 0.04, f"$G$ = {G:.1f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Two-way time (ns)")
    ax.set_xlim(x[0], x[-1])
ax_b.text(x0, 12.0, "aperture $D$", color=C["blue"], ha="center",
          fontsize=7, bbox=dict(fc="white", ec="none", pad=1.0))

# (d) G map -------------------------------------------------------------------------
depth = np.linspace(0.25, 5.0, 300)
width = np.linspace(0.05, 3.0, 300)
DD, WW = np.meshgrid(depth, width)
GG = physics.gap_number(WW, DD, v, f)
levels = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
norm = matplotlib.colors.BoundaryNorm(levels, plt.get_cmap("cividis").N, extend="max")
cf = ax_d.contourf(DD, WW, GG, levels=levels, cmap="cividis", norm=norm, extend="max")
cs = ax_d.contour(DD, WW, GG, levels=[0.25, 0.5, 1.0, 2.0], colors="white", linewidths=0.6)
label_xy = [(4.3, physics.aperture(4.3, v, f) * g) for g in (0.25, 0.5, 1.0)]
label_xy.append((3.2, physics.aperture(3.2, v, f) * 2.0))
ax_d.clabel(cs, fmt=lambda val: f"{val:g}", fontsize=6.5, manual=label_xy)
for width_case, marker in ((P["gap_narrow_m"], "o"), (P["gap_wide_m"], "s")):
    ax_d.plot(d, width_case, marker, ms=4.5, mfc=C["orange"], mec="black", mew=0.5)
ax_d.text(d + 0.12, P["gap_narrow_m"], "(b)", va="center", fontsize=7, color="white")
ax_d.text(d + 0.12, P["gap_wide_m"], "(c)", va="center", fontsize=7, color="black")
cb = fig.colorbar(cf, ax=ax_d, pad=0.02, spacing="uniform", format="%g")
cb.set_label("Gap number $G$")
cb.ax.tick_params(width=0.6, length=2.5)
ax_d.set_xlabel("Void depth $d$ (m)")
ax_d.set_ylabel("Gap width $W$ (m)")
ax_d.text(0.97, 0.04, f"{f * 1000:.0f} MHz, $v$ = {v:.2f} m/ns",
          transform=ax_d.transAxes, ha="right", fontsize=7, color="white")

for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
    style.panel_label(ax, label)

pdf, tif = style.save(fig, "Fig01_concept")

# ----------------------------------------------------------------------------
# Run log
# ----------------------------------------------------------------------------
log = {
    "figure": "Fig01_concept",
    "script": "scripts/fig01_concept.py",
    "gaptrust_version": __version__,
    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
    "parameters": P,
    "derived": {"wavelength_m": lam, "aperture_D_m": D,
                "G_narrow": P["gap_narrow_m"] / D, "G_wide": P["gap_wide_m"] / D},
    "data_source": "analytical (gaptrust.physics); no field or FDTD data",
    "outputs": [str(pdf.relative_to(style.FIG_ROOT.parent)),
                str(tif.relative_to(style.FIG_ROOT.parent))],
}
log_path = style.CODE_ROOT / "results" / "logs" / "fig01_concept.json"
log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text(json.dumps(log, indent=2))
print(json.dumps(log["derived"], indent=2))
