"""Physical relations used by GapTrust.

Units throughout: metres (m), nanoseconds (ns), gigahertz (GHz), so that
velocity is in m/ns and wavelength = v / f is in m.

Aperture definition
-------------------
For a point diffractor at depth d below a zero-offset common-offset profile,
the two-way time at lateral offset x is

    t(x) = (2 / v) * sqrt(d**2 + x**2).

The diffraction aperture D is the lateral width over which the two-way delay
t(x) - t(0) stays below half a dominant period, 1 / (2 f). Solving
sqrt(d**2 + x**2) - d = v / (4 f) = lambda / 4 gives

    D = 2 * sqrt(lambda * d / 2 + lambda**2 / 16)
      = sqrt(2 * lambda * d + lambda**2 / 4),

which reduces to the first-Fresnel-zone width sqrt(2 lambda d) for d >> lambda.

Gap number
----------
    G = W / D,

where W is the gap width. Because W and D can both be counted in traces,
G is independent of the trace spacing.
"""

import numpy as np


def wavelength(v, f):
    """Dominant wavelength (m) for velocity v (m/ns) and frequency f (GHz)."""
    return v / f


def aperture(d, v, f):
    """Diffraction aperture D (m) at depth d (m); exact half-period definition."""
    lam = wavelength(v, f)
    return np.sqrt(2.0 * lam * d + lam ** 2 / 4.0)


def gap_number(w, d, v, f):
    """Gap number G = W / D for gap width w (m) over a diffractor at depth d (m)."""
    return w / aperture(d, v, f)


def ricker(t, f):
    """Ricker wavelet with peak frequency f (GHz) evaluated at times t (ns)."""
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def diffraction_bscan(x, t, x0, d, v, f, spreading=True):
    """Analytical zero-offset B-scan of a single point diffractor.

    Parameters
    ----------
    x : 1-D array of trace positions (m).
    t : 1-D array of two-way times (ns).
    x0, d : diffractor lateral position and depth (m).
    v : velocity (m/ns).  f : Ricker peak frequency (GHz).
    spreading : apply 1/r amplitude decay normalised to the apex.

    Returns
    -------
    data : array of shape (len(t), len(x)).
    """
    r = np.sqrt(d ** 2 + (x - x0) ** 2)
    tt = 2.0 * r / v
    amp = d / r if spreading else np.ones_like(r)
    return amp[None, :] * ricker(t[:, None] - tt[None, :], f)
