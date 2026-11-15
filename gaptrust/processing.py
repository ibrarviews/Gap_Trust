"""Processing chain shared by the simulation and field analyses.

All B-scans are arrays of shape (nt, ntraces) with time step dt (ns) and
trace positions x (m). Gaps are boolean masks over traces (True = missing).

Steps
-----
time_zero       shift so that the direct-wave peak sits at t = 0
remove_mean     subtract the mean of the *recorded* traces only
fill_zero       leave missing traces as zeros
fill_linear     linear interpolation across the gap, sample by sample
fill_pocs       f-k projection onto convex sets with exponentially
                decreasing threshold
migrate         constant-velocity zero-offset Kirchhoff (diffraction-sum)
envelope_z      Hilbert envelope of a migrated image along depth
"""

import numpy as np
from scipy.signal import hilbert


# ----------------------------------------------------------------------------
# Gaps
# ----------------------------------------------------------------------------
def centred_gap(ntraces, centre, width_traces):
    """Mask of `width_traces` missing traces centred on trace index `centre`."""
    mask = np.zeros(ntraces, dtype=bool)
    if width_traces <= 0:
        return mask
    start = int(centre - (width_traces - 1) // 2)
    mask[max(start, 0):min(start + width_traces, ntraces)] = True
    return mask


# ----------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------
def time_zero(data, dt, search_ns=10.0):
    """Shift traces so the direct-wave peak (max |mean trace|) is at t = 0."""
    n = int(search_ns / dt)
    i0 = int(np.argmax(np.abs(data[:n].mean(axis=1))))
    return data[i0:], i0 * dt


def remove_mean(data, missing=None):
    """Subtract the mean trace computed from recorded traces only."""
    rec = np.ones(data.shape[1], bool) if missing is None else ~missing
    return data - data[:, rec].mean(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------------
def fill_zero(data, missing):
    out = data.copy()
    out[:, missing] = 0.0
    return out


def fill_linear(data, missing):
    """Linear interpolation across missing traces, sample by sample."""
    out = data.copy()
    if not missing.any():
        return out
    idx = np.arange(data.shape[1])
    rec = ~missing
    for k in range(data.shape[0]):
        out[k, missing] = np.interp(idx[missing], idx[rec], data[k, rec])
    return out


def fill_pocs(data, missing, n_iter=100, p_max=0.99, p_min=1e-3):
    """f-k POCS interpolation.

    Iterates: 2-D FFT -> hard threshold (threshold decays exponentially from
    p_max to p_min of the maximum spectral amplitude of the input) -> inverse
    FFT -> re-insert recorded traces.
    """
    rec = ~missing
    obs = fill_zero(data, missing)
    out = obs.copy()
    amax = np.abs(np.fft.fft2(obs)).max()
    lam = p_max * amax * (p_min / p_max) ** (np.arange(n_iter) / max(n_iter - 1, 1))
    for th in lam:
        F = np.fft.fft2(out)
        F[np.abs(F) < th] = 0.0
        out = np.real(np.fft.ifft2(F))
        out[:, rec] = obs[:, rec]
    return out


FILLERS = {"zero": fill_zero, "linear": fill_linear, "pocs": fill_pocs}


# ----------------------------------------------------------------------------
# Migration
# ----------------------------------------------------------------------------
def migrate(data, dt, x, v, z):
    """Zero-offset constant-velocity Kirchhoff migration (diffraction sum).

    Uses the time derivative of the data, an obliquity factor z/r and
    1/sqrt(r) spreading compensation. Returns an image of shape (len(z), len(x)).
    """
    nt = data.shape[0]
    d = np.gradient(data, dt, axis=0)
    xi = x[:, None]          # image position
    xj = x[None, :]          # trace position
    cols = np.arange(len(x))[None, :]
    img = np.zeros((len(z), len(x)))
    for iz, zz in enumerate(z):
        r = np.sqrt(zz ** 2 + (xi - xj) ** 2) + 1e-6
        it = 2.0 * r / v / dt
        i0 = np.floor(it).astype(int)
        frac = it - i0
        ok = i0 < nt - 1
        i0c = np.clip(i0, 0, nt - 2)
        val = (1 - frac) * d[i0c, cols] + frac * d[i0c + 1, cols]
        w = (zz / r) / np.sqrt(r)
        img[iz] = np.sum(np.where(ok, val * w, 0.0), axis=1)
    return img


def envelope_z(img):
    return np.abs(hilbert(img, axis=0))
