"""
Reference values for the tests: the weight functions of spatial_weights.kernels, vectorised, and the
average of a weight function along a branch, taken over the part of the branch within c and split where
the branch passes closest to the grid point, with 24 Gauss-Legendre points on each piece. The weight
function is smooth on each piece, so the average is right to round-off.
"""
import math

import numpy as np

from pysammos.spatial_weights import kernels

KERNEL_FUNCTIONS = {"Lucy": kernels.lucy, "Gaussian": kernels.gaussian, "HeavySide": kernels.heavySide}


def kernel_values(name, c, d):
    """The weight function `name` of spatial_weights.kernels at the distances d."""
    d = np.asarray(d, dtype=np.float64)
    if name == "Lucy":
        u = d / c
        return np.where(d > c, 0.0, 105.0 / (16.0 * np.pi * c**3) * (1 - 6 * u**2 + 8 * u**3 - 3 * u**4))
    if name == "HeavySide":
        return np.where(d < c, 1.0 / (4.0 / 3.0 * np.pi * c**3), 0.0)
    w = c / 3.0
    Vw = (2 * np.sqrt(2) * w**3 * np.pi**1.5 * math.erf(c * np.sqrt(2) / (2 * w))
          - 4 * c * w**2 * np.pi * np.exp(-c**2 / (2 * w**2)))
    return np.where(d > c, 0.0, np.exp(-d**2 / (2 * w**2)) / Vw)


def branch_average(r0, B, c, name):
    """Average of the weight function `name` along r0 + t B, 0 <= t <= 1, for each row of r0 and B
    (grid point from the particle centre, and branch vector)."""
    a = np.einsum("ij,ij->i", B, B)
    h = np.einsum("ij,ij->i", r0, B)
    q = np.einsum("ij,ij->i", r0, r0) - c * c
    a_safe = np.where(a > 0, a, 1.0)
    root = np.sqrt(np.maximum(h * h - a * q, 0.0))
    crosses = (a > 0) & (h * h - a * q > 0)
    lo = np.where(crosses, np.clip((-h - root) / a_safe, 0.0, 1.0), 0.0)
    hi = np.where(crosses, np.clip((-h + root) / a_safe, 0.0, 1.0), np.where((a == 0) & (q <= 0), 1.0, 0.0))
    middle = np.clip(-h / a_safe, lo, hi)
    x, w = np.polynomial.legendre.leggauss(24)
    total = np.zeros(len(r0))
    for start, end in ((lo, middle), (middle, hi)):
        t = start[:, np.newaxis] + (end - start)[:, np.newaxis] * 0.5 * (x + 1.0)[np.newaxis, :]
        d = np.linalg.norm(r0[:, np.newaxis, :] + t[:, :, np.newaxis] * B[:, np.newaxis, :], axis=2)
        total += kernel_values(name, c, d) @ w * 0.5 * (end - start)
    return total
