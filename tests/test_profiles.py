"""
Tests for VerticalIntegrator: the integral over the two other dimensions must be the
mean over the area of the grid. Run from the repository root with

    python -m pytest tests
"""
import numpy as np
import xarray as xr

from pysammos.post_averaging.profiles import VerticalIntegrator


def grid_dataset(field):
    """Dataset of a regular grid with both ends of every range (as the grid generation makes it)."""
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 2.0, 3)
    z = np.linspace(0.0, 0.5, 6)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pos = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return xr.Dataset({"positions": (("point", "xyz"), pos), "f": (("point",), field(pos))},
                      coords={"xyz": [0, 1, 2]})


def profile(field):
    out = VerticalIntegrator(grid_dataset(field), "y").integration()
    return out["f"].values


def test_uniform_field():
    assert np.allclose(profile(lambda p: np.full(len(p), 3.0)), 3.0, rtol=1e-12)


def test_linear_field():
    # mean of x over [0, 1] and of z over [0, 0.5]
    assert np.allclose(profile(lambda p: p[:, 0] + 4.0 * p[:, 2]), 0.5 + 1.0, rtol=1e-12)


def test_periodic_field():
    # one period in x with both faces on the grid (x = 0 and x = 1 are the same point): mean zero
    values = profile(lambda p: 1.0 + np.cos(2.0 * np.pi * p[:, 0]) * np.sin(2.0 * np.pi * p[:, 2] / 0.5))
    assert np.allclose(values, 1.0, rtol=1e-12)
