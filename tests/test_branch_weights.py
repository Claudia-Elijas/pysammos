"""
Tests for the weight of a contact half: the weight function averaged along its branch,
from the particle centre to the contact point. Run from the repository root with

    python -m pytest tests

The reference is the exact average of reference_weights.py, whose weight functions are
checked here against those of spatial_weights.kernels.
"""
import numpy as np
import pytest

from pysammos.spatial_weights import kernels
from pysammos.spatial_weights.hashtable_search import make_hash_table, hash_table_search
from pysammos.spatial_weights.utils import gauss_legendre_points, compute_dist_along_branch_in_support

from reference_weights import KERNEL_FUNCTIONS, kernel_values, branch_average

KERNELS = {"Lucy": kernels.lucy, "Gaussian": kernels.gaussian, "HeavySide": kernels.heavySide}


def weight(name, c, d):
    return np.vectorize(KERNELS[name])(c, np.asarray(d, dtype=np.float64))


def test_gauss_legendre_points():
    s, w = gauss_legendre_points(10)
    assert np.all((s > 0) & (s < 1)) and abs(w.sum() - 1.0) < 1e-14
    for k in range(20):  # exact for polynomials of degree 2n - 1
        assert abs(np.sum(w * s**k) - 1.0 / (k + 1)) < 1e-13


@pytest.mark.parametrize("name", list(KERNEL_FUNCTIONS))
def test_kernel_values(name):
    """The weight functions of the reference are those of Pysammos."""
    c = 1.5e-3
    d = np.linspace(0.0, 1.2 * c, 97)
    expected = np.array([KERNEL_FUNCTIONS[name](c, x) for x in d], dtype=np.float64)
    assert np.allclose(kernel_values(name, c, d), expected, rtol=1e-12, atol=1e-12 * expected.max())


@pytest.mark.parametrize("name", list(KERNELS))
def test_branch_average(name):
    """Weights of contact halves against the exact average, branches up to 0.8 c long,
    grid points up to c plus the branch length from the particle centre."""
    rng = np.random.default_rng(5)
    c = 1.513e-3
    table, step = make_hash_table(KERNELS[name], c, sensitivity=1000)
    n = 150
    B = rng.normal(size=(n, 3))
    B *= (rng.uniform(0.0, 0.8, n) * c / np.linalg.norm(B, axis=1))[:, np.newaxis]
    B = B.astype(np.float32)
    r0 = rng.normal(size=(n, 3))
    r0 *= (rng.uniform(0.0, 1.0, n) * (c + np.linalg.norm(B, axis=1)) / np.linalg.norm(r0, axis=1))[:, np.newaxis]
    s, w = gauss_legendre_points(10)
    dist, length = compute_dist_along_branch_in_support(r0, s, B, np.arange(n, dtype=np.int32), c)
    weights = (w @ hash_table_search(dist, table, step)) * length
    reference = branch_average(r0, B.astype(np.float64), c, name)
    scale = weight(name, c, 0.0)
    assert np.max(np.abs(weights - reference)) < 5e-5 * scale
    assert abs(np.sum(weights - reference)) < 2e-6 * np.sum(np.abs(reference))


def test_branch_support_interval():
    """The part of a branch inside the support, against the exact chord."""
    c = 1.0
    s, _ = gauss_legendre_points(4)
    cases = [  # (grid point - particle centre, branch, part of [0, 1] within c)
        ([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5),     # leaves the ball at t = 0.5
        ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0),     # inside throughout
        ([2.0, 0.0, 0.0], [0.5, 0.0, 0.0], 0.0),     # outside throughout
        ([-1.5, 0.0, 0.0], [2.0, 0.0, 0.0], 0.75),   # enters at t = 0.25, still inside at t = 1
        ([0.0, 0.9, 0.0], [1.0, 0.0, 0.0], np.sqrt(0.19)),  # passes 0.9 from the grid point: inside up to t = 0.19**0.5
        ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0], 1.0),     # branch of zero length, particle centre within c
        ([0.0, 0.0, 1.5], [0.0, 0.0, 0.0], 0.0),     # branch of zero length, particle centre beyond c
    ]
    r0 = np.array([case[0] for case in cases], dtype=np.float64)
    B = np.array([case[1] for case in cases], dtype=np.float32)
    expected = np.array([case[2] for case in cases])
    _, length = compute_dist_along_branch_in_support(r0, s, B, np.arange(len(cases), dtype=np.int32), c)
    assert np.allclose(length, expected, rtol=0.0, atol=1e-12)
