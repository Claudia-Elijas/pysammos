"""
Tests for the look-up table of the weight functions. Run from the repository root with

    python -m pytest tests

The references are the weight functions of spatial_weights.kernels evaluated directly.
"""
import numpy as np
import pytest

from pysammos.spatial_weights import kernels
from pysammos.spatial_weights.hashtable_search import make_hash_table, hash_table_search

KERNELS = {"Lucy": kernels.lucy, "Gaussian": kernels.gaussian, "HeavySide": kernels.heavySide}


def weight(name, c, d):
    return np.vectorize(KERNELS[name])(c, np.asarray(d, dtype=np.float64))


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("c", [1.0e-3, 1.513e-3, 5.302e-3, 9.9e-3])
def test_table_integrates_to_one(name, c):
    """The weight function looked up in the table integrates to 1 over its ball."""
    table, step = make_hash_table(KERNELS[name], c, sensitivity=1000)
    n = 200000
    r = (np.arange(n) + 0.5) * c / n  # midpoint rule in r, far finer than the table
    shell = 4.0 * np.pi * r**2 * (c / n)
    exact = np.sum(weight(name, c, r) * shell)
    looked_up = np.sum(hash_table_search(r, table, step) * shell)
    assert abs(exact - 1.0) < 1e-6
    assert abs(looked_up - 1.0) < 1e-5


@pytest.mark.parametrize("name", ["Lucy", "Gaussian"])
@pytest.mark.parametrize("c", [1.0e-3, 1.513e-3, 9.9e-3])
def test_table_lookup_accuracy(name, c):
    """Single look-ups: second order in the table step, below the bin that holds c; in that bin
    the truncated Gaussian, steepest there, is within a tenth of the step over c."""
    table, step = make_hash_table(KERNELS[name], c, sensitivity=1000)
    r = np.linspace(0.0, c, 100001)[:-1]
    err = np.abs(hash_table_search(r, table, step) - weight(name, c, r)) / weight(name, c, 0.0)
    last = r >= (len(table) - 3) * step  # the bin that holds c
    assert err[~last].max() < 2e-6
    if last.any():  # empty when the bin that holds c starts within rounding of c
        assert err[last].max() < (2e-6 if name == "Lucy" else 0.11 * step / c)


@pytest.mark.parametrize("name", list(KERNELS))
def test_table_beyond_cutoff_is_zero(name):
    """The zero at the end of the table, returned from one step past the bin that holds c."""
    c = 1.513e-3
    table, step = make_hash_table(KERNELS[name], c, sensitivity=1000)
    assert table[-1] == 0.0
    beyond = np.array([c + 2 * step, 2 * c, 10 * c])
    assert np.all(hash_table_search(beyond, table, step) == 0.0)


def test_table_lookup_outside_the_table():
    """Negative values get the first entry, values past the table and infinity the last (zero),
    NaN gets NaN; in 1D and 2D alike."""
    c = 1.513e-3
    table, step = make_hash_table(KERNELS["Lucy"], c, sensitivity=1000)
    q = np.array([-1.0, -np.inf, np.inf, 1e300, np.nan])
    expected = np.array([table[0], table[0], 0.0, 0.0, np.nan])
    np.testing.assert_array_equal(hash_table_search(q, table, step), expected)
    np.testing.assert_array_equal(hash_table_search(q.reshape(1, -1), table, step), expected.reshape(1, -1))
