"""
Tests for periodic boundaries in the grid-particle search (pysammos.neighbour_search.periodic),
using the package's own search, displacement, kernel and branch-vector functions.

Run from the repository root:   python -m pytest tests/test_periodic.py -v
"""
import warnings

import numpy as np
import pytest

from pysammos.neighbour_search.periodic import resolve_simulation_box, check_particles_in_box, periodic_images
from pysammos.neighbour_search.grid_particle_search import particle_node_match, calc_displacement
from pysammos.spatial_weights import kernels
from pysammos.spatial_weights.hashtable_search import make_hash_table, hash_table_search
from pysammos.data_handle.contacts.complete import branch_vectors

BOX_LO = np.array([0.0, 0.0, 0.0])
BOX_HI = np.array([0.10, 0.20, 0.05])
L = BOX_HI - BOX_LO
C = 0.012                                  # kernel cutoff


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def search(grid, positions, periodic, c=C):
    """The search as in CoarseGraining._assign_particles_to_grid_nodes (with images)."""
    pos_ext, original = periodic_images(positions, BOX_LO, BOX_HI, periodic, c)
    start, ext_ind = particle_node_match(grid, pos_ext, c)
    r_ri, dist = calc_displacement(grid, pos_ext, start, ext_ind)
    return start, original[ext_ind].astype(np.int32), r_ri, dist


def pairs(start, part_ind, r_ri):
    """{(node, particle): displacement} from the CSR-like output of the search."""
    out = {}
    for g in range(len(start) - 2):
        for k in range(start[g], start[g + 1]):
            out[(g, int(part_ind[k]))] = r_ri[k]
    return out


def brute_force_pairs(grid, positions, periodic, c=C):
    """All (node, particle) within c using the minimum-image displacement, by brute force."""
    out = {}
    for g, node in enumerate(grid):
        d = node - positions.astype(np.float64)
        for i in range(3):
            if periodic[i]:
                d[:, i] -= L[i] * np.round(d[:, i] / L[i])
        for p in np.nonzero(np.linalg.norm(d, axis=1) <= c)[0]:
            out[(g, int(p))] = d[p]
    return out


def random_particles(n, seed=0):
    rng = np.random.default_rng(seed)
    return (BOX_LO + rng.random((n, 3)) * L).astype(np.float32)


def grid_points(n_per_axis=(9, 15, 5), inset=0.0):
    axes = [np.linspace(lo + inset, hi - inset, n) for lo, hi, n in zip(BOX_LO, BOX_HI, n_per_axis)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)


def volume_fraction(grid, positions, volume, periodic, c=C):
    """phi(node) = sum_i W(r_i) V_i with the Lucy kernel, as in pysammos."""
    start, part_ind, _, dist = search(grid, positions, periodic, c)
    table, step = make_hash_table(kernels.lucy, c, sensitivity=1000)
    w = hash_table_search(dist, table, step)
    phi = np.zeros(len(grid))
    for g in range(len(grid)):
        phi[g] = np.sum(w[start[g]:start[g + 1]] * volume[part_ind[start[g]:start[g + 1]]])
    return phi


# ---------------------------------------------------------------------------------------------
# periodic_images
# ---------------------------------------------------------------------------------------------
def test_no_periodic_axis_returns_input_unchanged():
    pos = random_particles(50)
    ext, original = periodic_images(pos, BOX_LO, BOX_HI, [False, False, False], C)
    assert ext is pos
    np.testing.assert_array_equal(original, np.arange(50))


def test_images_are_shifted_copies_near_the_faces():
    pos = random_particles(2000, seed=1)
    periodic = np.array([True, False, True])
    ext, original = periodic_images(pos, BOX_LO, BOX_HI, periodic, C)
    n = len(pos)
    np.testing.assert_array_equal(ext[:n], pos)                     # originals first, unchanged
    np.testing.assert_array_equal(original[:n], np.arange(n))
    shift = (ext[n:].astype(np.float64) - pos[original[n:]]) / L    # whole box lengths, periodic axes only
    np.testing.assert_allclose(shift, np.round(shift), atol=1e-5)
    assert np.all(np.round(shift[:, 1]) == 0)                       # y not periodic
    assert np.all(np.abs(np.round(shift)) <= 1)
    assert np.all((ext[n:] >= BOX_LO - C - 1e-6) & (ext[n:] <= BOX_HI + C + 1e-6))
    assert np.all(np.any((ext[n:] < BOX_LO) | (ext[n:] > BOX_HI), axis=1))  # every image is outside the box
    assert ext.dtype == pos.dtype and ext.flags["C_CONTIGUOUS"]


def test_corner_particle_has_three_images():
    pos = np.array([[0.001, 0.1, 0.001]], dtype=np.float32)       # near the x = 0 and z = 0 faces
    ext, original = periodic_images(pos, BOX_LO, BOX_HI, [True, False, True], C)
    images = ext[1:].astype(np.float64)
    expected = pos.astype(np.float64) + np.array([[L[0], 0, 0], [0, 0, L[2]], [L[0], 0, L[2]]])
    assert len(images) == 3
    for e in expected:
        assert np.any(np.all(np.isclose(images, e, atol=1e-6), axis=1)), e
    np.testing.assert_array_equal(original, [0, 0, 0, 0])


def test_cutoff_larger_than_box_is_rejected():
    with pytest.raises(ValueError):
        periodic_images(random_particles(5), BOX_LO, BOX_HI, [False, False, True], margin=0.06)


# ---------------------------------------------------------------------------------------------
# simulation_box and check_particles_in_box
# ---------------------------------------------------------------------------------------------
def test_simulation_box_from_user_ranges():
    lo, hi = resolve_simulation_box([(0.0, 0.1), None, (0.0, 0.05)], [True, False, True])
    np.testing.assert_array_equal(lo, [0.0, -np.inf, 0.0])
    np.testing.assert_array_equal(hi, [0.1, np.inf, 0.05])


def test_simulation_box_missing_periodic_axis_raises_and_names_it():
    with pytest.raises(ValueError, match="z_range"):
        resolve_simulation_box([(0.0, 0.1), None, None], [True, False, True])
    with pytest.raises(ValueError, match="x_range"):
        resolve_simulation_box(None, [True, False, False])


def test_simulation_box_not_needed_without_periodic_axes():
    lo, hi = resolve_simulation_box(None, [False, False, False])
    assert np.all(np.isinf(lo)) and np.all(np.isinf(hi))


def test_simulation_box_rejects_inverted_limits():
    with pytest.raises(ValueError):
        resolve_simulation_box([(0.1, 0.0), None, None], [True, False, False])


def test_config_simulation_box_section():
    import configparser
    from pysammos.utils.config_loader import parse_simulation_box
    config = configparser.ConfigParser()
    config.read_string("[simulation_box]\nx_range = -0.1, 0.1\ny_range = [0.0, 0.2]\nz_range = None\n")
    assert parse_simulation_box(config) == [(-0.1, 0.1), (0.0, 0.2), None]
    assert parse_simulation_box(configparser.ConfigParser()) is None          # no section: None


def test_particles_outside_the_periodic_box_warn():
    lo, hi = resolve_simulation_box([(0.0, 0.1), None, None], [True, False, False])
    inside = np.array([[0.001, 0.099], [0, 1], [0, 1]])
    outside = np.array([[0.001, 0.12], [0, 1], [0, 1]])          # box too short in x
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_particles_in_box(inside, lo, hi, [True, False, False], tolerance=0.005)
    with pytest.warns(RuntimeWarning, match="x_range"):
        check_particles_in_box(outside, lo, hi, [True, False, False], tolerance=0.005)


def test_images_with_open_non_periodic_axes():
    """Infinite limits on non-periodic axes (not given by the user) must not give NaN images."""
    lo, hi = resolve_simulation_box([(0.0, 0.1), None, None], [True, False, False])
    pos = np.array([[0.001, 5.0, -3.0], [0.05, 0.0, 0.0]], dtype=np.float32)
    ext, original = periodic_images(pos, lo, hi, [True, False, False], C)
    assert np.isfinite(ext).all()
    np.testing.assert_allclose(ext[2], [0.101, 5.0, -3.0], atol=1e-6)
    np.testing.assert_array_equal(original, [0, 1, 0])


# ---------------------------------------------------------------------------------------------
# search with images = minimum-image brute force
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("periodic", [(True, False, False), (True, False, True), (True, True, True)])
def test_search_matches_minimum_image_brute_force(periodic):
    """Same (node, particle) pairs and displacements as a brute-force minimum-image search,
    including nodes on the faces and in the corners."""
    pos = random_particles(1500, seed=2)
    grid = grid_points()
    start, part_ind, r_ri, _ = search(grid, pos, np.array(periodic))
    got = pairs(start, part_ind, r_ri)
    ref = brute_force_pairs(grid, pos, periodic)
    assert set(got) == set(ref)
    for key in ref:
        np.testing.assert_allclose(got[key], ref[key], atol=1e-6)


def test_no_periodic_axis_identical_to_current_search():
    pos = random_particles(800, seed=3)
    grid = grid_points()
    start_old, ind_old = particle_node_match(grid, pos, C)
    r_old, d_old = calc_displacement(grid, pos, start_old, ind_old)
    start_new, ind_new, r_new, d_new = search(grid, pos, np.array([False, False, False]))
    for a, b in ((start_old, start_new), (ind_old, ind_new), (r_old, r_new), (d_old, d_new)):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------------------------
# coarse-grained fields
# ---------------------------------------------------------------------------------------------
def lattice(spacing=0.005):
    """Simple cubic lattice that tiles the periodic box exactly."""
    axes = [np.arange(lo + spacing / 2, hi, spacing) for lo, hi in zip(BOX_LO, BOX_HI)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)


def test_lattice_volume_fraction_is_uniform_up_to_the_faces():
    """Fully periodic lattice: phi is the same at every node, including nodes on the faces.
    Without images the face nodes lose about half of their neighbours."""
    pos = lattice()
    volume = np.full(len(pos), 1.0)
    grid = grid_points((11, 11, 6))      # nodes on the faces and corners, spaced by lattice multiples
    periodic = np.array([True, True, True])
    phi = volume_fraction(grid, pos, volume, periodic)
    assert phi.std() / phi.mean() < 1e-6
    phi_no_images = volume_fraction(grid, pos, volume, np.array([False, False, False]))
    assert phi_no_images.min() < 0.6 * phi.mean()           # the problem being fixed


def test_shifting_the_system_along_a_periodic_axis_changes_nothing():
    """Shift particles and nodes by the same amount along x and wrap them into the box:
    the coarse-grained field at every node is unchanged."""
    pos = random_particles(1500, seed=4)
    volume = np.random.default_rng(5).uniform(0.5, 1.5, len(pos))
    grid = grid_points()
    periodic = np.array([True, False, True])
    a = 0.37 * L[0]

    def wrap(x):
        x = x.astype(np.float64).copy()
        x[:, 0] = BOX_LO[0] + np.mod(x[:, 0] - BOX_LO[0] + a, L[0])
        return x

    phi = volume_fraction(grid, pos, volume, periodic)
    phi_shifted = volume_fraction(wrap(grid), wrap(pos).astype(np.float32), volume, periodic)
    np.testing.assert_allclose(phi_shifted, phi, rtol=1e-4, atol=1e-9 * phi.max())


# ---------------------------------------------------------------------------------------------
# branch vectors need the box length, not the range of particle centres
# ---------------------------------------------------------------------------------------------
def test_branch_vector_across_a_periodic_face_needs_the_box_length():
    """Two touching grains (d = 5.2 mm) on either side of the x faces of a 0.1 m periodic box."""
    d = 0.0052
    r_a = np.array([[0.0026, 0.05, 0.02]])
    r_b = np.array([[0.0974, 0.05, 0.02]])
    periodic = np.array([True, False, False])
    diam = np.array([d])

    _, center_box = branch_vectors.from_diameters(r_a.copy(), r_b.copy(), diam, diam, np.array([0.1, 1.0, 1.0]), periodic)
    np.testing.assert_allclose(np.linalg.norm(center_box, axis=1), d, rtol=1e-6)

    centre_range = r_b[0, 0] - r_a[0, 0]                     # what the particle-centre bounds give
    _, center_range = branch_vectors.from_diameters(r_a.copy(), r_b.copy(), diam, diam,
                                                    np.array([centre_range, 1.0, 1.0]), periodic)
    assert np.linalg.norm(center_range, axis=1)[0] < 0.1 * d
