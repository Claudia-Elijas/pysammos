"""
Tests for the contact data (ids, repeated pairs, missing particles), the searches
across periodic boundaries, and the coarse-grained fields of a small synthetic bed
in a box periodic in x and y against a brute-force evaluation of their definitions.
Run from the repository root with

    python -m pytest tests

No DEM data are needed: the particle and contact files are written by the tests
in the MFIX layout (vtkXMLPolyDataWriter, one point per particle or contact).
"""
import configparser

import h5py
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from pysammos.coarse_graining import CoarseGraining
from pysammos.data_handle.contacts.particle_mapper import map_contact_data
from pysammos.data_handle.contacts.qualitycheck import duplicates
from pysammos.data_read.dispatcher import dispatch_read
from pysammos.neighbour_search.grid_particle_search import particle_node_match, calc_displacement, branch_node_match
from pysammos.utils.config_loader import optional_float

from reference_weights import KERNEL_FUNCTIONS, kernel_values, branch_average

KEYMAP = dict(Global_ID="Particle_ID", Particle_Velocity="Velocity", Particle_Diameter="Diameter",
              Particle_Density="Density", Particle_Volume="Volume", Particle_Mass="Mass", Particle_Radius=None,
              Coordination_Number=None, Particle_i_ID="Particle_ID_1", Particle_j_ID="Particle_ID_2",
              Force_ij="FORCE_CHAIN_FC", Contact_ij="FORCE_CHAIN_CONTACT_POINT")


def write_points(path, points, arrays):
    """PolyData file with one vertex per point and the given point data."""
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for i in range(len(points)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    for name, a in arrays.items():
        va = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
        va.SetName(name)
        poly.GetPointData().AddArray(va)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToAscii()
    writer.Write()


def write_step(prefix_p, prefix_c, t, ids, x, v, d, rho, i1, i2, F, cp):
    vol = np.pi / 6.0 * d**3
    write_points(f"{prefix_p}{t:04d}.vtp", x, dict(Particle_ID=ids, Velocity=v, Diameter=d, Density=rho,
                                                   Volume=vol, Mass=rho * vol))
    mid = 0.5 * (x[np.searchsorted(ids, i1)] + x[np.searchsorted(ids, i2)]) if len(i1) else np.zeros((0, 3))
    write_points(f"{prefix_c}{t:04d}.vtp", mid, dict(Particle_ID_1=i1, Particle_ID_2=i2, FORCE_CHAIN_FC=F,
                                                     FORCE_CHAIN_CONTACT_POINT=cp))


def test_mfix_ids_above_float32_range(tmp_path):
    """Ids above 2**24 come back exact, so contacts find their particles."""
    ids = np.array([2**24 + 1, 2**24 + 2, 2**24 + 3], dtype=np.int64)
    x = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [2e-3, 0.0, 0.0]])
    write_step(f"{tmp_path}/p_", f"{tmp_path}/c_", 0, ids, x, np.zeros((3, 3)), np.full(3, 1e-3), np.full(3, 2500.0),
               ids[[0, 1]], ids[[1, 2]], np.ones((2, 3)), np.zeros((2, 3)))
    particles, contacts = dispatch_read("mfix", KEYMAP, f"{tmp_path}/p_", f"{tmp_path}/c_", 0, 4)
    assert np.array_equal(particles[1], ids)
    assert np.array_equal(contacts[0], ids[[0, 1]]) and np.array_equal(contacts[1], ids[[1, 2]])


def test_repeated_pairs(capsys):
    """A repeated pair is a duplicate only for the same contact: contact points within the tolerance (either
    order, or a periodic image), or within the larger tolerance with the same or the opposite force."""
    LL = np.array([1, 1, 2, 3, 1, 2, 1])
    II = np.array([2, 2, 1, 4, 2, 1, 2])
    points = np.array([[0.0, 0.0, 0.0],      # pair (1, 2), first contact
                       [5e-4, 0.0, 0.0],     # pair (1, 2), a second contact at another point: kept
                       [2e-5, 0.0, 0.0],     # reversed copy of the first, 2e-5 away (about two overlaps): removed
                       [0.0, 0.0, 0.0],      # pair (3, 4): kept
                       [1e-2, 0.0, 0.0],     # the first contact one box length away: removed
                       [5e-5, 0.0, 0.0],     # a third contact near the first, with another force: kept
                       [9.6e-3, 0.0, 0.0]])  # a contact 0.4 mm from the first across the face, another force: kept
    forces = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [-1.0, -2.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0],
                       [1.0, 2.5, 0.0], [0.0, 0.0, 4.0]])
    known = np.array([True, True, False])
    rule = dict(tolerance=1e-6, tolerance_with_forces=1e-4, box_lengths=np.array([1e-2, 1e-2, 0.0]),
                periodic=np.array([True, True, False]))
    keep = duplicates.get_unique_pairs(LL, II, points, Forces=forces, box_known=known, **rule)
    assert keep.tolist() == [0, 1, 3, 5, 6]
    # a box length that may be an estimate: the component across the face is not compared, which merges row 6
    assert duplicates.get_unique_pairs(LL, II, points, Forces=forces, **rule).tolist() == [0, 1, 3, 5]
    # without forces, by contact point only
    assert duplicates.get_unique_pairs(LL, II, points, box_known=known, **rule).tolist() == [0, 1, 2, 3, 5, 6]
    assert duplicates.get_unique_pairs(LL, II).tolist() == [0, 3]  # without contact points: first row of each pair
    out = duplicates.delete(LL, II, forces, points, box_known=known, **rule)
    assert out[0].tolist() == LL[[0, 1, 3, 5, 6]].tolist() and np.array_equal(out[2], forces[[0, 1, 3, 5, 6]])
    printed = capsys.readouterr().out
    assert "in contact data:  2" in printed and "forces):  3" in printed


def test_contact_with_missing_particle():
    """A contact whose particle is missing is dropped from every contact array, which stay aligned."""
    gid = np.array([1, 2, 3, 5], dtype=np.int64)
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    ones = np.ones(4, dtype=np.float32)
    i1 = np.array([1, 2, 3, 3], dtype=np.int64)
    i2 = np.array([2, 4, 5, 9], dtype=np.int64)  # ids 4 and 9 are not in the particle data, 9 above the largest
    F = np.array([[1.0, 0.0, 0.0], [9.0, 9.0, 9.0], [3.0, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    cp = np.array([[0.5, 0.0, 0.0], [9.0, 9.0, 9.0], [2.5, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    out = map_contact_data(gid, x, ones, ones, ones, i1, i2, F, cp, ModelAxesRanges=np.zeros(3),
                           AxesPeriodicity=np.array([False, False, False]), Return_Volume=True, Particle_Phase_Array_t=None)
    position, force, branch = out[0], out[1], out[2]
    assert len(force) == 4  # two contacts, both halves
    assert np.allclose(force[:2, 0], [1.0, 3.0]) and np.allclose(force[2:, 0], [-1.0, -3.0])
    assert np.allclose(branch[:2, 0], [-0.5, -0.5]) and np.allclose(branch[2:, 0], [0.5, 0.5])


def box_length_owner(dem_software, **lengths):
    """A CoarseGraining with only the attributes that _periodic_box_lengths reads."""
    cg = CoarseGraining.__new__(CoarseGraining)
    cg.DEM_software = dem_software
    cg.grid_info = dict(x_axis_periodic=True, y_axis_periodic=True, z_axis_periodic=False,
                        x_box_length=None, y_box_length=None, z_box_length=None)
    cg.grid_info.update(lengths)
    return cg


def test_periodic_box_lengths(capsys):
    """Box lengths as given; for MFIX data without them the particle extent, with one warning per axis, which
    only wraps branches (no length for the searches); the box of a LAMMPS dump; a length that is not a positive
    number stops the run."""
    ranges = np.array([0.0195, 0.0098, 0.005])
    cg = box_length_owner("mfix")
    cg.grid_info["x_box_length"] = 0.02
    box, search = cg._periodic_box_lengths(ranges)
    assert box.tolist() == [0.02, 0.0098, 0.0] and search.tolist() == [0.02, 0.0, 0.0]
    cg._periodic_box_lengths(ranges)
    assert capsys.readouterr().out.count("WARNING") == 1
    box, search = box_length_owner("lammps")._periodic_box_lengths(ranges)
    assert box.tolist() == search.tolist() == [0.0195, 0.0098, 0.0] and "WARNING" not in capsys.readouterr().out
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            box_length_owner("mfix", **{"y_box_length": bad})._periodic_box_lengths(ranges)


def test_optional_float():
    config = configparser.ConfigParser()
    config.read_string("[grid_info]\na = 0.034\nb = None\nc = none\nd =\ne = abc\n")
    section = config["grid_info"]
    assert optional_float(section, "a") == 0.034
    assert [optional_float(section, k) for k in ("b", "c", "d", "missing")] == [None] * 4
    with pytest.raises(ValueError, match="e must be a number"):
        optional_float(section, "e")


def brute_force_pairs(nodes, points, c, box):
    pairs = set()
    for g, node in enumerate(nodes):
        d = node - points
        for k in range(3):
            if box[k] > 0:
                d[:, k] -= box[k] * np.rint(d[:, k] / box[k])
        for p in np.flatnonzero(np.einsum("ij,ij->i", d, d) <= c * c):
            pairs.add((g, p))
    return pairs


@pytest.mark.parametrize("box", [np.array([0.0, 0.0, 0.0]), np.array([1e-2, 8e-3, 0.0])])
def test_particle_search(box):
    rng = np.random.default_rng(1)
    nodes = np.stack(np.meshgrid(np.linspace(0, 1e-2, 6), np.linspace(0, 8e-3, 5), np.linspace(1e-3, 4e-3, 3),
                                 indexing="ij"), -1).reshape(-1, 3)
    points = (rng.uniform(0, 1, (300, 3)) * np.array([1e-2, 8e-3, 5e-3])).astype(np.float32)
    c = 1.5e-3
    start, found = particle_node_match(nodes, points, c, box)
    got = {(g, int(p)) for g in range(len(nodes)) for p in found[start[g]:start[g + 1]]}
    assert got == brute_force_pairs(nodes, points.astype(np.float64), c, box)
    r, dist = calc_displacement(nodes, points, start, found, box)
    assert np.all(dist <= c * (1 + 1e-12))


@pytest.mark.parametrize("chunk_size", [7, 20000])
@pytest.mark.parametrize("box", [np.array([0.0, 0.0, 0.0]), np.array([1e-2, 8e-3, 0.0])])
def test_branch_search(box, chunk_size):
    """Every contact half whose branch passes within c of a grid point is found, and only those, with the
    displacements of calc_displacement; searched in many chunks or in one."""
    rng = np.random.default_rng(2)
    nodes = np.stack(np.meshgrid(np.linspace(0, 1e-2, 6), np.linspace(0, 8e-3, 5), np.linspace(1e-3, 4e-3, 3),
                                 indexing="ij"), -1).reshape(-1, 3)
    centres = (rng.uniform(0, 1, (200, 3)) * np.array([1e-2, 8e-3, 5e-3])).astype(np.float32)
    branches = (rng.normal(size=(200, 3)) * 4e-4).astype(np.float32)
    c = 1.5e-3
    start, halves, r = branch_node_match(nodes, centres, branches, c, box, chunk_size=chunk_size)
    got = {(g, int(h)) for g in range(len(nodes)) for h in halves[start[g]:start[g + 1]]}
    assert np.array_equal(r, calc_displacement(nodes, centres, start, halves, box)[0])
    expected = set()
    t = np.linspace(0.0, 1.0, 2001)
    for g, node in enumerate(nodes):
        for h in range(len(centres)):
            seg = centres[h].astype(np.float64)[np.newaxis, :] - t[:, np.newaxis] * branches[h].astype(np.float64)
            dd = node - seg
            for k in range(3):
                if box[k] > 0:
                    dd[:, k] -= box[k] * np.rint(dd[:, k] / box[k])
            dmin = np.sqrt(np.min(np.einsum("ij,ij->i", dd, dd)))
            if dmin <= c * (1 - 1e-6):
                expected.add((g, h))
            elif dmin <= c * (1 + 1e-6):  # within the resolution of the sampled segment: either is right
                if (g, h) in got:
                    expected.add((g, h))
    assert got == expected
    for g in range(len(nodes)):  # halves of each grid point in increasing order
        assert np.all(np.diff(halves[start[g]:start[g + 1]]) > 0)


def test_branch_search_inputs():
    """float64 positions give the float32 result, no halves give empty lists, and a grid point at the middle
    of a branch (distance zero to the candidate centre of the search) is found."""
    nodes = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [5e-3, 0.0, 0.0]])
    # half 0: middle of the branch exactly on grid point 0; half 1: centre 2.2e-3 from grid point 1, branch within 7e-4
    centres = np.array([[2.0**-12, 0.0, 0.0], [3.2e-3, 0.0, 0.0]], dtype=np.float32)
    branches = np.array([[2.0**-11, 0.0, 0.0], [1.5e-3, 0.0, 0.0]], dtype=np.float32)
    out32 = branch_node_match(nodes, centres, branches, 1.5e-3)
    out64 = branch_node_match(nodes, centres.astype(np.float64), branches, 1.5e-3)
    assert all(np.array_equal(a, b) for a, b in zip(out32, out64))
    start, halves, r = out32
    assert halves[start[0]:start[1]].tolist() == [0] and halves[start[1]:start[2]].tolist() == [0, 1]
    assert start[2] == start[3]  # grid point 2 is reached by no branch
    start, halves, r = branch_node_match(nodes, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3)), 1.5e-3)
    assert start.tolist() == [0] * (len(nodes) + 2) and len(halves) == 0 and r.shape == (0, 3)


# ----------------------------------------------------------------------------------------------
# End to end: a small synthetic bed, periodic in x and y, against brute force
L, H = 1.0e-2, 8.0e-3   # box lengths in x and y
D, RHO = 1.0e-3, 2500.0
FIELDS = dict(volume_fraction=True, density_particle=False, density_mixture=True, momentum_density=True, velocity=True,
              velocity_gradient=False, kinetic_tensor=True, contact_tensor=True, total_stress_tensor=True,
              pressure=False, granular_temperature=False, granular_temperature_slices=False, fabric_tensor=False,
              inertial_number=False, coordination_number=False, d43=False, d32=False, frictional_coefficient=False,
              shear_rate_tensor=False)


def synthetic_bed(rng):
    """Random particles in the box and random contact rows between near pairs, some across the periodic
    boundaries, with contact points off the line of centres, written next to particle 1 as MFIX-Exa does."""
    n = 60
    x = rng.uniform(0, 1, (n, 3)) * np.array([L, H, 6e-3]) + np.array([0, 0, 1e-3])
    v = rng.normal(size=(n, 3)) * 0.1
    ids = np.arange(1, n + 1, dtype=np.int64) * 7  # sparse ids
    box = np.array([L, H, 0.0])
    i1, i2, F, cp = [], [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            dx = x[b] - x[a]
            dx[:2] -= box[:2] * np.rint(dx[:2] / box[:2])
            dist = np.linalg.norm(dx)
            if dist < 1.6 * D:  # contacts and bonds apart
                nvec = dx / dist
                lateral = rng.normal(size=3) * 1e-4
                lateral -= (lateral @ nvec) * nvec
                cp.append(x[a] + nvec * 0.5 * dist + lateral)
                i1.append(ids[a])
                i2.append(ids[b])
                F.append(rng.normal(size=3) * 1e-3)
    return ids, x, v, np.array(i1), np.array(i2), np.array(F), np.array(cp)


def nearest(d):
    d = np.array(d, dtype=np.float64)
    d[..., 0] -= L * np.rint(d[..., 0] / L)
    d[..., 1] -= H * np.rint(d[..., 1] / H)
    return d


def reference_fields(nodes, x, v, m, x1, x2, F, cp, c, name):
    """Definitions of the fields, by brute force with the nearest periodic image (c < half the box)."""
    rho = np.zeros(len(nodes))
    mom = np.zeros((len(nodes), 3))
    kin = np.zeros((len(nodes), 3, 3))
    sig = np.zeros((len(nodes), 3, 3))
    halves = [(x1, F, nearest(x1 - cp)), (x2, -F, nearest(x2 - cp))]
    for g, node in enumerate(nodes):
        W = kernel_values(name, c, np.linalg.norm(nearest(node - x), axis=1))
        rho[g] = np.sum(m * W)
        mom[g] = (m * W) @ v
        if rho[g] > 0:
            dv = v - mom[g] / rho[g]
            kin[g] = np.einsum("n,na,nb->ab", m * W, dv, dv)
        for centre, force, branch in halves:
            wline = branch_average(nearest(node - centre), branch, c, name)  # grid point from the particle centre
            sig[g] += np.einsum("h,ha,hb->ab", wline, force, branch)
    return rho, mom, kin, sig


def coarse_grain(path, name):
    """Pysammos on the files written in path, periodic in x and y: grid points, fields and cutoff."""
    grid_info = dict(grid_dimension=3, grid_axes="xyz", automatic_grid=False, x_min=0.0, x_max=L, y_min=0.0,
                     y_max=H, z_min=1.5e-3, z_max=6.5e-3, x_transect=None, y_transect=None, z_transect=None,
                     x_axis_periodic=True, y_axis_periodic=True, z_axis_periodic=False,
                     x_box_length=L, y_box_length=H, z_box_length=None)
    cg = CoarseGraining(particle_path=f"{path}/particles_", contacts_path=f"{path}/contacts_",
                        output_path=f"{path}/out/", start_timestep=0, end_timestep=0, dt_time_step=1, t_pad=4,
                        DEM_software="mfix", DEM_keymap=KEYMAP, grid_info=grid_info, weight_function=name,
                        fields_to_export=FIELDS, ignore_phases=True, vkthdf_output=False, h5_output=True)
    bounds, diameter, density, mass, gid = cg.data_sampling()
    cg.get_particle_size_statistics(diameter, mass)
    cg.get_particle_phases(diameter, density, gid, 8, plot=False)
    cg.set_resolution(D)
    cg.generate_grid()
    cg.fields_in_time()
    with h5py.File(f"{path}/out/CG_{name}_{cg.cg_calc_mode}.h5", "r") as f:
        nodes = f["positions"][:]
        got = {k: np.asarray(f[k][0], dtype=np.float64) for k in ("density_mixture", "momentum_density",
                                                                    "kinetic_tensor", "contact_tensor")}
    return nodes, got, cg.c


@pytest.mark.parametrize("name", list(KERNEL_FUNCTIONS))
def test_periodic_bed_against_brute_force(tmp_path, name):
    rng = np.random.default_rng(42)
    ids, x, v, i1, i2, F, cp = synthetic_bed(rng)
    write_step(f"{tmp_path}/particles_", f"{tmp_path}/contacts_", 0, ids, x, v, np.full(len(ids), D),
               np.full(len(ids), RHO), i1, i2, F, cp)
    nodes, got, c = coarse_grain(tmp_path, name)
    x32 = x.astype(np.float32).astype(np.float64)  # positions and contact points as Pysammos holds them
    cp32 = cp.astype(np.float32).astype(np.float64)
    order = {k: i for i, k in enumerate(ids)}
    x1 = x32[[order[k] for k in i1]]
    x2 = x32[[order[k] for k in i2]]
    m = np.full(len(ids), RHO * np.pi / 6 * D**3)
    rho, mom, kin, sig = reference_fields(nodes, x32, v, m, x1, x2, F, cp32, c, name)
    assert len(i1) > 30 and np.sum(np.abs(nearest(x2 - x1) - (x2 - x1)) > 0) > 0  # some pairs cross a boundary
    for key, ref in (("density_mixture", rho), ("momentum_density", mom), ("kinetic_tensor", kin), ("contact_tensor", sig)):
        err = np.max(np.abs(got[key] - ref)) / np.max(np.abs(ref))
        assert err < 1e-5, f"{key}: largest difference {err:.2e} of the largest value"


def test_contacts_written_from_both_particles(tmp_path):
    """Each contact written a second time from its other particle (ids swapped, opposite force, contact point
    2 % of a diameter away, as when each copy is placed next to its own particle) gives the same fields as each
    contact written once."""
    rng = np.random.default_rng(7)
    ids, x, v, i1, i2, F, cp = synthetic_bed(rng)
    order = {k: i for i, k in enumerate(ids)}
    normal = nearest(x[[order[k] for k in i2]] - x[[order[k] for k in i1]])
    normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
    rows = {"once": (i1, i2, F, cp),
            "twice": (np.concatenate([i1, i2]), np.concatenate([i2, i1]), np.concatenate([F, -F]),
                      np.concatenate([cp, cp + 0.02 * D * normal]))}
    fields = {}
    for key, (a, b, f, p) in rows.items():
        path = tmp_path / key
        path.mkdir()
        write_step(f"{path}/particles_", f"{path}/contacts_", 0, ids, x, v, np.full(len(ids), D),
                   np.full(len(ids), RHO), a, b, f, p)
        fields[key] = coarse_grain(path, "Lucy")[1]
    for name in fields["once"]:
        assert np.array_equal(fields["once"][name], fields["twice"][name]), name
