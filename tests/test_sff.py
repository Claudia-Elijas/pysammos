"""
Synthetic tests for pysammos.macroscopic_fields.gridded.sff (SFF friction with cross terms).

Run from the repository root:   pytest tests/test_sff.py -v
"""
import numpy as np
import pytest

from pysammos.macroscopic_fields.gridded import derived_fields as df
from pysammos.macroscopic_fields.gridded import sff
from pysammos.macroscopic_fields.field_dependencies import get_fields_to_compute
from pysammos.time_average.accumulator import TimeAverageAccumulator
from pysammos.utils.config_loader import EXPORT_FLAGS


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def sphere_quadrature(n_theta=12, n_phi=24):
    """Unit vectors and weights (summing to 1) of a quadrature that is exact for low-degree polynomials."""
    z, wz = np.polynomial.legendre.leggauss(n_theta)
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    s = np.sqrt(1.0 - zz**2)
    n = np.stack([s * np.cos(pp), s * np.sin(pp), zz], axis=-1).reshape(-1, 3)
    w = (wz[:, None] * np.ones(n_phi) / (2.0 * n_phi)).reshape(-1) / 2.0
    return n, w / w.sum()


def random_traceless(rng, norm):
    """Random symmetric traceless 3x3 tensor with the given Frobenius norm."""
    a = np.einsum("k,kab->ab", rng.normal(size=5), sff.BASIS)
    return a * norm / np.linalg.norm(a)


def model_contacts(Ac, An, At, g0=2.0):
    """Contacts that follow the SFF model exactly: unit normals, g vectors, probability weights."""
    n, w = sphere_quadrature()
    q_c = np.einsum("ij,ci,cj->c", Ac, n, n)
    q_n = np.einsum("ij,ci,cj->c", An, n, n)
    q_t = np.einsum("ij,ci,cj->c", At, n, n)
    gn = g0 * (1.0 + q_n)
    gt = g0 * (n @ At.T - q_t[:, None] * n)
    return n, gn[:, None] * n + gt, w * (1.0 + q_c)


def dev(a):
    return a - np.trace(a, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3.0


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------
def test_basis_is_orthonormal_symmetric_traceless():
    np.testing.assert_allclose(np.einsum("kab,lab->kl", sff.BASIS, sff.BASIS), np.eye(5), atol=1e-14)
    np.testing.assert_allclose(np.trace(sff.BASIS, axis1=1, axis2=2), 0.0, atol=1e-14)
    np.testing.assert_allclose(sff.BASIS, np.swapaxes(sff.BASIS, 1, 2), atol=1e-14)


def test_grid_kernel_matches_numpy_reference():
    rng = np.random.default_rng(0)
    m = 300
    normals = rng.normal(size=(m, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    c2c = normals * rng.uniform(0.5, 1.5, m)[:, None]
    force = rng.normal(size=(m, 3))
    branch = rng.normal(size=(m, 3))
    n, g = sff.contact_vectors(c2c, force, branch)
    np.testing.assert_allclose(n, normals)
    np.testing.assert_allclose(g, force * np.linalg.norm(branch, axis=1)[:, None])

    visible = [rng.choice(m, 40, replace=False), rng.choice(m, 70, replace=False), np.array([], dtype=int)]
    visibility = np.concatenate(visible).astype(np.int32)
    weights = rng.uniform(0.1, 1.0, len(visibility))
    offsets = np.cumsum([0] + [len(v) for v in visible])
    grid_indices = np.concatenate([offsets, [offsets[-1]]]).astype(np.int32)       # length Npoints + 2

    moments = sff.coarse_grain_moments(weights, visibility, grid_indices, c2c, force, branch)
    assert moments.shape == (3, sff.N_MOMENTS)
    for p in range(2):
        sel = slice(offsets[p], offsets[p + 1])
        expected = sff.moments_from_contacts(n[visible[p]], g[visible[p]], weights[sel])
        np.testing.assert_allclose(moments[p], expected, rtol=1e-12, atol=1e-12)
    assert not moments[2].any()


# ---------------------------------------------------------------------------------------------
# anisotropies
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(4))
def test_recovers_the_anisotropies_of_the_model(seed):
    rng = np.random.default_rng(seed)
    Ac, An, At = (random_traceless(rng, 0.4) for _ in range(3))
    n, g, w = model_contacts(Ac, An, At)
    ac, an, at = sff.anisotropies(sff.moments_from_contacts(n, g, w)[None])
    np.testing.assert_allclose(ac[0], Ac, atol=1e-10)
    np.testing.assert_allclose(an[0], An, atol=1e-10)
    np.testing.assert_allclose(at[0], At, atol=1e-10)


def test_fit_equals_least_squares_on_noisy_contacts():
    """The normal equations built from the moments give the least-squares fit of the contact vectors."""
    rng = np.random.default_rng(5)
    m = 800
    n = rng.normal(size=(m, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    An, At = random_traceless(rng, 0.3), random_traceless(rng, 0.2)
    gn = 1.5 * (1.0 + np.einsum("ij,ci,cj->c", An, n, n)) + 0.3 * rng.normal(size=m)
    noise = rng.normal(size=(m, 3)) * 0.2
    gt = 1.5 * (n @ At.T - np.einsum("ij,ci,cj->c", At, n, n)[:, None] * n) + noise
    gt -= np.einsum("ci,ci->c", gt, n)[:, None] * n
    g = gn[:, None] * n + gt

    # least squares written directly on the contacts, with a different (non-orthonormal) basis
    basis = np.array([np.diag([1, 0, -1]), np.diag([0, 1, -1]), [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                      [[0, 0, 1], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 1, 0]]], dtype=float)
    q = np.einsum("ci,kij,cj->ck", n, basis, n)
    coeff = np.linalg.lstsq(np.column_stack([np.ones(m), q]), gn, rcond=None)[0]
    g0 = coeff[0]
    An_ref = np.einsum("k,kij->ij", coeff[1:] / g0, basis)
    design = np.einsum("kij,cj->cik", basis, n) - n[:, :, None] * q[:, None, :]
    At_ref = np.einsum("k,kij->ij", np.linalg.lstsq(design.reshape(-1, 5), gt.ravel(), rcond=None)[0] / g0, basis)
    Ac_ref = 7.5 * dev(n.T @ n / m)

    ac, an, at = sff.anisotropies(sff.moments_from_contacts(n, g)[None])
    np.testing.assert_allclose(ac[0], Ac_ref, atol=1e-10)
    np.testing.assert_allclose(an[0], An_ref, atol=1e-10)
    np.testing.assert_allclose(at[0], At_ref, atol=1e-10)


def test_anisotropies_rotate_with_the_contacts_and_ignore_the_weight_scale():
    rng = np.random.default_rng(8)
    Ac, An, At = (random_traceless(rng, 0.3) for _ in range(3))
    n, g, w = model_contacts(Ac, An, At)
    base = sff.anisotropies(sff.moments_from_contacts(n, g, w)[None])
    R = random_rotation(rng)
    rotated = sff.anisotropies(sff.moments_from_contacts(n @ R.T, g @ R.T, w)[None])
    scaled = sff.anisotropies(sff.moments_from_contacts(n, g, 7.0 * w)[None])
    for b, r, s in zip(base, rotated, scaled):
        np.testing.assert_allclose(r[0], R @ b[0] @ R.T, atol=1e-10)
        np.testing.assert_allclose(s[0], b[0], atol=1e-10)


def test_undetermined_nodes_are_nan():
    rng = np.random.default_rng(2)
    few_n = rng.normal(size=(3, 3))
    few_n /= np.linalg.norm(few_n, axis=1)[:, None]
    few = sff.moments_from_contacts(few_n, rng.normal(size=(3, 3)))
    empty = np.zeros(sff.N_MOMENTS)
    broken = sff.moments_from_contacts(*model_contacts(*(random_traceless(rng, 0.2) for _ in range(3))))
    good = broken.copy()
    broken[3] = np.nan
    ac, an, at = sff.anisotropies(np.stack([few, empty, broken, good]))
    for field in (ac, an, at):
        assert np.isnan(field[:3]).all() and np.isfinite(field[3]).all()


# ---------------------------------------------------------------------------------------------
# cross terms
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_cross_terms_are_the_angular_integral_of_the_model(seed):
    """sigma_c / P_c from the formula equals the integral of p(n) (g_n n + g_t) (x) n over the sphere."""
    rng = np.random.default_rng(100 + seed)
    Ac, An, At = (random_traceless(rng, 0.45) for _ in range(3))
    n, g, w = model_contacts(Ac, An, At)
    sigma = np.einsum("c,ci,cj->ij", w, g, n)
    sigma = 0.5 * (sigma + sigma.T)
    exact = sigma / (np.trace(sigma) / 3.0)

    with_cross = sff.contact_stress_shape(Ac, An, At, cross_terms=True)
    linear = sff.contact_stress_shape(Ac, An, At, cross_terms=False)
    np.testing.assert_allclose(with_cross, exact, atol=1e-12)
    assert np.abs(linear - exact).max() > 1e-3, "the cross terms should matter at this anisotropy"
    assert np.trace(with_cross) == pytest.approx(3.0)


def test_cross_terms_vanish_for_one_isotropic_factor():
    rng = np.random.default_rng(3)
    A = random_traceless(rng, 0.4)
    zero = np.zeros((3, 3))
    for args in [(A, zero, zero), (zero, A, zero), (zero, zero, A)]:
        np.testing.assert_allclose(sff.contact_stress_shape(*args, cross_terms=True),
                                   sff.contact_stress_shape(*args, cross_terms=False), atol=1e-14)


def test_isotropic_contacts_give_an_isotropic_contact_stress():
    n, w = sphere_quadrature()
    g = 3.0 * n                                         # constant normal force, no tangential force
    ac, an, at = sff.anisotropies(sff.moments_from_contacts(n, g, w)[None])
    for a in (ac, an, at):
        np.testing.assert_allclose(a, 0.0, atol=1e-12)
    np.testing.assert_allclose(sff.contact_stress_shape(ac, an, at), np.broadcast_to(np.eye(3), (1, 3, 3)), atol=1e-12)


# ---------------------------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------------------------
def two_node_moments(rng, anis_norm=0.4):
    anis = [tuple(random_traceless(rng, anis_norm) for _ in range(3)) for _ in range(2)]
    return np.stack([sff.moments_from_contacts(*model_contacts(*a)) for a in anis]), anis


@pytest.mark.parametrize("polydisperse", [False, True])
def test_sff_fields_use_the_measured_pressure_and_the_kinetic_stress(polydisperse):
    rng = np.random.default_rng(11)
    moments, anis = two_node_moments(rng)
    pc = np.array([2.0, 5.0])
    contact = pc[:, None, None] * np.eye(3)               # only its trace enters
    kinetic = np.array([np.diag([0.2, 0.1, 0.3]), np.diag([0.5, 0.4, 0.1]) + 0.05 * (np.ones((3, 3)) - np.eye(3))])
    if polydisperse:                                      # (N, Nphases+1, 3, 3), bulk at index 0
        contact = np.stack([contact, 2 * contact, 3 * contact], axis=1)
        kinetic = np.stack([kinetic, 2 * kinetic, 3 * kinetic], axis=1)

    out = sff.sff_fields(moments, contact, kinetic)
    for i, (Ac, An, At) in enumerate(anis):
        kin = kinetic[i, 0] if polydisperse else kinetic[i]
        for key, cross in (("", True), ("_linear", False)):
            sigma = pc[i] * sff.contact_stress_shape(Ac, An, At, cross_terms=cross) + kin
            pressure = pc[i] + np.trace(kin) / 3.0
            mu = np.sqrt(0.5 * np.sum(dev(sigma) ** 2)) / pressure
            assert out[f"frictional_coefficient_SFF{key}_Dxyz_Pxyz"][i] == pytest.approx(mu, rel=1e-12)
        np.testing.assert_allclose(out["fabric_anisotropy_tensor"][i], Ac, atol=1e-10)
    assert out["contact_stress_tensor_SFF"].shape == (2, 3, 3)
    assert np.trace(out["contact_stress_tensor_SFF"][0]) == pytest.approx(3 * pc[0])


def test_derived_fields_export_the_sff_group():
    rng = np.random.default_rng(12)
    moments, _ = two_node_moments(rng)
    fields = {"density_mixture": np.ones(2), "sff_moments": moments,
              "contact_tensor": np.broadcast_to(2.0 * np.eye(3), (2, 3, 3)).copy(),
              "kinetic_tensor": np.broadcast_to(0.1 * np.eye(3), (2, 3, 3)).copy()}
    need = get_fields_to_compute({"sff_friction": True})
    assert {"sff_moments", "contact_tensor", "kinetic_tensor"} <= need
    out = df.compute_derived_fields(fields, fields_to_compute=need, phases=np.zeros((1, 2)),
                                    grid_info={"grid_dimension": 3})
    exported = df.select_exports(out, {"sff_friction": True})
    assert list(exported) == df.EXPORT_GROUPS["sff_friction"]
    assert set(exported) <= set(df.PHASE_INDEPENDENT_FIELDS), "bulk-only fields are written once in Polydisperse mode"
    assert "sff_friction" in EXPORT_FLAGS and "sff_moments" in df.PRIMARY_FIELDS


def test_derived_fields_skip_the_group_without_moments():
    fields = {"density_mixture": np.ones(2), "contact_tensor": np.zeros((2, 3, 3)), "kinetic_tensor": np.zeros((2, 3, 3))}
    out = df.compute_derived_fields(fields, fields_to_compute={"sff_friction"}, phases=np.zeros((1, 2)),
                                    grid_info={"grid_dimension": 3})
    assert df.select_exports(out, {"sff_friction": True}) == {}


def test_time_average_pools_the_contacts():
    """The averaged moments are those of the pooled contacts, so the anisotropies are fitted on all time steps."""
    rng = np.random.default_rng(13)
    m1, _ = two_node_moments(rng)
    m2, _ = two_node_moments(rng)
    acc = TimeAverageAccumulator()
    for t, m in enumerate((m1, m2)):
        acc.add({"volume_fraction": np.ones(2), "sff_moments": m}, t)
    mean = acc.mean()
    np.testing.assert_allclose(mean["sff_moments"], 0.5 * (m1 + m2))
    # the fit of the averaged sums equals the fit of the two sets of contacts taken together
    for i in range(2):
        np.testing.assert_allclose(sff.anisotropies((m1[i] + m2[i])[None])[1], sff.anisotropies(mean["sff_moments"][i][None])[1],
                                   atol=1e-10)
