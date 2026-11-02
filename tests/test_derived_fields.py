"""
Synthetic tests for pysammos.macroscopic_fields.gridded.derived_fields.

Run from the repository root:   pytest tests/test_derived_fields.py -v
"""
import numpy as np
import pytest

from pysammos.macroscopic_fields.gridded import derived_fields as df
from pysammos.macroscopic_fields.field_dependencies import get_fields_to_compute

ALL_FLAGS = {flag: True for flag in df.EXPORT_GROUPS if flag != "coordination_number_without_rattlers"}
ALL_COMPUTE = get_fields_to_compute(ALL_FLAGS)
NPHASES = 2


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def grid_coords(nodes, spacing):
    """Node coordinates in the order secondary.compute_vector_bulk_gradient assumes (C order, x slowest)."""
    axes = [np.arange(n) * h for n, h in zip(nodes, spacing)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)            # (N, ndim)


def phase_table():
    """phases array as in CoarseGraining: column 0 diameters, column 1 densities."""
    return np.array([[1.0e-3, 2500.0], [3.0e-3, 2700.0]])[:NPHASES]


def analytic_fields(N, mode, PD=(4.0, 5.0, 6.0), tau=1.5, gamma=2.0, T0=0.3, rho=1200.0, phi=0.5, d=2e-3):
    """
    Fields with known derived values:
      total stress  = diag(PD) + tau*(e_x e_y + e_y e_x)    (split 1/4 kinetic + 3/4 contact)
      velocity grad = simple shear du/dy = gamma
      fluctuation   = T0 * rho * I                         (granular temperature T0)
    In polydisperse mode every phase (and the bulk, index 0) carries the same values.
    """
    shape = (N,) if mode == "Monodisperse" else (N, NPHASES + 1)
    diag = np.broadcast_to(np.diag(PD), shape + (3, 3))
    eye = np.broadcast_to(np.eye(3), shape + (3, 3))
    shear = np.zeros(shape + (3, 3)); shear[..., 0, 1] = shear[..., 1, 0] = tau
    grad_v = np.zeros((N, 3, 3)); grad_v[:, 0, 1] = gamma
    return {
        "volume_fraction": np.full(shape, phi),
        "density_mixture": np.full(shape, rho),
        "momentum_density": np.zeros(shape + (3,)),
        "kinetic_tensor": 0.25 * diag.copy(),
        "contact_tensor": 0.75 * diag + shear,
        "fluctuation_tensor": T0 * rho * eye.copy(),
        "d43": np.full(N, d),
        "d32": np.full(N, d),
        "coordination_number": np.full(N, 4.0),
        "fabric_tensor": np.broadcast_to(np.eye(3) / 3, (N, 3, 3)).copy(),
        "velocity": np.zeros(shape + (3,)),
        "velocity_gradient": grad_v,
    }


# ---------------------------------------------------------------------------------------------
# velocity helpers
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["Monodisperse", "Polydisperse"])
def test_velocity_from_momentum(mode):
    rng = np.random.default_rng(0)
    shape = (10,) if mode == "Monodisperse" else (10, NPHASES + 1)
    rho = rng.uniform(1, 2, shape)
    u = rng.normal(size=shape + (3,))
    np.testing.assert_allclose(df.velocity_from_momentum(rho[..., None] * u, rho), u, rtol=1e-14)


def test_bulk_velocity():
    v_mono = np.ones((5, 3))
    v_poly = np.arange(5 * 3 * 3, dtype=float).reshape(5, 3, 3)
    assert df.bulk_velocity(v_mono, "Monodisperse") is v_mono
    np.testing.assert_array_equal(df.bulk_velocity(v_poly, "Polydisperse"), v_poly[:, 0, :])


@pytest.mark.parametrize("mode", ["Monodisperse", "Polydisperse"])
@pytest.mark.parametrize("nodes, spacing", [((6, 5, 4), (0.1, 0.2, 0.3)), ((7, 5), (0.1, 0.25))])
def test_velocity_gradient_fd_exact_for_linear_field(mode, nodes, spacing):
    """Finite differences are exact for u = A x, including the one-sided boundary nodes."""
    rng = np.random.default_rng(1)
    x = grid_coords(nodes, spacing)
    A = rng.normal(size=(3, len(nodes)))
    u = x @ A.T                                                    # (N, 3)
    v = u if mode == "Monodisperse" else np.stack([u, 2 * u, -u], axis=1)  # bulk = phase index 0
    grad = df.velocity_gradient_fd(v, mode, nodes, spacing)
    np.testing.assert_allclose(grad, np.broadcast_to(A, grad.shape), atol=1e-10)


# ---------------------------------------------------------------------------------------------
# add_velocity_fields
# ---------------------------------------------------------------------------------------------
def test_add_velocity_fields_is_noop_when_given():
    """Per time step: velocity and gradient from _compute_fields are passed through, not recomputed."""
    f = analytic_fields(24, "Monodisperse")
    out = df.add_velocity_fields(f, fields_to_compute=ALL_COMPUTE, cg_calc_mode="Monodisperse",
                                 nodes=(4, 3, 2), spacing=(1, 1, 1))
    assert out["velocity"] is f["velocity"]
    assert out["velocity_gradient"] is f["velocity_gradient"]


def test_add_velocity_fields_fills_missing():
    nodes, spacing = (5, 4, 3), (0.1, 0.1, 0.1)
    x = grid_coords(nodes, spacing)
    rho = np.full(len(x), 2.0)
    u = np.column_stack((3.0 * x[:, 1], np.zeros(len(x)), np.zeros(len(x))))   # du/dy = 3
    f = {"momentum_density": rho[:, None] * u, "density_mixture": rho}
    out = df.add_velocity_fields(f, fields_to_compute=ALL_COMPUTE, cg_calc_mode="Monodisperse",
                                 nodes=nodes, spacing=spacing)
    np.testing.assert_allclose(out["velocity"], u)
    assert np.allclose(out["velocity_gradient"][:, 0, 1], 3.0)
    assert "velocity" not in f, "input dict must not be modified"


def test_add_velocity_fields_respects_fields_to_compute():
    f = {"momentum_density": np.ones((4, 3)), "density_mixture": np.ones(4)}
    out = df.add_velocity_fields(f, fields_to_compute={"momentum_density", "density_mixture"},
                                 cg_calc_mode="Monodisperse", nodes=(4,), spacing=(1.0,))
    assert out["velocity"] is None and out["velocity_gradient"] is None


# ---------------------------------------------------------------------------------------------
# compute_derived_fields: analytic values
# ---------------------------------------------------------------------------------------------
PX, PY, PZ, TAU, GAMMA, T0, RHO, PHI, D = 4.0, 5.0, 6.0, 1.5, 2.0, 0.3, 1200.0, 0.5, 2e-3
PRESSURE = {"xyz": (PX + PY + PZ) / 3, "xy": (PX + PY) / 2, "x": PX, "y": PY, "z": PZ}
_dev = lambda diag, mean: np.sqrt(0.5 * (sum((p - mean) ** 2 for p in diag) + 2 * TAU**2))
DEV_MAG = {"Dxyz": _dev((PX, PY, PZ), PRESSURE["xyz"]), "Dxy": _dev((PX, PY), PRESSURE["xy"]), "01": TAU}


@pytest.mark.parametrize("mode", ["Monodisperse", "Polydisperse"])
def test_derived_values_analytic(mode):
    N = 24
    out = df.compute_derived_fields(analytic_fields(N, mode), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 3})
    close = lambda key, value: np.testing.assert_allclose(out[key], value, rtol=1e-12, err_msg=key)

    close("density_particle", RHO / PHI)
    for comp, p in PRESSURE.items():
        close(f"pressure_{comp}", p)
    close("total_stress_tensor_xyz_mag", np.sqrt(0.5 * (PX**2 + PY**2 + PZ**2 + 2 * TAU**2)))
    close("total_stress_tensor_xyz_dev_mag", DEV_MAG["Dxyz"])
    close("total_stress_tensor_xy_dev_mag", DEV_MAG["Dxy"])
    for comp in ("xy", "xyz"):                         # simple shear: |gamma_dot| = gamma
        close(f"shear_rate_tensor_{comp}_mag", GAMMA)
        close(f"shear_rate_tensor_{comp}_dev_mag", GAMMA)
    for s_name, tau in DEV_MAG.items():
        for p_name in ("xyz", "xy", "y"):
            close(f"frictional_coefficient_{s_name}_P{p_name}", tau / PRESSURE[p_name])
    for axis in "xyz":
        close(f"granular_temperature_{axis}", T0)


def test_inertial_number_analytic_monodisperse():
    out = df.compute_derived_fields(analytic_fields(24, "Monodisperse"), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 3})
    for d_name in ("d43", "d32"):
        for p_name in ("xyz", "xy", "y"):
            key = f"inertial_number_Sxy_P{p_name}_{d_name}"
            expected = GAMMA * D * np.sqrt(RHO / PHI) / np.sqrt(PRESSURE[p_name])
            np.testing.assert_allclose(out[key], expected, rtol=1e-12, err_msg=key)


def test_inertial_number_analytic_polydisperse():
    """Bulk (index 0) uses d43/d32 and rho/phi; phase p uses its own diameter and solid density."""
    phases = phase_table()
    out = df.compute_derived_fields(analytic_fields(24, "Polydisperse"), fields_to_compute=ALL_COMPUTE,
                                    phases=phases, grid_info={"grid_dimension": 3})
    for p_name in ("xyz", "xy", "y"):
        I = out[f"inertial_number_Sxy_P{p_name}_d43"]
        p = PRESSURE[p_name]
        np.testing.assert_allclose(I[:, 0], GAMMA * D * np.sqrt(RHO / PHI) / np.sqrt(p), rtol=1e-12)
        for k in range(NPHASES):
            np.testing.assert_allclose(I[:, k + 1], GAMMA * phases[k, 0] * np.sqrt(phases[k, 1]) / np.sqrt(p),
                                       rtol=1e-12)


def test_granular_temperature_xyz_polydisperse():
    out = df.compute_derived_fields(analytic_fields(24, "Polydisperse"), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 3})
    np.testing.assert_allclose(out["granular_temperature_xyz"], T0, rtol=1e-12)


def test_granular_temperature_xyz_monodisperse():
    out = df.compute_derived_fields(analytic_fields(24, "Monodisperse"), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 3})
    np.testing.assert_allclose(out["granular_temperature_xyz"], T0, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# compute_derived_fields: only what is needed, nothing mutated, all export keys produced
# ---------------------------------------------------------------------------------------------
def test_only_requested_fields_are_computed():
    """Exporting only pressure must not touch shear rate, temperature, inertial number, ...
    (velocity_gradient and fluctuation_tensor are None: computing anything that needs them would fail)."""
    ftc = get_fields_to_compute({"pressure": True})
    f = analytic_fields(24, "Monodisperse")
    f["velocity_gradient"] = f["fluctuation_tensor"] = f["d43"] = f["d32"] = None
    out = df.compute_derived_fields(f, fields_to_compute=ftc, phases=phase_table(),
                                    grid_info={"grid_dimension": 3})
    computed = set(out) - set(f)
    assert computed == set(df.EXPORT_GROUPS["pressure"]) | set(df.EXPORT_GROUPS["total_stress_tensor"])


def test_2d_grid_has_no_3d_shear_rate():
    out = df.compute_derived_fields(analytic_fields(24, "Monodisperse"), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 2})
    assert "shear_rate_tensor_xy" in out
    assert not any(k.startswith("shear_rate_tensor_xyz") for k in out)


def test_compute_derived_fields_does_not_mutate_input():
    f = analytic_fields(24, "Polydisperse")
    keys, copies = list(f), {k: v.copy() for k, v in f.items()}
    df.compute_derived_fields(f, fields_to_compute=ALL_COMPUTE, phases=phase_table(),
                              grid_info={"grid_dimension": 3})
    assert list(f) == keys
    for k in keys:
        np.testing.assert_array_equal(f[k], copies[k])


@pytest.mark.parametrize("mode", ["Monodisperse", "Polydisperse"])
def test_every_export_key_is_produced(mode):
    """Catches typos between EXPORT_GROUPS and the keys compute_derived_fields writes."""
    out = df.compute_derived_fields(analytic_fields(24, mode), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 3})
    missing = [key for flag in ALL_FLAGS for key in df.EXPORT_GROUPS[flag] if key not in out]
    assert not missing, missing


# ---------------------------------------------------------------------------------------------
# select_exports
# ---------------------------------------------------------------------------------------------
def test_select_exports_filters_and_orders():
    fields = {"pressure_xyz": 1, "pressure_xy": 2, "pressure_x": 3, "pressure_y": 4, "pressure_z": 5,
              "volume_fraction": 6, "density_mixture": None, "velocity": 7}
    flags = {"pressure": True, "density_mixture": True, "volume_fraction": True, "velocity": False}
    out = df.select_exports(fields, flags)
    assert list(out) == ["volume_fraction", "pressure_xyz", "pressure_xy", "pressure_x", "pressure_y", "pressure_z"]


def test_select_exports_skips_missing_3d_keys():
    out = df.compute_derived_fields(analytic_fields(24, "Monodisperse"), fields_to_compute=ALL_COMPUTE,
                                    phases=phase_table(), grid_info={"grid_dimension": 2})
    sel = df.select_exports(out, {"shear_rate_tensor": True})
    assert list(sel) == ["shear_rate_tensor_xy", "shear_rate_tensor_xy_mag",
                         "shear_rate_tensor_xy_dev", "shear_rate_tensor_xy_dev_mag"]
