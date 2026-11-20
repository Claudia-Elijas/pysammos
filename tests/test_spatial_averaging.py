"""
End-to-end checks of export_primary_fields + SpatialAverager, driving the real CoarseGraining export
methods (_compute_derived_fields, _write_results, _export_time_average) with synthetic primary fields.
Run from the repository root:  pytest tests/test_spatial_averaging.py -v
"""
import io, os, sys
import numpy as np
import pytest

from pysammos.coarse_graining import CoarseGraining
from pysammos.data_write.h5.writer import H5XarrayManager
from pysammos.macroscopic_fields.gridded import derived_fields, secondary
from pysammos.post_averaging.profiles import SpatialAverager, VerticalIntegrator
from pysammos.time_average.accumulator import TimeAverageAccumulator, summable_terms, finalise

NODES = (4, 6, 5)
SPACING = (0.1, 0.05, 0.2)
GAMMA = 3.0
EXPORT = {"velocity": True, "velocity_gradient": True, "pressure": True, "shear_rate_tensor": True,
          "granular_temperature": True, "inertial_number": True, "frictional_coefficient": True,
          "density_particle": True, "total_stress_tensor": True, "sff_moments": True}


def grid():
    axes = [np.arange(n) * h for n, h in zip(NODES, SPACING)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def make_cg(tmp_path, export_primary_fields=True, poly=False, time_average=True):
    grid_info = {"grid_dimension": 3, **{f"{a}_axis_periodic": False for a in "xyz"}}
    cg = CoarseGraining(particle_path="", contacts_path="", output_path=str(tmp_path) + "/",
                        start_timestep=0, end_timestep=3, dt_time_step=1, t_pad=4, DEM_software="LAMMPS",
                        DEM_keymap={}, grid_info=grid_info, weight_function="Lucy", fields_to_export=EXPORT,
                        time_average=time_average, export_primary_fields=export_primary_fields)
    cg.GridPoints, cg.Nodes, cg.Spacing = grid(), NODES, SPACING
    cg.cg_calc_mode = "Polydisperse" if poly else "Monodisperse"
    cg.phases = np.array([[1e-3, 2500.0], [2e-3, 2600.0]]) if poly else np.array([[1.5e-3, 2500.0]])
    return cg


def primaries(t, rng, shear=False, poly=False):
    pos = grid(); N = len(pos)
    sh = (N, 3) if poly else (N,)
    rho = rng.uniform(500, 1500, sh) if not shear else np.full(sh, 1000.0)
    if not shear:
        rho[rng.random(N) < 0.1] = 0.0                                   # some empty nodes
    if poly:
        rho[:, 0] = rho[:, 1:].sum(axis=1)                               # bulk = sum of the phases
    u = np.zeros(sh + (3,))
    u[..., 0] = (GAMMA * pos[:, 1])[:, None] if poly else GAMMA * pos[:, 1]
    if not shear:
        u += rng.normal(scale=0.1, size=u.shape)
    fl = rng.normal(size=sh + (3, 3)); fl = fl @ np.swapaxes(fl, -1, -2)   # positive
    fl[rho == 0] = 0.0
    fl[rng.random(N) < 0.2] = np.nan                                      # undefined next to empty nodes
    sig = lambda: np.eye(3) * -rng.uniform(50, 100, sh + (1, 1)) + rng.normal(size=sh + (3, 3))
    return dict(volume_fraction=rho / 2500, density_mixture=rho, momentum_density=rho[..., None] * u,
                kinetic_tensor=sig(), fluctuation_tensor=fl, contact_tensor=sig(),
                d43=rng.uniform(1e-3, 2e-3, N), d32=rng.uniform(1e-3, 2e-3, N),
                coordination_number=None, fabric_tensor=None, sff_moments=rng.normal(size=(N, 35)))


def run(cg, steps, poly=False):
    """What fields_in_time does, with synthetic primaries instead of particles."""
    acc = TimeAverageAccumulator() if cg.time_average else None
    for t, primary in enumerate(steps):
        vel = derived_fields.velocity_from_momentum(primary["momentum_density"], primary["density_mixture"])
        grad = derived_fields.velocity_gradient_fd(vel, cg.cg_calc_mode, cg.Nodes, cg.Spacing)
        results = cg._compute_derived_fields(primary, {"velocity": vel, "velocity_gradient": grad})
        cg._write_results(results, t, t)
        if acc is not None:
            acc.add(primary, t)
    if acc is not None:
        cg._export_time_average(acc)
    mode = cg.cg_calc_mode
    series = H5XarrayManager(f"{cg.output_path}CG_Lucy_{mode}.h5").h5_to_xarray()
    tavg = (H5XarrayManager(f"{cg.output_path}CG_Lucy_{mode}_time_averaged.h5").h5_to_xarray()
            if acc is not None else None)
    return series, tavg


# ------------------------------------------------------------------------------------------------
def test_export_primary_fields_and_attrs(tmp_path):
    rng = np.random.default_rng(1)
    cg = make_cg(tmp_path)
    series, tavg = run(cg, [primaries(t, rng) for t in range(3)])
    for key in ("volume_fraction", "density_mixture", "momentum_density", "kinetic_tensor",
                "fluctuation_tensor", "contact_tensor", "d43", "d32", "sff_moments"):
        assert key in series.data_vars, key
    assert "fluctuation_weight" not in series and "fluctuation_weight" in tavg
    assert series["sff_moments"].dims[:2] == ("time", "point")
    assert "fluctuation_tensor" in series.attrs["primary_fields"]
    assert "granular_temperature_xyz" in series.attrs["derived_fields"]
    assert series.attrs["cg_calc_mode"] == "Monodisperse" and series.attrs["grid_dimension"] == 3
    # .vtkhdf written, without the h5-only fields
    import h5py
    with h5py.File(f"{cg.output_path}CG_Lucy_Monodisperse_0000.vtkhdf") as f:
        names = list(f["VTKHDF/PointData"].keys())
    assert "fluctuation_tensor" in names and "sff_moments" not in names
    # the caller's dict is not modified
    assert "fluctuation_tensor" not in EXPORT


def test_linear_shear_profile_and_mean(tmp_path):
    rng = np.random.default_rng(2)
    cg = make_cg(tmp_path, time_average=False)
    series, _ = run(cg, [primaries(0, rng, shear=True)])
    prof = SpatialAverager(series).profile("y")
    y = prof["y"].values
    np.testing.assert_allclose(y, np.arange(NODES[1]) * SPACING[1])
    np.testing.assert_allclose(prof["velocity"].values[0, :, 0], GAMMA * y, atol=1e-6)
    G = prof["velocity_gradient"].values[0]
    np.testing.assert_allclose(G[:, 0, 1], GAMMA, rtol=1e-6)
    G[:, 0, 1] = 0; np.testing.assert_allclose(G, 0, atol=1e-6)
    np.testing.assert_allclose(prof["shear_rate_tensor_xy"].values[0, :, 0, 1], GAMMA / 2, rtol=1e-6)
    mean = SpatialAverager(series).mean()
    np.testing.assert_allclose(mean["velocity_gradient"].values[0, 0, 1], GAMMA, rtol=1e-6)


def test_uniform_field_is_unchanged(tmp_path):
    rng = np.random.default_rng(3)
    p = primaries(0, rng, shear=True)
    for k in ("kinetic_tensor", "contact_tensor"):
        p[k] = np.broadcast_to(p[k][0], p[k].shape).copy()
    p["fluctuation_tensor"] = np.broadcast_to(np.diag([1., 2., 3.]), p["fluctuation_tensor"].shape).copy()
    p["d43"][:] = 1.7e-3; p["d32"][:] = 1.2e-3
    cg = make_cg(tmp_path, time_average=False)
    series, _ = run(cg, [p])
    m = SpatialAverager(series).mean()
    for key in ("pressure_xyz", "granular_temperature_xyz", "d43", "d32", "density_particle",
                "frictional_coefficient_Dxyz_Pxyz"):
        assert key in m, key
        np.testing.assert_allclose(m[key].values[0], series[key].values[0, 0], rtol=1e-6, err_msg=key)


def test_favre_velocity_and_region(tmp_path):
    rng = np.random.default_rng(4)
    cg = make_cg(tmp_path, time_average=False)
    p = primaries(0, rng)
    series, _ = run(cg, [p])
    region = {"x": (0.1, 0.2), "z": (None, 0.4)}
    prof = SpatialAverager(series, region=region).profile("y")
    pos = grid()
    sel = (pos[:, 0] >= 0.1) & (pos[:, 0] <= 0.2) & (pos[:, 2] <= 0.4)
    for j, yv in enumerate(np.unique(pos[:, 1])):
        g = sel & (pos[:, 1] == yv)
        expected = p["momentum_density"][g].sum(0) / p["density_mixture"][g].sum()
        np.testing.assert_allclose(prof["velocity"].values[0, j], expected, rtol=1e-6)
        np.testing.assert_allclose(prof["volume_fraction"].values[0, j], p["volume_fraction"][g].mean(), rtol=1e-6)


def test_time_average_then_space_equals_pooled(tmp_path):
    """Plane average of the time-averaged file == averages over all (time, node) samples pooled."""
    rng = np.random.default_rng(5)
    cg = make_cg(tmp_path)
    steps = [primaries(t, rng) for t in range(4)]
    series, tavg = run(cg, steps)
    m = SpatialAverager(tavg).mean()
    pooled = {k: (np.concatenate([s[k] for s in steps]) if steps[0][k] is not None else None) for k in steps[0]}
    expected = finalise({k: v.mean(axis=0) for k, v in summable_terms(pooled).items()})
    for key in ("density_mixture", "volume_fraction", "kinetic_tensor", "fluctuation_tensor", "d43", "d32"):
        np.testing.assert_allclose(m[key].values[0], expected[key], rtol=1e-6, err_msg=key)
    T = secondary.compute_granular_temperature(np.atleast_1d(expected["density_mixture"]),
                                               expected["fluctuation_tensor"][None])
    np.testing.assert_allclose(m["granular_temperature_xyz"].values[0], T[0], rtol=1e-6)
    # same through the time series: the time average of the plane-averaged primaries
    per_step = [finalise({k: v.mean(axis=0, keepdims=True) for k, v in summable_terms(s).items()}) for s in steps]
    ms = SpatialAverager(series).mean()
    for t in range(4):
        np.testing.assert_allclose(ms["density_mixture"].values[t], per_step[t]["density_mixture"][0], rtol=1e-6)


def test_naive_average_when_primaries_missing(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    cg = make_cg(tmp_path, export_primary_fields=False, time_average=False)
    series, _ = run(cg, [primaries(0, rng)])
    assert "fluctuation_tensor" not in series and "momentum_density" not in series
    with pytest.raises(ValueError, match="Re-run the coarse graining"):
        SpatialAverager(series, naive=False).profile("y")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))           # batch job: no answer -> error
    with pytest.raises(ValueError, match="Re-run the coarse graining"):
        SpatialAverager(series).profile("y")
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))        # user says no
    with pytest.raises(ValueError, match="Re-run the coarse graining"):
        SpatialAverager(series).profile("y")
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))        # user says yes
    prof = SpatialAverager(series).profile("y")
    assert "granular_temperature_xyz" in prof.attrs["naive_average"]
    expected = series["granular_temperature_xyz"].values[0].reshape(NODES)
    np.testing.assert_allclose(prof["granular_temperature_xyz"].values[0],
                               np.nanmean(expected, axis=(0, 2)), rtol=1e-6)
    with pytest.warns(UserWarning):
        SpatialAverager(series, naive=True).mean()


def test_polydisperse_and_vertical_integrator(tmp_path):
    rng = np.random.default_rng(7)
    cg = make_cg(tmp_path, poly=True)
    series, tavg = run(cg, [primaries(t, rng, poly=True) for t in range(2)], poly=True)
    prof = VerticalIntegrator(tavg, "z").integration()
    assert prof["velocity"].dims == ("time", "z", "phase", "dim1_3D")
    assert prof["velocity_gradient"].dims == ("time", "z", "dim1_3D", "dim2_3D")
    assert prof["inertial_number_Sxy_Pxyz_d43"].dims[:3] == ("time", "z", "phase")
    # bulk = sum over phases is kept by the averaging (linear)
    np.testing.assert_allclose(prof["density_mixture"].values[..., 0],
                               prof["density_mixture"].values[..., 1:].sum(-1), rtol=1e-6)
