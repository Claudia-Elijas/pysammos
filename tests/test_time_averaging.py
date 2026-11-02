"""
Synthetic tests for pysammos.time_averaging.TimeAverageAccumulator, and for the time-averaging
path end to end (accumulator -> add_velocity_fields -> compute_derived_fields).

Run from the repository root:   pytest tests/test_time_averaging.py -v
"""
import numpy as np
import pytest

from pysammos.time_average.accumulator import TimeAverageAccumulator, ARITHMETIC_FIELDS
from pysammos.macroscopic_fields.gridded import derived_fields as df
from pysammos.macroscopic_fields.field_dependencies import get_fields_to_compute

NPHASES = 2
NODES, SPACING = (5, 4, 3), (0.1, 0.2, 0.3)
N = int(np.prod(NODES))
ALL_FLAGS = {flag: True for flag in df.EXPORT_GROUPS if flag != "coordination_number_without_rattlers"}
ALL_COMPUTE = get_fields_to_compute(ALL_FLAGS)
PHASES = np.array([[1.0e-3, 2500.0], [3.0e-3, 2700.0]])


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def random_step(rng, mode, nan_nodes=()):
    """
    Random primary fields of one time step. In polydisperse mode the bulk (index 0) is the sum
    over phases, as in tensors.py / dispatcher, so bulk/phase consistency can be checked.
    """
    sym = lambda a: a + np.swapaxes(a, -1, -2)
    if mode == "Monodisperse":
        f = {
            "volume_fraction": rng.uniform(0.1, 0.6, N),
            "density_mixture": rng.uniform(500, 1500, N),
            "momentum_density": rng.normal(0, 100, (N, 3)),
            "kinetic_tensor": sym(rng.normal(0, 10, (N, 3, 3))) + 50 * np.eye(3),
            "fluctuation_tensor": sym(rng.normal(0, 10, (N, 3, 3))) + 50 * np.eye(3),
            "contact_tensor": sym(rng.normal(0, 100, (N, 3, 3))) + 500 * np.eye(3),
        }
    else:
        phase = lambda shape, lo, hi: rng.uniform(lo, hi, (N, NPHASES) + shape)
        with_bulk = lambda a: np.concatenate([a.sum(axis=1, keepdims=True), a], axis=1)
        f = {
            "volume_fraction": with_bulk(phase((), 0.05, 0.3)),
            "density_mixture": with_bulk(phase((), 250, 750)),
            "momentum_density": with_bulk(rng.normal(0, 50, (N, NPHASES, 3))),
            "kinetic_tensor": with_bulk(sym(rng.normal(0, 5, (N, NPHASES, 3, 3))) + 25 * np.eye(3)),
            "fluctuation_tensor": with_bulk(sym(rng.normal(0, 5, (N, NPHASES, 3, 3))) + 25 * np.eye(3)),
            "contact_tensor": with_bulk(sym(rng.normal(0, 50, (N, NPHASES, 3, 3))) + 250 * np.eye(3)),
        }
    f["d43"] = rng.uniform(1e-3, 3e-3, N)
    f["d32"] = rng.uniform(1e-3, 3e-3, N)
    f["coordination_number"] = rng.uniform(2, 6, N)
    f["fabric_tensor"] = sym(rng.normal(0, 1, (N, 3, 3)))
    f["fluctuation_tensor"][list(nan_nodes)] = np.nan
    return f


def derive(primary, mode, velocity_fields=None):
    """Mirror of CoarseGraining._derive (without the export selection)."""
    fields = {**primary, **(velocity_fields or {})}
    fields = df.add_velocity_fields(fields, fields_to_compute=ALL_COMPUTE, cg_calc_mode=mode,
                                    nodes=NODES, spacing=SPACING)
    return df.compute_derived_fields(fields, fields_to_compute=ALL_COMPUTE, phases=PHASES,
                                     grid_info={"grid_dimension": 3})


def accumulate(steps, times=None):
    acc = TimeAverageAccumulator()
    for i, step in enumerate(steps):
        acc.add(step, 100 + 10 * i if times is None else times[i])
    return acc


MODES = ["Monodisperse", "Polydisperse"]


# ---------------------------------------------------------------------------------------------
# bookkeeping: properties, errors, None, aliasing
# ---------------------------------------------------------------------------------------------
def test_empty_accumulator_raises():
    acc = TimeAverageAccumulator()
    assert len(acc) == 0 and not acc.has_slices
    with pytest.raises(ValueError):
        acc.mean()
    with pytest.raises(ValueError):
        acc.mean_slices()


def test_properties_and_len():
    rng = np.random.default_rng(0)
    acc = accumulate([random_step(rng, "Monodisperse") for _ in range(3)], times=[5, 15, 25])
    assert len(acc) == 3
    assert acc.time_steps.dtype.kind == "i"
    np.testing.assert_array_equal(acc.time_steps, [5, 15, 25])
    with pytest.raises(AttributeError):
        acc.time_steps = [1, 2, 3]          # read-only property
    with pytest.raises(AttributeError):
        acc.has_slices = True


def test_none_fields_stay_none():
    rng = np.random.default_rng(1)
    steps = [random_step(rng, "Monodisperse") for _ in range(2)]
    for s in steps:
        s["fabric_tensor"] = s["d32"] = s["fluctuation_tensor"] = None
    mean = accumulate(steps).mean()
    assert mean["fabric_tensor"] is None and mean["d32"] is None and mean["fluctuation_tensor"] is None
    assert mean["d43"] is not None
    assert set(mean) == set(df.PRIMARY_FIELDS)


def test_no_aliasing_with_inputs():
    """The first add must copy: later in-place sums must not change the caller's arrays, and vice versa."""
    rng = np.random.default_rng(2)
    step = random_step(rng, "Monodisperse")
    original = {k: v.copy() for k, v in step.items()}
    acc = TimeAverageAccumulator()
    acc.add(step, 0)
    acc.add(step, 1)                                    # same arrays twice
    for k in step:
        np.testing.assert_array_equal(step[k], original[k], err_msg=k)
    step["volume_fraction"][:] = -1.0                   # caller reuses its buffer afterwards
    np.testing.assert_allclose(acc.mean()["volume_fraction"], original["volume_fraction"])


# ---------------------------------------------------------------------------------------------
# averages of primary fields
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_single_step_mean_is_the_step(mode):
    rng = np.random.default_rng(3)
    step = random_step(rng, mode)
    mean = accumulate([step]).mean()
    for key in df.PRIMARY_FIELDS:
        np.testing.assert_allclose(mean[key], step[key], rtol=1e-12, err_msg=key)


@pytest.mark.parametrize("mode", MODES)
def test_arithmetic_means(mode):
    rng = np.random.default_rng(4)
    steps = [random_step(rng, mode) for _ in range(5)]
    mean = accumulate(steps).mean()
    for key in ARITHMETIC_FIELDS:
        np.testing.assert_allclose(mean[key], np.mean([s[key] for s in steps], axis=0), rtol=1e-12, err_msg=key)


@pytest.mark.parametrize("mode", MODES)
def test_order_of_time_steps_does_not_matter(mode):
    rng = np.random.default_rng(5)
    steps = [random_step(rng, mode, nan_nodes=[i]) for i in range(4)]
    a, b = accumulate(steps).mean(), accumulate(steps[::-1]).mean()
    for key in df.PRIMARY_FIELDS:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, err_msg=key)


@pytest.mark.parametrize("mode", MODES)
def test_favre_velocity(mode):
    """Averaged velocity is sum(rho u)/sum(rho) per node (and phase), not the mean of the velocities."""
    rng = np.random.default_rng(6)
    steps = [random_step(rng, mode) for _ in range(4)]
    v = derive(accumulate(steps).mean(), mode)["velocity"]
    favre = (np.sum([s["momentum_density"] for s in steps], axis=0)
             / np.sum([s["density_mixture"] for s in steps], axis=0)[..., None])
    naive = np.mean([s["momentum_density"] / s["density_mixture"][..., None] for s in steps], axis=0)
    np.testing.assert_allclose(v, favre, rtol=1e-12)
    assert not np.allclose(v, naive, rtol=1e-6), "test data should distinguish Favre from arithmetic mean"


@pytest.mark.parametrize("mode", MODES)
def test_fluctuation_tensor_masked_per_node(mode):
    """
    Node 0: undefined (NaN) in step 1 only  -> T = sum(fluct)/sum(rho) over steps 0 and 2.
    Node 1: undefined in every step          -> fluctuation tensor 0.
    Other nodes: all steps.
    """
    rng = np.random.default_rng(7)
    steps = [random_step(rng, mode, nan_nodes=[1]), random_step(rng, mode, nan_nodes=[0, 1]),
             random_step(rng, mode, nan_nodes=[1])]
    mean = accumulate(steps).mean()
    T = mean["fluctuation_tensor"] / mean["density_mixture"][..., None, None]   # temperature tensor

    def expected(node, used):
        fl = sum(steps[i]["fluctuation_tensor"][node] for i in used)
        rho = sum(steps[i]["density_mixture"][node] for i in used)
        return fl / np.asarray(rho)[..., None, None]

    np.testing.assert_allclose(T[0], expected(0, [0, 2]), rtol=1e-12)
    np.testing.assert_allclose(T[2], expected(2, [0, 1, 2]), rtol=1e-12)
    np.testing.assert_array_equal(mean["fluctuation_tensor"][1], 0.0)
    assert np.isfinite(mean["fluctuation_tensor"]).all()


def test_grain_sizes_are_pooled_over_time():
    """
    Synthetic particles: at every node and step, a few particles with kernel weights w.
      phi = sum(w d^3) (times a constant), d43 = sum(w d^4)/sum(w d^3), d32 = sum(w d^3)/sum(w d^2)
    The time average must equal d43 and d32 of all particles of all steps pooled together.
    """
    rng = np.random.default_rng(8)
    n_steps, n_part = 4, 6
    w = rng.uniform(0.1, 1.0, (n_steps, N, n_part))
    d = rng.uniform(1e-3, 4e-3, (n_steps, N, n_part))
    M = lambda k, s: np.sum(w[s] * d[s] ** k, axis=-1)          # moment k of step s, per node
    steps = []
    for s in range(n_steps):
        step = random_step(rng, "Monodisperse")
        step["volume_fraction"] = np.pi / 6 * M(3, s)
        step["d43"] = M(4, s) / M(3, s)
        step["d32"] = M(3, s) / M(2, s)
        steps.append(step)
    mean = accumulate(steps).mean()
    pooled = lambda k: np.sum(w * d ** k, axis=(0, 2))
    np.testing.assert_allclose(mean["d43"], pooled(4) / pooled(3), rtol=1e-12)
    np.testing.assert_allclose(mean["d32"], pooled(3) / pooled(2), rtol=1e-12)


def test_polydisperse_bulk_equals_sum_of_phases():
    """Linear fields keep bulk = sum over phases after averaging; bulk velocity = mass-weighted phase velocities."""
    rng = np.random.default_rng(9)
    steps = [random_step(rng, "Polydisperse") for _ in range(4)]
    mean = accumulate(steps).mean()
    for key in ("volume_fraction", "density_mixture", "momentum_density",
                "kinetic_tensor", "contact_tensor", "fluctuation_tensor"):
        np.testing.assert_allclose(mean[key][:, 0], mean[key][:, 1:].sum(axis=1), rtol=1e-12, err_msg=key)
    v = derive(mean, "Polydisperse")["velocity"]
    rho = mean["density_mixture"]
    v_bulk_from_phases = (rho[:, 1:, None] * v[:, 1:]).sum(axis=1) / rho[:, 1:].sum(axis=1)[:, None]
    np.testing.assert_allclose(v[:, 0], v_bulk_from_phases, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# slices
# ---------------------------------------------------------------------------------------------
def test_slices_average_and_positions():
    rng = np.random.default_rng(10)
    acc = TimeAverageAccumulator()
    pos0 = rng.normal(size=(7, 3))
    vals = [{"granular_temperature_Kamrin": rng.normal(size=7),
             "granular_temperature_LAMMPS": rng.normal(size=7)} for _ in range(3)]
    for i, v in enumerate(vals):
        acc.add_slices(v, pos0 if i == 0 else pos0 + 1.0)
    assert acc.has_slices
    np.testing.assert_array_equal(acc.slice_positions, pos0)          # from the first call
    m = acc.mean_slices()
    for key in vals[0]:
        np.testing.assert_allclose(m[key], np.mean([v[key] for v in vals], axis=0), rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# end to end: average -> derived fields
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_identical_steps_average_equals_single_step(mode):
    """If every time step is the same, the time-averaged derived fields equal those of one step
    (velocity gradient by finite differences in both)."""
    rng = np.random.default_rng(11)
    step = random_step(rng, mode)
    one = derive(step, mode)
    avg = derive(accumulate([step] * 4).mean(), mode)
    assert set(one) == set(avg)
    for key in one:
        if one[key] is None:
            assert avg[key] is None, key
        else:
            np.testing.assert_allclose(avg[key], one[key], rtol=1e-10, err_msg=key)


@pytest.mark.parametrize("mode", MODES)
def test_rheology_from_averages_not_averaged_rheology(mode):
    """Pressure and mu of the average come from the averaged stress (not the mean of per-step values)."""
    rng = np.random.default_rng(12)
    steps = [random_step(rng, mode) for _ in range(4)]
    avg = derive(accumulate(steps).mean(), mode)
    sigma = np.mean([s["kinetic_tensor"] + s["contact_tensor"] for s in steps], axis=0)
    p = np.trace(sigma, axis1=-2, axis2=-1) / 3
    np.testing.assert_allclose(avg["pressure_xyz"], p, rtol=1e-12)
    np.testing.assert_allclose(avg["frictional_coefficient_01_Pxyz"], sigma[..., 0, 1] / p, rtol=1e-12)
    mean_of_mu = np.mean([derive(s, mode)["frictional_coefficient_01_Pxyz"] for s in steps], axis=0)
    assert not np.allclose(avg["frictional_coefficient_01_Pxyz"], mean_of_mu, rtol=1e-6)


def test_least_squares_gradient_is_kept_per_step_but_fd_for_average():
    """Per step, a given velocity_gradient (e.g. least squares) is used as is; the average uses FD."""
    rng = np.random.default_rng(13)
    step = random_step(rng, "Monodisperse")
    ls_like = rng.normal(size=(N, 3, 3))
    per_step = derive(step, "Monodisperse", velocity_fields={"velocity": None, "velocity_gradient": ls_like})
    assert per_step["velocity_gradient"] is ls_like
    avg = derive(accumulate([step]).mean(), "Monodisperse")
    v = step["momentum_density"] / step["density_mixture"][:, None]
    np.testing.assert_allclose(avg["velocity_gradient"], df.velocity_gradient_fd(v, "Monodisperse", NODES, SPACING),
                               rtol=1e-12)
