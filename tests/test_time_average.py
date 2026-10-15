"""
Tests for the time-averaged output of CoarseGraining and for the fields derived from
the primary ones. Run from the repository root with

    python -m pytest tests

The tests feed synthetic primary fields of a steady shear flow (vx = gdot * y plus a
velocity noise whose time mean is zero at every node) to the same methods the time
loop uses, so no DEM data are needed.
"""
import h5py
import numpy as np

from pysammos.coarse_graining import CoarseGraining
from pysammos.macroscopic_fields.gridded import secondary

FIELDS = dict(volume_fraction=True, density_mixture=True, density_particle=True, velocity=True,
              velocity_gradient=True, total_stress_tensor=True, pressure=True, granular_temperature=True,
              granular_temperature_slices=False, inertial_number=True, d43=True, d32=True,
              frictional_coefficient=True, shear_rate_tensor=True)
NX, NY, NZ, DX = 3, 12, 3, 1.0e-3
D, RHO_P, PHI = 1.0e-3, 2500.0, 0.6


def make_cg(tmp_path):
    grid_info = dict(grid_dimension=3, grid_axes="xyz", automatic_grid=False,
                     x_min=0.0, x_max=(NX - 1) * DX, y_min=0.0, y_max=(NY - 1) * DX,
                     z_min=0.0, z_max=(NZ - 1) * DX, x_transect=None, y_transect=None, z_transect=None,
                     x_axis_periodic=True, y_axis_periodic=False, z_axis_periodic=True)
    cg = CoarseGraining(particle_path="", contacts_path="", output_path=f"{tmp_path}/",
                        start_timestep=0, end_timestep=0, dt_time_step=1, t_pad=4,
                        DEM_software="mfix", DEM_keymap={}, grid_info=grid_info, weight_function="Lucy",
                        fields_to_export=FIELDS, ignore_phases=True, vkthdf_output=False, h5_output=True,
                        time_average=True)
    x, y, z = np.meshgrid(np.arange(NX) * DX, np.arange(NY) * DX, np.arange(NZ) * DX, indexing="ij")
    cg.GridPoints = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    cg.Nodes = np.array([NX, NY, NZ])
    cg.Spacing = np.array([DX, DX, DX])
    cg.cg_calc_mode = "Monodisperse"
    cg.phases = np.array([[D, RHO_P]])
    return cg, cg.GridPoints[:, 1]


def snapshot(cg, y, gdot, noise, pressure, tau, temperature):
    """Primary fields of one time step: vx = gdot * y + noise, uniform stress and temperature."""
    n = y.size
    rho = np.full(n, PHI * RHO_P)
    velocity = np.zeros((n, 3))
    velocity[:, 0] = gdot * y + noise
    stress = np.zeros((n, 3, 3))
    stress[:, 0, 0] = stress[:, 1, 1] = stress[:, 2, 2] = pressure
    stress[:, 0, 1] = stress[:, 1, 0] = tau
    fluctuation = np.zeros((n, 3, 3))
    fluctuation[:, 0, 0] = fluctuation[:, 1, 1] = fluctuation[:, 2, 2] = rho * temperature
    return dict(volume_fraction=np.full(n, PHI), density_mixture=rho,
                momentum_density=rho[:, np.newaxis] * velocity, velocity=velocity,
                velocity_gradient=secondary.compute_vector_bulk_gradient(velocity, cg.Nodes, cg.Spacing),
                kinetic_tensor=np.zeros((n, 3, 3)), fluctuation_tensor=fluctuation, contact_tensor=stress,
                d43=np.full(n, D), d32=np.full(n, D), coordination_number=None, fabric_tensor=None)


def test_time_average_of_steady_shear(tmp_path):
    cg, y = make_cg(tmp_path)
    rng = np.random.default_rng(1)
    gdot, steps = 10.0, 40
    noise = rng.normal(0.0, 0.05, size=(steps // 2, y.size))  # m/s, five times gdot * DX
    noise = np.concatenate((noise, -noise))                    # zero time mean at every node
    pressure = np.where(np.arange(steps) % 2 == 0, 50.0, 150.0)  # Pa, time mean 100 Pa
    temperature = np.where(np.arange(steps) % 2 == 0, 1.0e-4, 2.0e-4)
    inertial_number_per_step = []
    for t in range(steps):
        fields = snapshot(cg, y, gdot, noise[t], pressure[t], 0.3 * pressure[t], temperature[t])
        inertial_number_per_step.append(cg._secondary_fields(fields)["inertial_number_Sxy_Py_d43"])
        cg._primary_fields = fields
        cg._time_average_add(t)
    average = cg._time_averaged_results()

    inertial_number = gdot * D / np.sqrt(100.0 / RHO_P)
    assert np.allclose(average["velocity"][:, 0], gdot * y)
    assert np.allclose(average["shear_rate_tensor_xy_dev_mag"], gdot)
    assert np.allclose(average["pressure_y"], 100.0)
    assert np.allclose(average["inertial_number_Sxy_Py_d43"], inertial_number)
    assert np.allclose(average["frictional_coefficient_01_Py"], 0.3)
    assert np.allclose(average["granular_temperature_x"], 1.5e-4)
    assert np.allclose(average["granular_temperature_xyz"], 1.5e-4)
    assert np.allclose(average["d43"], D)
    assert np.allclose(average["d32"], D)
    # the mean of the inertial numbers of the time steps is several times too large
    assert np.mean(inertial_number_per_step) > 3.0 * inertial_number

    cg._write_time_average()
    with h5py.File(tmp_path / "CG_Lucy_Monodisperse_time_averaged.h5", "r") as f:
        assert list(f.attrs["time_steps_averaged"]) == list(range(steps))
        assert np.allclose(f["inertial_number_Sxy_Py_d43"][0], inertial_number)
        assert f["positions"].shape == (y.size, 3)


def test_time_average_skips_undefined_temperature(tmp_path):
    cg, y = make_cg(tmp_path)
    for t, temperature in enumerate((1.0e-4, 3.0e-4, 5.0e-4)):
        fields = snapshot(cg, y, 1.0, 0.0, 100.0, 30.0, temperature)
        if t == 2:  # undefined at the first node in the last step
            fields["fluctuation_tensor"][0] = np.nan
        cg._primary_fields = fields
        cg._time_average_add(t)
    temperature = cg._time_averaged_results()["granular_temperature_x"]
    assert np.isclose(temperature[0], 2.0e-4)
    assert np.allclose(temperature[1:], 3.0e-4)


def test_inertial_number_uses_deviatoric_shear_rate(tmp_path):
    cg, y = make_cg(tmp_path)
    fields = snapshot(cg, y, 0.0, 0.0, 100.0, 30.0, 1.0e-4)
    dilatation, gdot = 4.0, 10.0
    gradient = np.zeros((y.size, 3, 3))
    gradient[:, 0, 0] = dilatation  # dvx/dx
    gradient[:, 0, 1] = gdot        # dvx/dy
    fields["velocity_gradient"] = gradient
    results = cg._secondary_fields(fields)
    # deviator of the 2D strain rate: |D'| = sqrt(dilatation^2 + gdot^2)
    expected = np.sqrt(dilatation ** 2 + gdot ** 2) * D * np.sqrt(RHO_P) / np.sqrt(100.0)
    assert np.allclose(results["inertial_number_Sxy_Py_d43"], expected)
