"""
Derived coarse-grained fields.

The coarse-grained fields are split in two groups:

- **Primary fields** are kernel-weighted sums over particles or contacts at one time step
  (volume fraction, mixture density, momentum density, kinetic, fluctuation and contact tensors,
  mean grain sizes, coordination number, fabric tensor). They are computed from the particle data
  in :meth:`CoarseGraining._compute_fields`, and they are the only fields that are averaged in time
  (see :class:`pysammos.time_averaging.accumulator.TimeAverageAccumulator`).

- **Derived fields** are nonlinear functions of the primary fields (ratios, gradients, invariants):
  velocity and its gradient, particle density, total stress, pressure, granular temperature,
  shear rate, inertial number and frictional coefficient. They are computed here, from the primary
  fields of a single time step or from their time averages, with the same code.

Within a time step the velocity and its gradient are also needed *before* the kinetic and fluctuation
tensors, so :meth:`CoarseGraining._compute_fields` computes them there (with the helpers below) and
passes them on; :func:`add_velocity_fields` only fills them in when they are missing (time averages).
"""

import numpy as np

from . import secondary

# Fields written once per grid point (not per phase) in the polydisperse .vtkhdf ouput 
PHASE_INDEPENDENT_FIELDS = ["d32", "d43", "coordination_number", "velocity_gradient", "fabric_tensor",
                            "shear_rate_tensor_xyz", "shear_rate_tensor_xyz_mag",
                            "shear_rate_tensor_xy", "shear_rate_tensor_xy_mag",
                            "shear_rate_tensor_xyz_dev", "shear_rate_tensor_xyz_dev_mag",
                            "shear_rate_tensor_xy_dev", "shear_rate_tensor_xy_dev_mag"]

#: Primary fields: the inputs of the derived fields, and the fields averaged in time.
PRIMARY_FIELDS = ("volume_fraction", "density_mixture", "momentum_density",
                  "kinetic_tensor", "fluctuation_tensor", "contact_tensor",
                  "d43", "d32", "coordination_number", "fabric_tensor")

#: Output keys written for each flag of ``fields_to_export``, in the order they are written.
EXPORT_GROUPS = {
    "volume_fraction": ["volume_fraction"],
    "density_mixture": ["density_mixture"],
    "density_particle": ["density_particle"],
    "d32": ["d32"],
    "d43": ["d43"],
    "coordination_number": ["coordination_number"],
    "coordination_number_without_rattlers": ["coordination_number_without_rattlers"],  # not implemented yet
    "momentum_density": ["momentum_density"],
    "velocity": ["velocity"],
    "velocity_gradient": ["velocity_gradient"],
    "kinetic_tensor": ["kinetic_tensor"],
    "contact_tensor": ["contact_tensor"],
    "total_stress_tensor": ["total_stress_tensor_xy_dev", "total_stress_tensor_xy_dev_mag",
                            "total_stress_tensor_xyz", "total_stress_tensor_xyz_mag",
                            "total_stress_tensor_xyz_dev", "total_stress_tensor_xyz_dev_mag"],
    "fabric_tensor": ["fabric_tensor"],
    "shear_rate_tensor": ["shear_rate_tensor_xy", "shear_rate_tensor_xy_mag",
                          "shear_rate_tensor_xy_dev", "shear_rate_tensor_xy_dev_mag",
                          # 3D grids only
                          "shear_rate_tensor_xyz", "shear_rate_tensor_xyz_mag",
                          "shear_rate_tensor_xyz_dev", "shear_rate_tensor_xyz_dev_mag"],
    "pressure": ["pressure_xyz", "pressure_xy", "pressure_x", "pressure_y", "pressure_z"],
    "granular_temperature": ["granular_temperature_xyz", "granular_temperature_x",
                             "granular_temperature_y", "granular_temperature_z"],
    "inertial_number": ["inertial_number_Sxy_Pxyz_d43", "inertial_number_Sxy_Pxy_d43", "inertial_number_Sxy_Py_d43",
                        "inertial_number_Sxy_Pxyz_d32", "inertial_number_Sxy_Pxy_d32", "inertial_number_Sxy_Py_d32"],
    "frictional_coefficient": ["frictional_coefficient_Dxy_Pxyz", "frictional_coefficient_Dxy_Pxy",
                               "frictional_coefficient_Dxy_Py", "frictional_coefficient_Dxyz_Pxyz",
                               "frictional_coefficient_Dxyz_Pxy", "frictional_coefficient_Dxyz_Py",
                               "frictional_coefficient_01_Pxyz", "frictional_coefficient_01_Pxy",
                               "frictional_coefficient_01_Py"],
}


# ---------------------------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------------------------
def velocity_from_momentum(momentum_density: np.ndarray, density_mixture: np.ndarray) -> np.ndarray:
    """
    Coarse-grained (density-weighted) velocity, momentum density / mixture density.

    Inputs
    ------
    momentum_density : np.ndarray, shape (Npoints, 3) or (Npoints, Nphases+1, 3)
    density_mixture : np.ndarray, shape (Npoints,) or (Npoints, Nphases+1)

    Outputs
    -------
    velocity : np.ndarray, same shape as ``momentum_density``
    """
    return momentum_density / density_mixture[..., np.newaxis]


def bulk_velocity(velocity: np.ndarray, cg_calc_mode: str) -> np.ndarray:
    """Velocity of the bulk: the field itself (Monodisperse) or its phase index 0 (Polydisperse)."""
    return velocity if cg_calc_mode == "Monodisperse" else velocity[:, 0, :]


def velocity_gradient_fd(velocity: np.ndarray, cg_calc_mode: str, nodes, spacing) -> np.ndarray:
    """Gradient of the bulk velocity by finite differences on the grid, shape (Npoints, 3, 3)."""
    return secondary.compute_vector_bulk_gradient(bulk_velocity(velocity, cg_calc_mode), nodes, spacing)


def add_velocity_fields(fields: dict, *, fields_to_compute, cg_calc_mode: str, nodes, spacing) -> dict:
    """
    Fill in ``velocity`` and ``velocity_gradient`` (finite differences) where they are missing
    and needed. For a time step both were already computed in :meth:`CoarseGraining._compute_fields`,
    so this does nothing; for time averages it computes them from the averaged primary fields.

    Outputs
    -------
    fields : dict
        A copy of ``fields`` with ``velocity`` and ``velocity_gradient`` set (None if not needed).
    """
    fields = dict(fields)
    fields.setdefault("velocity", None)
    fields.setdefault("velocity_gradient", None)

    if (fields["velocity"] is None and "velocity" in fields_to_compute and fields.get("momentum_density") is not None):
        fields["velocity"] = velocity_from_momentum(fields["momentum_density"], fields["density_mixture"])

    if (fields["velocity_gradient"] is None and "velocity_gradient" in fields_to_compute and fields["velocity"] is not None):
        fields["velocity_gradient"] = velocity_gradient_fd(fields["velocity"], cg_calc_mode, nodes, spacing)
    
    return fields


# ---------------------------------------------------------------------------------------------
# derived fields
# ---------------------------------------------------------------------------------------------
def compute_derived_fields(fields: dict, *, fields_to_compute, phases: np.ndarray, grid_info: dict) -> dict:
    """
    Compute the derived fields specified in ``fields_to_compute`` from the primary fields.

    Inputs
    ------
    fields : dict
        Primary fields (keys :data:`PRIMARY_FIELDS`) plus ``velocity`` and ``velocity_gradient``,
        None where not computed.
    fields_to_compute : set
        Fields to compute, from :func:`get_fields_to_compute`.
    phases : np.ndarray, shape (Nphases, 2)
        Phase diameters (column 0) and densities (column 1), used by the inertial number.
    grid_info : dict
        Grid information; ``grid_info["grid_dimension"]`` decides whether 3D shear rates are computed.

    Outputs
    -------
    fields : dict
        A new dictionary with the input fields and the derived fields, keyed by output name
        (see :data:`EXPORT_GROUPS`).
    """
    out = dict(fields)
    rho = fields["density_mixture"]

    # particle density
    if "density_particle" in fields_to_compute:
        out["density_particle"] = rho / fields["volume_fraction"]
        print('  particle density done')

    # total stress tensor, its deviators and magnitudes
    if "total_stress_tensor" in fields_to_compute:
        sigma = fields["kinetic_tensor"] + fields["contact_tensor"]
        dev_xyz = secondary.compute_deviatoric_tensor(sigma[..., :, :])
        dev_xy = secondary.compute_deviatoric_tensor(sigma[..., :2, :2])
        out["total_stress_tensor_xyz"] = sigma
        out["total_stress_tensor_xyz_mag"] = secondary.compute_second_invariant(sigma, factor=0.5)
        out["total_stress_tensor_xyz_dev"] = dev_xyz
        out["total_stress_tensor_xyz_dev_mag"] = secondary.compute_second_invariant(dev_xyz, factor=0.5)
        out["total_stress_tensor_xy_dev"] = dev_xy
        out["total_stress_tensor_xy_dev_mag"] = secondary.compute_second_invariant(dev_xy, factor=0.5)
        print('  total stress done')

    # pressure
    if "pressure" in fields_to_compute:
        sigma = out["total_stress_tensor_xyz"]
        out["pressure_xyz"] = secondary.compute_pressure(sigma)
        out["pressure_xy"] = secondary.compute_pressure(sigma[..., 0:2, 0:2])
        out["pressure_x"] = sigma[..., 0, 0]
        out["pressure_y"] = sigma[..., 1, 1]
        out["pressure_z"] = sigma[..., 2, 2]
        print('  pressure done')

    # granular temperature
    if "granular_temperature" in fields_to_compute:
        fluct = fields["fluctuation_tensor"]
        out["granular_temperature_xyz"] = secondary.compute_granular_temperature(rho, fluct)
        for i, axis in enumerate("xyz"):
            out[f"granular_temperature_{axis}"] = secondary.compute_granular_temperature(rho, fluct[..., i, i])
        print('  granular temp done')

    # shear rate tensor, its deviator and magnitudes (2D, and 3D on 3D grids)
    if "shear_rate_tensor" in fields_to_compute:
        grad_v = fields["velocity_gradient"]
        components = {"xy": grad_v[..., 0:2, 0:2]}
        if grid_info["grid_dimension"] == 3:
            components["xyz"] = grad_v
        for name, g in components.items():
            rate = secondary.compute_shear_rate_tensor(g)
            rate_dev = secondary.compute_deviatoric_tensor(rate)
            out[f"shear_rate_tensor_{name}"] = rate
            out[f"shear_rate_tensor_{name}_mag"] = secondary.compute_second_invariant(rate, factor=2)
            out[f"shear_rate_tensor_{name}_dev"] = rate_dev
            out[f"shear_rate_tensor_{name}_dev_mag"] = secondary.compute_second_invariant(rate_dev, factor=2)
        print('  shear rate done')

    # inertial number: 2D shear rate magnitude, three pressures, two grain sizes
    if "inertial_number" in fields_to_compute:
        for d_name in ("d43", "d32"):
            for p_name in ("xyz", "xy", "y"):
                out[f"inertial_number_Sxy_P{p_name}_{d_name}"] = secondary.compute_inertial_number(
                    out["shear_rate_tensor_xy_mag"], out[f"pressure_{p_name}"], out["density_particle"],
                    fields[d_name], phases[:, 1], phases[:, 0])
        print('  inertial number done')

    # frictional coefficient: deviatoric stress magnitude (or shear stress) over pressure
    if "frictional_coefficient" in fields_to_compute:
        shear_stresses = {"Dxy": out["total_stress_tensor_xy_dev_mag"],
                          "Dxyz": out["total_stress_tensor_xyz_dev_mag"],
                          "01": out["total_stress_tensor_xyz"][..., 0, 1]}
        for s_name, tau in shear_stresses.items():
            for p_name in ("xyz", "xy", "y"):
                out[f"frictional_coefficient_{s_name}_P{p_name}"] = tau / out[f"pressure_{p_name}"]
        print('  mu done')

    return out


def select_exports(fields: dict, field_to_export: dict) -> dict:
    """
    Keep the fields requested in ``field_to_export``, in the order of :data:`EXPORT_GROUPS`.
    Keys that were not computed (None or absent, e.g. 3D shear rates on a 2D grid) are skipped.
    """
    return {key: fields[key]
            for flag, keys in EXPORT_GROUPS.items() if field_to_export.get(flag)
            for key in keys if fields.get(key) is not None}