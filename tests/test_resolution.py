"""
Tests for the coarse-graining resolution. Run from the repository root with

    python -m pytest tests
"""
import numpy as np
import pytest

from pysammos.coarse_graining import CoarseGraining


def make_cg(tmp_path):
    grid_info = dict(grid_dimension=3, grid_axes="xyz", automatic_grid=True, x_min=None, x_max=None,
                     y_min=None, y_max=None, z_min=None, z_max=None, x_transect=None, y_transect=None,
                     z_transect=None, x_axis_periodic=False, y_axis_periodic=False, z_axis_periodic=False)
    return CoarseGraining(particle_path="", contacts_path="", output_path=f"{tmp_path}/",
                          start_timestep=0, end_timestep=0, dt_time_step=1, t_pad=4,
                          DEM_software="mfix", DEM_keymap={}, grid_info=grid_info, weight_function="Lucy",
                          fields_to_export={}, ignore_phases=True, vkthdf_output=False, h5_output=False)


def test_default_diameter_is_d32(tmp_path):
    cg = make_cg(tmp_path)
    # three sizes in equal numbers, 1:1:1 by count
    diameter = np.repeat([0.64e-3, 0.80e-3, 0.96e-3], 100)
    cg.get_particle_size_statistics(diameter, 2500.0 * np.pi / 6 * diameter ** 3)
    assert np.isclose(cg.d32, np.sum(diameter ** 3) / np.sum(diameter ** 2))
    cg.set_resolution()
    assert np.isclose(cg.w, 0.75 * cg.d32)
    assert np.isclose(cg.c, 2 * cg.w)  # Lucy kernel
    cg.set_resolution(cg.d43, w_mult=1.0)
    assert np.isclose(cg.w, cg.d43)


def test_default_diameter_needs_size_statistics(tmp_path):
    cg = make_cg(tmp_path)
    with pytest.raises(ValueError):
        cg.set_resolution()
