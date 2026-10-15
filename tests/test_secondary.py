"""
Tests for pysammos.macroscopic_fields.gridded.secondary. Run from the repository root with

    python -m pytest tests
"""
import numpy as np

from pysammos.macroscopic_fields.gridded import secondary


def test_monodisperse_granular_temperature_from_full_tensor():
    density = np.array([1500.0, 1200.0, 0.0])
    tensor = np.zeros((3, 3, 3))
    tensor[:, 0, 0] = [3.0, 1.0, 0.0]
    tensor[:, 1, 1] = [6.0, 2.0, 0.0]
    tensor[:, 2, 2] = [9.0, 3.0, 0.0]
    temperature = secondary.compute_granular_temperature(density, tensor)
    assert np.allclose(temperature, [6.0 / 1500.0, 2.0 / 1200.0, 0.0])
