"""
Module for Generic Scalar, Vector, Tensor, and Kinetic Tensor Calculations
==========================================================================

This module provides a unified interface for computing scalar, vector, 
and tensor quantities for both *monodisperse* and *polydisperse* systems, 
with or without data scaling. It also supports kinetic tensor calculations 
with interpolation.

The functions here act as generic wrappers that delegate to specialized 
implementations based on the `cg_calc_mode` parameter.

Modes
-----
- **Monodisperse**: All particles or elements are of the same size/type.
- **Polydisperse**: Particles or elements have varying sizes/types.
"""

from typing import Tuple, Optional
import numpy as np
from .scalars import (scalar_polydisperse, scalar_polydisperse_scaled, scalar_monodisperse, scalar_monodisperse_scaled) 
from .vectors import (vector_polydisperse, vector_polydisperse_scaled, vector_monodisperse, vector_monodisperse_scaled)  
from .tensors import (tensor_polydisperse, tensor_polydisperse_scaled, tensor_monodisperse, tensor_monodisperse_scaled, 
                      kinetic_tensor_interpolation_polydisperse, kinetic_tensor_interpolation_monodisperse)


# generic scalar
def scalar(weights:np.ndarray, visibility:np.ndarray, grid_indices:np.ndarray, 
           Data:np.ndarray, Data_scale:Optional[np.ndarray], Phase:Optional[np.ndarray], 
           cg_calc_mode:str)->np.ndarray:
    """
    Compute a SCALAR quantity for either monodisperse or polydisperse systems.

    Parameters
    ----------
    weights : ndarray, shape(N,).
        Array of weights for each element/particle.
    visibility : ndarray, shape(N,).
        Visibility mask array.
    grid_indices : ndarray, shape(N,).
        Indices mapping elements to grid locations.
    Data : ndarray, shape(N,).
        Primary scalar data values.
    Data_scale : ndarray or None, shape(N,).
        Scaling factors for the data. If None, unscaled computation is used.
    Phase : ndarray or None, shape(N,).
        Phase values, used only in polydisperse mode.
    cg_calc_mode : {'Monodisperse', 'Polydisperse'}
        Calculation mode.

    Returns
    -------
    ndarray
        Computed scalar values mapped to the grid. Options:
        - `cg_calc_mode`='Monodisperse': np.ndarray, shape(Npoints,)
        - `cg_calc_mode`='Polydisperse': np.ndarray, shape(Npoints, Nphases+1)

    """
    if cg_calc_mode == 'Monodisperse':
        if Data_scale is None:
            return scalar_monodisperse(weights, visibility, grid_indices, Data)
        else:
            return scalar_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        if Data_scale is None:
            return scalar_polydisperse(weights, visibility, grid_indices, Data, Phase)
        else:
            return scalar_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")  

# generic vector
def vector(weights:np.ndarray, visibility:np.ndarray, grid_indices:np.ndarray, Data:np.ndarray, 
           Data_scale:Optional[np.ndarray], Phase:Optional[np.ndarray], 
           cg_calc_mode:str)->np.ndarray:
    """
    Compute a VECTOR quantity for either monodisperse or polydisperse systems.

    Parameters
    ----------
    weights : ndarray, shape(N,).
        Array of weights for each element/particle.
    visibility : ndarray, shape(N,).
        Visibility mask array.
    grid_indices : ndarray, shape(N,).
        Indices mapping elements to grid locations.
    Data : ndarray, shape(N,3).
        Primary scalar data values.
    Data_scale : ndarray or None, shape(N,).
        Scaling factors for the data. If None, unscaled computation is used.
    Phase : ndarray or None, shape(N,).
        Phase values, used only in polydisperse mode.
    cg_calc_mode : {'Monodisperse', 'Polydisperse'}
        Calculation mode.

    Returns
    -------
    ndarray
        Computed vector values mapped to the grid. Options:
        - `cg_calc_mode`='Monodisperse': np.ndarray, shape(Npoints, 3)
        - `cg_calc_mode`='Polydisperse': np.ndarray, shape(Npoints, Nphases+1, 3)

    """
    if cg_calc_mode == 'Monodisperse':
        if Data_scale is None:
            return vector_monodisperse(weights, visibility, grid_indices, Data)
        else:
            return vector_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        if Data_scale is None:
            return vector_polydisperse(weights, visibility, grid_indices, Data, Phase)
        else:
            return vector_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")

# generic tensor
def tensor(weights:np.ndarray, visibility:np.ndarray, grid_indices:np.ndarray,
           Data1:np.ndarray, Data2:np.ndarray, Data_scale:Optional[np.ndarray], 
           Phase:Optional[np.ndarray], cg_calc_mode:str)->np.ndarray:
    """
    Compute a TENSOR quantity for either monodisperse or polydisperse systems.

    Parameters
    ----------
    weights : ndarray, shape(N,).
        Array of weights for each element/particle.
    visibility : ndarray, shape(N,).
        Visibility mask array.
    grid_indices : ndarray, shape(N,).
        Indices mapping elements to grid locations.
    Data1 : ndarray, shape(N,3).
        Primary scalar data values.
    Data2 : ndarray, shape(N,3).
        Primary scalar data values.
    Data_scale : ndarray or None, shape(N,).
        Scaling factors for the data. If None, unscaled computation is used.
    Phase : ndarray or None, shape(N,).
        Phase values, used only in polydisperse mode.
    cg_calc_mode : {'Monodisperse', 'Polydisperse'}
        Calculation mode.

    Returns
    -------
    ndarray
        Computed tensor values mapped to the grid. Options:
        - `cg_calc_mode`='Monodisperse': np.ndarray, shape(Npoints, 3, 3)
        - `cg_calc_mode`='Polydisperse': np.ndarray, shape(Npoints, Nphases+1, 3, 3)

    """
    if cg_calc_mode == 'Monodisperse':
        if Data_scale is None:
            return tensor_monodisperse(weights, visibility, grid_indices, Data1, Data2)
        else:
            return tensor_monodisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        if Data_scale is None:
            return tensor_polydisperse(weights, visibility, grid_indices, Data1, Data2, Phase)
        else:
            return tensor_polydisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")

# kinetic tensor including interpolation
def kinetic_tensor(weights:np.ndarray, visibility:np.ndarray, grid_indices:np.ndarray,
                Displacement:np.ndarray, Particle_Velocity:np.ndarray, Particle_Mass:np.ndarray, 
                Velocity_Field:np.ndarray, Velocity_Field_Gradient:np.ndarray, 
                phase_array:np.ndarray, cg_calc_mode:str)->np.ndarray:
    """
    Compute the kinetic tensor, including interpolation, for either monodisperse or polydisperse systems.

    Parameters
    ----------
    weights : ndarray, shape (N,).
        Array of weights for each element/particle.
    visibility : ndarray, shape (N,).
        Visibility mask array.
    grid_indices : ndarray, shape (N,).
        Indices mapping elements to grid locations.
    Displacement : ndarray, shape (N,3).
        Displacement vectors for particles/elements.
    Particle_Velocity : ndarray, shape (Nparticles,3).
        Velocities of particles.
    Particle_Mass : ndarray, shape (Nparticles,3).
        Mass of particles.
    Velocity_Field : ndarray, shape (Npoints,3).
        Velocity field on the grid.
    Velocity_Field_Gradient : ndarray, shape (Npoints,3,3).
        Gradient of the velocity field.
    phase_array : ndarray, shape (Nparticles).
        Phase values, used only in polydisperse mode.
    cg_calc_mode : {'Monodisperse', 'Polydisperse'}
        Calculation mode.

    Returns
    -------
    ndarray
        Computed kinetic tensor values mapped to the grid. Options:
        - `cg_calc_mode`='Monodisperse': np.ndarray, shape(Npoints, 3, 3)
        - `cg_calc_mode`='Polydisperse': np.ndarray, shape(Npoints, Nphases+1, 3, 3)
    """
    if cg_calc_mode == 'Monodisperse':
        return kinetic_tensor_interpolation_monodisperse(weights, visibility, grid_indices, Displacement,
                                             Particle_Velocity, Particle_Mass, 
                                             Velocity_Field, Velocity_Field_Gradient)
    elif cg_calc_mode == 'Polydisperse':
        return kinetic_tensor_interpolation_polydisperse(weights, visibility, grid_indices, Displacement,
                                Particle_Velocity, Particle_Mass, 
                                Velocity_Field, Velocity_Field_Gradient, 
                                phase_array)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")
    
