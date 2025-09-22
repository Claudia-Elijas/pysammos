r"""
This module provides functions to compute coarse-grained tensor fields 
from particle data using weighted averaging over local neighborhoods.

Mathematically, the coarse-grained field :math:`\mathbf{T}(\mathbf{x})` at a grid point :math:`\mathbf{x}` 
is defined as follows:

.. math::

    \mathbf{T}(\mathbf{x}) = \sum_{i \in \text{neigh}(\mathbf{x})} w_i \, \mathbf{d}_i \otimes \mathbf{d}_i

where:

    - :math:`\mathbf{x}` is a coarse-graining grid point,
    - :math:`\text{neigh}(\mathbf{x})` is the set of particles contributing to the grid point :math:`\mathbf{x}`,
    - :math:`w_i` is the coarse-graining weight for particle :math:`i`,
    - :math:`\mathbf{d}_i` is a vector particle property,
    - :math:`\otimes` denotes the outer product.



Functions

    - :func:`tensor_polydisperse_scaled`: Computes coarse-grained tensor fields for polydisperse particles with
      an additional per-particle scaling factor.
    - :func:`tensor_polydisperse`: Computes coarse-grained tensor fields for polydisperse particles without
      scaling.
    - :func:`tensor_monodisperse_scaled`: Computes coarse-grained tensor fields for monodisperse particles with
      an additional scaling factor.
    - :func:`tensor_monodisperse`: Computes coarse-grained tensor fields for monodisperse particle mixtures without
      scaling.
    - :func:`kinetic_tensor_interpolation_polydisperse`: Computes coarse-grained kinetic tensor for polydisperse mixtures 
      by interpolating particle velocity fluctuations relative to a continuum velocity field.
    - :func:`kinetic_tensor_interpolation_monodisperse`: Computes coarse-grained kinetic tensor for monodisperse mixtures 
      by interpolating particle velocity fluctuations relative to a continuum velocity field.


Overview

These functions transform particle-based quantities (e.g., mass, velocity
components, diameters) into grid-based fields by applying a coarse-graining
weighting scheme. The input typically consists of:

    - **weights**: Coarse-graining kernel weights per visible particle–grid point interaction.
    - **visibility**: Mapping from weight entries back to particle indices.
    - **grid_indices**: Index offsets marking which particles contribute to each grid point (with padding at start and end).
    - **Data**: Scalar quantities per particle.
    - **Phase** (polydisperse only): Phase identifiers for multi-phase simulations.

Monodisperse vs. Polydisperse

    - **Monodisperse**: All particles are treated identically; output is a single tensor field per grid point.
    - **Polydisperse**: Particles are grouped by phase; output contains both per-phase fields and a total field.

Functions with `_scaled` apply an additional per-particle multiplicative factor
(`Data_scale`), useful for scaling properties before coarse-graining.

Terminology

    - :math:`N_\mathrm{particles}`: Number of particles in the simulation.
    - :math:`N_\mathrm{vis}`: Number of particle–grid point interactions (visible weights).
    - :math:`N_\mathrm{points}`: Number of grid points (excluding padding).
    - :math:`\mathrm{Phase}`: Integer labels identifying particle classes (e.g., material type).

Performance Notes

    - All functions are Numba-jitted (`@njit`) with explicit type signatures.
    - `prange` is used to parallelize over grid points.
    - Temporary arrays are allocated per grid point; results are accumulated into output arrays in a thread-safe manner.
    - Arrays must have consistent types and shapes to avoid recompilation overhead.

"""

import numpy as np
from numba import njit, prange, int32, float32, float64


# ======================================================================
# Polydisperse tensor coarse-graining functions
# ======================================================================
@njit(float64[:,:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:,:], float32[:], int32[:]), parallel=True)
def tensor_polydisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase):
    """
    Compute coarse-grained tensor fields for polydisperse systems with an additional scaling factor.

    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data1 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data2 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Returns
    -------
    CG_Field : (N_points, N_phases + 1, 3, 3) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3, 3), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_tensor = np.zeros((Nphases, 3, 3))
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d1 = Data1[idx]
                d2 = Data2[idx]
                ds = Data_scale[idx]
                for j in range(3):
                    for k in range(3):
                        phase_tensor[phase, j, k] += w * d1[j] * d2[k] * ds
        for p in range(Nphases):
            tensor = phase_tensor[p]
            for j in range(3):
                for k in range(3):
                    CG_Field[g, p + 1, j, k] = tensor[j, k]
                    CG_Field[g, 0, j, k] += tensor[j, k]
    return CG_Field

@njit(float64[:,:,:,:](float64[:], int32[:], int32[:], float32[:,:], float32[:,:], int32[:]), parallel=True)
def tensor_polydisperse(weights, visibility, grid_indices, Data1, Data2, Phase):
    """
    Compute coarse-grained tensor fields for polydisperse systems.

    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data1 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data2 : (N_particles,3) float32 array
        Vector quantity per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Outputs
    -------
    CG_Field : (N_points, N_phases + 1, 3, 3) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """

    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3, 3), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_tensor = np.zeros((Nphases, 3, 3))
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d1 = Data1[idx]
                d2 = Data2[idx]
                for j in range(3):
                    for k in range(3):
                        phase_tensor[phase, j, k] += w * d1[j] * d2[k]
        for p in range(Nphases):
            tensor = phase_tensor[p]
            for j in range(3):
                for k in range(3):
                    CG_Field[g, p + 1, j, k] = tensor[j, k]
                    CG_Field[g, 0, j, k] += tensor[j, k]
    return CG_Field

# ======================================================================
# Monodisperse tensor coarse-graining functions
# ======================================================================
@njit(float64[:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:,:], float32[:]), parallel=True)
def tensor_monodisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale):
    """
    Compute coarse-grained tensor fields for monodisperse systems with an additional scaling factor.

    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data1 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data2 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.

    Outputs
    -------
    CG_Field : (N_points, 3, 3) float64 array
        Coarse-grained scalar field.
    """
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3, 3), dtype=np.float64)  # Only one tensor per grid point
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        tensor = np.zeros((3, 3))
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d1 = Data1[idx]
            d2 = Data2[idx]
            ds = Data_scale[idx]
            for j in range(3):
                for k in range(3):
                    tensor[j, k] += w * d1[j] * d2[k] * ds
        for j in range(3):
            for k in range(3):
                CG_Field[g, j, k] = tensor[j, k]

    return CG_Field

@njit(float64[:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:,:]), parallel=True)
def tensor_monodisperse(weights, visibility, grid_indices, Data1, Data2):
    """
    Compute coarse-grained tensor fields for monodisperse systems.

    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data1 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data2 : (N_particles,3) float32 array
        Vector quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Outputs
    -------
    CG_Field : (N_points, N_phases + 1, 3, 3) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3, 3), dtype=np.float64)  # Only one tensor per grid point
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        tensor = np.zeros((3, 3))
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d1 = Data1[idx]
            d2 = Data2[idx]
            for j in range(3):
                for k in range(3):
                    tensor[j, k] += w * d1[j] * d2[k]
        for j in range(3):
            for k in range(3):
                CG_Field[g, j, k] = tensor[j, k]

    return CG_Field

# ======================================================================
# Kinetic Tensor coarse-graining functions
# ======================================================================
@njit(float64[:,:,:,:](float64[:],int32[:], int32[:], float64[:,:], float32[:,:], float32[:], float64[:,:], float64[:,:,:], int32[:]), parallel=True)
def kinetic_tensor_interpolation_polydisperse(weights, visibility, grid_indices, displacement,
                                     Particle_Velocity, Particle_Mass, 
                                     Velocity_Field, Velocity_Field_Gradient, 
                                     phase_array): 
    
    r"""
    Compute the kinetic stress tensor for a polydisperse system by interpolating
    particle velocity fluctuations relative to a continuum velocity field.

    The tensor is computed separately for each phase and summed to form a total tensor. For each grid point \( g \) and phase \( p \), the kinetic tensor is:

    .. math::

        K_{g,p} = \sum_{i \in I_g^{(p)}} m_i w_i (v_i - v_g^{\text{interp}}) \otimes (v_i - v_g^{\text{interp}})

    where:

      - :math:`I_g^{(p)}` is the set of particles of phase p contributing to grid point g
      - :math:`m_i` is the mass of particle i
      - :math:`w_i` is the kernel weight of particle i for grid point g
      - :math:`v_i` is the velocity of particle i
      - :math:`v_g^{interp} = v_g + G_g \cdot r_{g,i}` is the interpolated continuum velocity at particle i position
      - :math:`r_{g,i}` is the displacement vector from grid point g to particle i
      - :math:`\otimes` denotes the outer product


    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    displacement : (N_vis, 3) float64 array
        Displacement vectors from grid points to visible particles.
    Particle_Velocity : (N_particles, 3) float64 array
        Velocity vectors of all particles.
    Particle_Mass : (N_particles,) float64 array
        Mass of each particle.
    Velocity_Field : (N_points, 3) float64 array
        Velocity field values at each grid point.
    Velocity_Field_Gradient : (N_points, 3, 3) float32 array
        Velocity gradient tensors at each grid point.
    phase_array : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Outputs
    -------
    KineticTensor : (N_points, N_phases + 1, 3, 3) float64 array
        Kinetic stress tensor per grid point and phase. Index 0 is total over phases.
    """
    
    Ngridpoints = len(grid_indices) - 2
    Nphases = 0
    for i in range(len(phase_array)):
        if phase_array[i] > Nphases:
            Nphases = phase_array[i]
    Nphases += 1

    KineticTensor = np.zeros((Ngridpoints, Nphases + 1, 3, 3), dtype=np.float64)

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        Velocity_Field_g = Velocity_Field[g]
        GradV_g = Velocity_Field_Gradient[g]

        phase_tensor = np.zeros((Nphases, 3, 3), dtype=np.float64)

        for i in range(start, end):
            idx = visibility[i]
            phase = phase_array[idx]

            if 0 <= phase < Nphases:
                r_ri = displacement[i]
                w = weights[i]
                v_particle = Particle_Velocity[idx]
                m_particle = Particle_Mass[idx]

                # interpolated_velocity = np.zeros(3, dtype=np.float64)
                # for j in range(3):
                #     interpolated_velocity[j] = Velocity_Field_g[j]
                #     for k in range(3):
                #         interpolated_velocity[j] += r_ri[k] * GradV_g[k, j]
    
                interpolated_velocity = np.zeros(3, dtype=np.float64)
                for j in range(3):
                    interpolated_velocity_j = Velocity_Field_g[j]
                    for k in range(3):
                        interpolated_velocity_j += r_ri[k] * GradV_g[k, j]
                    interpolated_velocity[j] = interpolated_velocity_j

                velocity_fluctuation = np.zeros(3, dtype=np.float64)
                for j in range(3):
                    velocity_fluctuation[j] = v_particle[j] - interpolated_velocity[j]

                for j in range(3):
                    for k in range(3):
                        phase_tensor[phase, j, k] += m_particle * velocity_fluctuation[j] * velocity_fluctuation[k] * w

        for p in range(Nphases):
            for j in range(3):
                for k in range(3):
                    KineticTensor[g, p + 1, j, k] = phase_tensor[p, j, k]
                    KineticTensor[g, 0, j, k] += phase_tensor[p, j, k]

    return KineticTensor

# kinetic tensor for monodisperse case
@njit(float64[:,:,:](float64[:],int32[:], int32[:], float64[:,:], float32[:,:], float32[:], float64[:,:], float64[:,:,:]),parallel=True)
def kinetic_tensor_interpolation_monodisperse(weights, visibility, grid_indices, displacement,
                                             Particle_Velocity, Particle_Mass, 
                                             Velocity_Field, Velocity_Field_Gradient):
    
    """
    Compute the kinetic stress tensor for a monodisperse system by interpolating
    particle velocity fluctuations relative to a continuum velocity field.

    Mathematical formulation:

    For each grid point \( g \), the kinetic tensor is:

    .. math::

        \mathbf{K}_{g} = \sum_{i \in I_g} m_i w_i (\mathbf{v}_i - \mathbf{v}_g^{\text{interp}}) \otimes (\mathbf{v}_i - \mathbf{v}_g^{\text{interp}})
    where:
      - :math:`I_g` is the set of particles contributing to grid point \( g \)
      - :math:`m_i` is the mass of particle \( i \)
      - :math:`w_i` is the kernel weight of particle \( i \) for grid point \( g \)
      - :math:`\mathbf{v}_i` is the velocity of particle \( i \)
      - :math:`\mathbf{v}_g^{\text{interp}} = \mathbf{v}_g + \mathbf{G}_g \cdot \mathbf{r}_{g,i}` is the interpolated continuum velocity at particle \( i \)
      - :math:`\mathbf{r}_{g,i}` is the displacement vector from grid point \( g \) to particle \( i \)

    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    displacement : (N_vis, 3) float64 array
        Displacement vectors from grid points to visible particles.
    Particle_Velocity : (N_particles, 3) float64 array
        Velocity vectors of all particles.
    Particle_Mass : (N_particles,) float32 array
        Mass of each particle.
    Velocity_Field : (N_points, 3) float64 array
        Velocity field values at each grid point.
    Velocity_Field_Gradient : (N_points, 3, 3) float64 array
        Velocity gradient tensors at each grid point.

    Outputs
    -------
    KineticTensor : (N_points, 3, 3) float64 array
        Kinetic stress tensor per grid point.
    """

    Ngridpoints = len(grid_indices) - 2
    KineticTensor = np.zeros((Ngridpoints, 3, 3), dtype=np.float64)  # Only one tensor per grid point

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        tensor = np.zeros((3, 3))

        Velocity_Field_g = Velocity_Field[g]
        GradV_g = Velocity_Field_Gradient[g]

        for i in range(start, end):
            idx = visibility[i]
            r_ri = displacement[i]
            w = weights[i]
            v_particle = Particle_Velocity[idx]
            m_particle = Particle_Mass[idx]

            # Interpolate velocity
            # interpolated_velocity = Velocity_Field_g.copy()
            # for j in range(3):
            #     for k in range(3):
            #         interpolated_velocity[j] += r_ri[k] * GradV_g[k, j]
            interpolated_velocity = np.zeros(3, dtype=np.float64)
            for j in range(3):
                interpolated_velocity_j = Velocity_Field_g[j]
                for k in range(3):
                    interpolated_velocity_j += r_ri[k] * GradV_g[k, j]
                interpolated_velocity[j] = interpolated_velocity_j

            # Velocity fluctuation
            v_fluc = np.zeros(3)
            for j in range(3):
                v_fluc[j] = v_particle[j] - interpolated_velocity[j]

            # Outer product for tensor
            for j in range(3):
                for k in range(3):
                    tensor[j, k] += m_particle * v_fluc[j] * v_fluc[k] * w

        for j in range(3):
            for k in range(3):
                KineticTensor[g, j, k] = tensor[j, k]

    return KineticTensor
