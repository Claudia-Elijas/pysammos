r"""

This module provides functions to compute coarse-grained vector fields 
from particle data using weighted averaging over local neighborhoods.

Mathematically, the coarse-grained field \(\mathbf{F}(\mathbf{x})\) at a grid point \(\mathbf{x}\) 
is defined as follows:

Vector field (e.g. momentum density):
---------------
\[
\mathbf{v}(\mathbf{x}) = \sum_{i \in \text{neigh}(\mathbf{x})} w_i \, \mathbf{d}_i
\]

where:

- \(\mathbf{x}\) is a coarse-graining grid point,
- \(\text{neigh}(\mathbf{x})\) is the set of particles contributing to the grid point \(\mathbf{x}\),
- \(w_i\) is the coarse-graining weight for particle \(i\),
- \(\mathbf{d}_i\) is a vector particle property,


Functions 
---------
- `vector_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase)`
  Computes coarse-grained vector fields for polydisperse particles with
  an additional per-particle scaling factor.

- `vector_polydisperse(weights, visibility, grid_indices, Data, Phase)`
  Computes coarse-grained vector fields for polydisperse particles without
  scaling.

- `vector_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale)`
  Computes coarse-grained tensor fields for monodisperse particles with
  an additional scaling factor.

- `vector_monodisperse(weights, visibility, grid_indices, Data)`
  Computes coarse-grained vector fields for monodisperse particle mixtures without
  scaling.

Overview
--------
These functions transform particle-based quantities (e.g., mass, velocity
components, diameters) into grid-based fields by applying a coarse-graining
weighting scheme. The input typically consists of:

- **weights**: Coarse-graining kernel weights per visible particle–grid point
  interaction.
- **visibility**: Mapping from weight entries back to particle indices.
- **grid_indices**: Index offsets marking which particles contribute to each
  grid point (with padding at start and end).
- **Data**: Scalar quantities per particle.
- **Phase** (polydisperse only): Phase identifiers for multi-phase simulations.

Monodisperse vs. Polydisperse
-----------------------------
- **Monodisperse**: All particles are treated identically; output is a single
  tensor field per grid point.
- **Polydisperse**: Particles are grouped by phase; output contains both
  per-phase fields and a total field.

Functions with `_scaled` apply an additional per-particle multiplicative factor
(`Data_scale`), useful for scaling properties before coarse-graining.

  Terminology
-----------
- **N_particles**: Number of particles in the simulation.
- **N_vis**: Number of particle–grid point interactions (visible weights).
- **N_points**: Number of grid points (excluding padding).
- **Phase**: Integer labels identifying particle classes (e.g., material type).

Performance Notes
-----------------
- All functions are Numba-jitted (`@njit`) with explicit type signatures.
- `prange` is used to parallelize over grid points.
- Temporary arrays are allocated per grid point; results are accumulated into
  output arrays in a thread-safe manner.
- Arrays must have consistent types and shapes to avoid recompilation overhead.

"""

import numpy as np
from numba import njit, prange, int32, float32, float64

# ======================================================================
# Polydisperse vector coarse-graining functions
# ======================================================================

@njit(float64[:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:], int32[:]),parallel=True)
def vector_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase):

    """
    Compute coarse-grained vector fields for polydisperse systems with an additional scaling factor.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data : (N_particles,3) float32 array
        Vector quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Returns
    -------
    CG_Field : (N_points, N_phases + 1, 3) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """

    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_vector = np.zeros((Nphases, 3))
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                ds = Data_scale[idx]
                phase_vector[phase, 0] += w * d[0] * ds
                phase_vector[phase, 1] += w * d[1] * ds
                phase_vector[phase, 2] += w * d[2] * ds
        for p in range(Nphases):
            vec = phase_vector[p]
            for d in range(3):
                CG_Field[g, p + 1, d] = vec[d]
                CG_Field[g, 0, d] += vec[d]
    return CG_Field

@njit(float64[:,:,:](float64[:],int32[:], int32[:], float32[:,:], int32[:]),parallel=True)
def vector_polydisperse(weights, visibility, grid_indices, Data, Phase):

    """
    Compute coarse-grained vector fields for polydisperse systems.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data : (N_particles,3) float32 array
        Vector quantity per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Returns
    -------
    CG_Field : (N_points, N_phases + 1, 3) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """

    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_vector = np.zeros((Nphases, 3))
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                phase_vector[phase, 0] += w * d[0]
                phase_vector[phase, 1] += w * d[1]
                phase_vector[phase, 2] += w * d[2]
        for p in range(Nphases):
            vec = phase_vector[p]
            for d in range(3):
                CG_Field[g, p + 1, d] = vec[d]
                CG_Field[g, 0, d] += vec[d]
    return CG_Field

# ======================================================================
# Monodisperse vector coarse-graining functions
# ======================================================================

@njit(float64[:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:]), parallel=True)
def vector_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale):

    """
    Compute coarse-grained vector fields for monodisperse systems with an additional scaling factor.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data : (N_particles,3) float32 array
        Vector quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.

    Returns
    -------
    CG_Field : (N_points, 3) float64 array
        Coarse-grained scalar field.
    """

    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3), dtype=np.float64)  # No phase info in monodisperse case
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        vec = np.zeros(3)
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            ds = Data_scale[idx]
            vec[0] += w * d[0] * ds
            vec[1] += w * d[1] * ds
            vec[2] += w * d[2] * ds
        for d in range(3):
            CG_Field[g, d] = vec[d]

    return CG_Field

@njit(float64[:,:](float64[:],int32[:], int32[:], float32[:,:]),parallel=True)
def vector_monodisperse(weights, visibility, grid_indices, Data):

    """
    Compute coarse-grained vector fields for monodisperse systems.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data : (N_particles,3) float32 array
        Vector quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Returns
    -------
    CG_Field : (N_points, N_phases + 1, 3) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """

    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3), dtype=np.float64)  # No phase info in monodisperse case
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        vec = np.zeros(3)
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            vec[0] += w * d[0]
            vec[1] += w * d[1]
            vec[2] += w * d[2]
        for d in range(3):
            CG_Field[g, d] = vec[d]
    return CG_Field