r"""

This module provides functions to compute coarse-grained scalar fields 
from particle data using weighted averaging over local neighborhoods.

Mathematically, the coarse-grained field \(\mathbf{F}(\mathbf{x})\) at a grid point \(\mathbf{x}\) is defined as follows:

Scalar field \(\phi(\mathbf{x})\):
---------------------------------
\[
\phi(\mathbf{x}) = \sum_{i \in \text{neigh}(\mathbf{x})} w_i \, d_i
\]

where: 

- \(\mathbf{x}\) is a coarse-graining grid point,
- \(\text{neigh}(\mathbf{x})\) is the set of particles contributing to the grid point \(\mathbf{x}\),
- \(w_i\) is the coarse-graining weight for particle \(i\),
- \(d_i\) is a scalar particle property,

Functions 
---------
- `scalar_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase)`
  Computes coarse-grained scalar fields for polydisperse particles with
  an additional per-particle scaling factor.

- `scalar_polydisperse(weights, visibility, grid_indices, Data, Phase)`
  Computes coarse-grained scalar fields for polydisperse particles without
  scaling.

- `scalar_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale)`
  Computes coarse-grained scalar fields for monodisperse particles with
  an additional scaling factor.

- `scalar_monodisperse(weights, visibility, grid_indices, Data)`
  Computes coarse-grained scalar fields for monodisperse particles without
  scaling.

- `mean_grainsize(weights, visibility, grid_indices, Data, n_flag)`
  Computes a weighted mean grain size metric, where `n_flag` controls the
  moment order (e.g., 2 for area, 3 for volume).

- `scalar_x_volume(weights, visibility, grid_indices, Data)`
  Computes the coarse-grained average scalar field weighted by particle volume.


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
This module implements high-performance, parallelized functions for computing
coarse-grained scalar fields from discrete particle data using Numba `@njit`
and `prange` for loop-level parallelism. It supports both monodisperse and
polydisperse particle systems, with optional per-particle scaling factors.

- **Monodisperse**: All particles are treated identically; output is a single
  scalar field per grid point.
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
# Polydisperse scalar coarse-graining functions
# ======================================================================

@njit(float64[:, :](float64[:],int32[:], int32[:], float32[:], float32[:], int32[:]), parallel=True)
def scalar_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase):
    """
    Compute coarse-grained scalar fields for polydisperse systems with an additional scaling factor.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    Data : (N_particles,) float32 array
        Scalar quantity per particle (e.g., mass, velocity component).
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Returns
    -------
    CG_Field : (N_points, N_phases + 1) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """
    Ngridpoints = len(grid_indices) - 2  # because it's padded [0], [len(flat_visibility)]
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_sum = np.zeros(Nphases)
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                ds = Data_scale[idx]
                phase_sum[phase] += w * d * ds
        for p in range(Nphases):
            CG_Field[g, p + 1] = phase_sum[p]
            CG_Field[g, 0] += phase_sum[p]
    return CG_Field

@njit(float64[:, :](float64[:],int32[:], int32[:], float32[:], int32[:]),parallel=True)
def scalar_polydisperse(weights, visibility, grid_indices, Data, Phase):
    """
    Compute coarse-grained scalar fields for polydisperse systems.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
    Data : (N_particles,) float32 array
        Scalar quantity per particle (e.g., mass, velocity component).
    Phase : (N_particles,) int32 array
        Phase index (0..P-1) for each particle.

    Returns
    -------
    CG_Field : (N_points, N_phases + 1) float64 array
        Coarse-grained scalar field per phase (columns 1..P) and total (column 0).
    """
    Ngridpoints = len(grid_indices) - 2  # because it's padded [0], [len(flat_visibility)]
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_sum = np.zeros(Nphases)  # Local buffer for each grid point
        for i in range(start, end): # loop over particles
            idx = visibility[i] # get the particle index
            phase = Phase[idx] # get the phase
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                phase_sum[phase] += w * d # weighted sum
        for p in range(Nphases): # store the weighted sum in the CG_Field
            CG_Field[g, p + 1] = phase_sum[p]
            CG_Field[g, 0] += phase_sum[p]
    return CG_Field

# ======================================================================
# Monodisperse scalar coarse-graining functions
# ======================================================================

@njit(float64[:](float64[:],int32[:], int32[:], float32[:], float32[:]),parallel=True)
def scalar_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale):
    """
    Compute coarse-grained scalar fields for monodisperse systems with an additional scaling factor.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
    Data : (N_particles,) float32 array
        Scalar quantity per particle.
    Data_scale : (N_particles,) float32 array
        Additional scaling factor per particle.

    Returns
    -------
    CG_Field : (N_points,) float64 array
        Coarse-grained scalar field for each grid point.
    """

    Ngridpoints = len(grid_indices) - 2  # Padding assumed: grid_indices[0] = 0, grid_indices[-1] = len(visibility)
    CG_Field = np.zeros(Ngridpoints, dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        val = 0.0
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            ds = Data_scale[idx]
            val += w * d * ds
        CG_Field[g] = val
    return CG_Field

@njit(float64[:](float64[:],int32[:], int32[:], float32[:]),parallel=True)
def scalar_monodisperse(weights, visibility, grid_indices, Data):
    """
    Compute coarse-grained scalar fields for monodisperse systems.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
    Data : (N_particles,) float32 array
        Scalar quantity per particle.

    Returns
    -------
    CG_Field : (N_points,) float64 array
        Coarse-grained scalar field for each grid point.
    """

    Ngridpoints = len(grid_indices) - 2  # Padding assumed: grid_indices[0] = 0, grid_indices[-1] = len(visibility)
    CG_Field = np.zeros(Ngridpoints, dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        val = 0.0
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            val += w * d
        CG_Field[g] = val
    return CG_Field

# ======================================================================
# Additional scalar-based calculations
# ======================================================================

@njit(float64[:](float64[:],int32[:], int32[:], float32[:], int32),parallel=True)
def mean_grainsize(weights, visibility, grid_indices, Data, n_flag):
    """
    Compute mean grain size using a weighted average with a size exponent.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible particle.
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
    Data : (N_particles,) float32 array
        Particle diameters or characteristic sizes.
    n_flag : int
        Exponent flag:
        - 3 for volume-weighted mean size.
        - 2 for area-weighted mean size.

    Returns
    -------
    CG_Field : (N_points,) float64 array
        Mean grain size for each grid point.
    """

    # nflag = 3 for VOLUME, nflag = 2 for AREA
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros(Ngridpoints)

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        numerator = 0.0
        denominator = 0.0

        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            numerator += w * d**(n_flag + 1)
            denominator += w * d**n_flag

        if denominator != 0.0:
            CG_Field[g] = numerator / denominator
        else:
            CG_Field[g] = 0.0

    return CG_Field

@njit(float64[:](float64[:],int32[:], int32[:], float32[:]),parallel=True)
def scalar_x_volume(weights, visibility, grid_indices, Data):
    """
    Compute the average of a scalar quantity weighted by particle volume.

    Parameters
    ----------
    weights : (N_vis,) float64 array
        Coarse-graining weights (often proportional to particle volume).
    visibility : (N_vis,) int32 array
        Particle indices corresponding to each entry in the weights array.
    grid_indices : (N_grid + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
    Data : (N_particles,) float32 array
        Scalar quantity per particle.

    Returns
    -------
    CG_Field : (N_points,) float64 array
        Volume-weighted scalar average per grid point.
    """
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros(Ngridpoints)

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        numerator = 0.0
        denominator = 0.0

        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            numerator += w * d
            denominator += w

        if denominator != 0.0:
            CG_Field[g] = numerator / denominator
        else:
            CG_Field[g] = 0.0

    return CG_Field