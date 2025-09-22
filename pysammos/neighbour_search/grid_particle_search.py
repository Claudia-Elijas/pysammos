"""
Module for efficiently matching particles to grid points within a cutoff radius and calculating displacement vectors and distances between them.

This module contains two main functions:

    - :func:`particle_node_match`: Uses kd-tree spatial queries to find particles within a cutoff radius of each grid point, returning
    start indices and a flattened array of matching particle indices.
    
    - :func:`calc_displacement`: Computes displacement vectors and distances between each grid point and its neighboring particles,
    using the output of `particle_node_match`.

These functions facilitate coarse-graining operations by quickly associating particles to grid points
and calculating relative positional data in an optimized manner.

**Terminology:**
    - :math:`N_{points}`: Number of coarse-graining grid points.
    - :math:`N_{particles}`: Number of particles in the system.
    - :math:`N_{total\_neighbors}`: Total number of particle-grid point neighbor pairs found within the cutoff distance.
    - :math:`c`: Cutoff distance for neighbor searching.

"""


# import relevant libraries
import numpy as np
from numba.types import Tuple as NumbaTuple 
from typing import Tuple as TypingTuple
from numba import njit, prange, float64, float32, int32
from scipy.spatial import cKDTree
from itertools import accumulate
# =======================================================================

def particle_node_match(GridPoints:np.ndarray, Particle_Position:np.ndarray, c:float)-> TypingTuple[np.ndarray, np.ndarray]:
    """
    Find particles within a cutoff distance of each grid point using kd-tree queries.

    Inputs
    ------
    GridPoints : np.ndarray, shape(N_points, 3) 
        Coordinates of the coarse-graining grid points in 3D space.
    Particle_Position : np.ndarray, (N_particles, 3) 
        Coordinates of particles in 3D space.
    c : float
        Cutoff distance within which particles are considered neighbors to grid points.

    Outputs
    -------
    start_indices : np.ndarray, shape(N_points + 2,) 
        Array of start indices into the flattened particle indices array for each grid point.
        Includes padding: start_indices[0] = 0 and start_indices[-1] = total number of matched particles.
    particle_indices_flat : np.ndarray, shape(N_total_neighbors,)
        Flattened array of particle indices that lie within cutoff distance of each grid point.
        The neighbors of grid point i are located in particle_indices_flat[start_indices[i]:start_indices[i+1]].
    """
    tree = cKDTree(GridPoints)
    particle_indices = tree.query_ball_tree(cKDTree(Particle_Position), c)

    # Check if there are any non-empty lists
    if any(len(indices) > 0 for indices in particle_indices):
        particle_indices_flat = np.concatenate([indices for indices in particle_indices if len(indices) > 0])
    else:
        particle_indices_flat = np.array([], dtype=int)

    # Compute start indices
    lengths = [len(indices) for indices in particle_indices]
    len_indices = GridPoints.shape[0] + 2
    start_indices = np.zeros(len_indices, dtype=np.int64) 
    start_indices[1:-1] = np.array(list(accumulate(lengths)), dtype=np.int64) # Accumulate lengths to get start indices
    start_indices[0] = 0  # Start index for the first grid point
    start_indices[-1] = len(particle_indices_flat)  # Last index must be total number of particles
    
    return start_indices.astype(np.int32), particle_indices_flat.astype(np.int32)

@njit(NumbaTuple((float64[:,:], float64[:]))(float64[:,:], float32[:,:], int32[:], int32[:]),parallel=True) 
def calc_displacement(GridPoints, Particle_Position, start_indices, visibility):
    """
    Calculate displacement vectors and distances between grid points and visible particles.

    Inputs
    ------
    GridPoints : np.ndarray, shape(N_points, 3) 
        Coordinates of coarse-graining grid points.
    Particle_Position : np.ndarray, shape(N_particles, 3)
        Coordinates of particles.
    start_indices : np.ndarray, shape(N_points + 2,) 
        Start indices into the visibility array for each grid point.
        Assumes padding: start_indices[0] = 0, start_indices[-1] = len(visibility).
    visibility : np.ndarray, shape(N_total_neighbors,)
        Flattened array of particle indices visible to each grid point.

    Outputs
    -------
    displacement_vectors : np.ndarray, shape(N_total_neighbors, 3) 
        Displacement vectors from each particle to the corresponding grid point.
    distances : np.ndarray, shape(N_total_neighbors,)
        Euclidean distances corresponding to displacement vectors.
    """

    # Preallocate arrays for results 
    total_particles = len(visibility)
    displacement_vectors = np.empty((total_particles, 3), dtype=np.float64)
    distances = np.empty(total_particles, dtype=np.float64)
    NPoints = GridPoints.shape[0]  # Number of grid points
    
    for i in prange(NPoints): # Iterate over grid points in parallel
        
        # Get the start and end indices for this grid point
        start = start_indices[i] ; end = start_indices[i+1]
        idx = visibility[start:end]
        
        # Calculate displacement vectors
        r_ri = GridPoints[i] - Particle_Position[idx]
        
        # Calculate squared distances manually (to replace np.einsum)
        r_ri_dist_squared = r_ri[:, 0]**2 + r_ri[:, 1]**2 + r_ri[:, 2]**2
        r_ri_dist = np.sqrt(r_ri_dist_squared)
        
        # Store results in preallocated arrays
        displacement_vectors[start:end] = r_ri
        distances[start:end] = r_ri_dist
    
    return displacement_vectors, distances
    