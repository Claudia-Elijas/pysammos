import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
from itertools import accumulate
# =======================================================================

def Particle_Node_Correspondance(GridPoints, Particle_Position, c):
    """
    This function uses a kd-tree to find the particles within a cutoff distance of each grid point.
    Returns the start indices and a flattened array of particle indices.
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
    
    return start_indices, particle_indices_flat

@njit#(parallel=True)
def Calc_Displacement_and_Distance(GridPoints, Particle_Position, 
                                   start_indices, visibility,  
                                   return_disp, return_dist):
    """
    Optimized calculation of displacement vectors and distances between grid points and particles within a cutoff distance.
    Returns them as 1D arrays.
    """
    # Preallocate arrays for results 
    total_particles = len(visibility)
    displacement_vectors = np.empty((total_particles, 3), dtype=np.float64)
    distances = np.empty(total_particles, dtype=np.float64)
    NPoints = GridPoints.shape[0]  # Number of grid points
    
    for i in range(NPoints): # Iterate over grid points in parallel
        
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
    
    # Return the results
    if return_disp and return_dist:
        return displacement_vectors, distances
    elif return_disp:
        return displacement_vectors, None
    elif return_dist:
        return None, distances
    else:
        return None, None
