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

    

# ========================================================================
# # new but not accounting for empty grid points
# def Particle_Node_Correspondance___forfullpoints(GridPoints, Particle_Position, c):
#     """
#     This function uses a kd-tree to find the particles within a cutoff distance of each grid point.
#     """
    
#     # Create a kd-tree from the grid points
#     tree = cKDTree(GridPoints)
#     # Query the kd-tree for particles within the cutoff distance
#     particle_indices = tree.query_ball_tree(cKDTree(Particle_Position), c)
#     start_indices = np.array(list(accumulate(len(indices) for indices in particle_indices)))
#     particle_indices_flat = np.concatenate(particle_indices)
  
#     return start_indices, particle_indices_flat
# # old approach
# def Particle_Node_Assignation__(GridPoints, spacing, c, Particle_Position, r_ri_return = False, r_ri_dist_return = True):
    
    # # Calculate the minimum grid point coordinates
    # xyz_min_gridpoints = np.min(GridPoints, axis=0)

    # # Determine the dimensionality of the grid
    # dim = np.sum(np.ptp(GridPoints, axis=0) > 0)  # Count non-constant dimensions

    # # Find node indices for each particle and grid point
    # if dim == 1:  # Line (1D)
    #     grid_nodes = np.round((GridPoints[:, 0:1] - xyz_min_gridpoints[0]) / spacing).astype(int)  # Only use the x-axis
    #     particle_nodes = np.round((Particle_Position[:, 0:1] - xyz_min_gridpoints[0]) / spacing).astype(int)
    # elif dim == 2:  # Plane (2D)
    #     grid_nodes = np.round((GridPoints[:, 0:2] - xyz_min_gridpoints[0:2]) / spacing).astype(int)  # Use x and y axes
    #     particle_nodes = np.round((Particle_Position[:, 0:2] - xyz_min_gridpoints[0:2]) / spacing).astype(int)
    # else:  # Full 3D
    #     grid_nodes = np.round((GridPoints - xyz_min_gridpoints) / spacing).astype(int)
    #     particle_nodes = np.round((Particle_Position - xyz_min_gridpoints) / spacing).astype(int)

    # # Find the particles at a distance 2 x grid spacing to each grid point
    # node_min, node_max = (grid_nodes - 2), (grid_nodes + 2)  # Precompute bounds to avoid redundant computation
    # maskXYZ = np.all((particle_nodes >= node_min[:, None, :]) & (particle_nodes <= node_max[:, None, :]), axis=2)  # Mask nodes

    # # Find correction for distances above cutoff
    # masked_grid_indices, masked_particle_indices = np.nonzero(maskXYZ)  # Get indices where the condition holds
    # masked_positions, masked_grid_points = Particle_Position[masked_particle_indices], GridPoints[masked_grid_indices]  # Select positions and grid points
    # r_ri = masked_grid_points - masked_positions # Distance vector
    # r_ri_dist = np.sqrt(np.einsum('ij,ij->i', r_ri, r_ri))  # Distance norm
    # cutoff = r_ri_dist <= c  # Find indices for distances below cutoff

    # # Group the particles into arrays for each grid point index
    # unique_grid_indices = np.append(np.unique(masked_grid_indices[cutoff], return_index=True)[1][1:], len(masked_grid_indices[cutoff]))

    # # Return the result
    # if r_ri_return  == True and r_ri_dist_return  == False:
    #     return unique_grid_indices, masked_particle_indices[cutoff], r_ri[cutoff]
    # elif r_ri_return  == False and r_ri_dist_return  == True:
    #     return unique_grid_indices, masked_particle_indices[cutoff], r_ri_dist[cutoff]
    # elif r_ri_return  == True and r_ri_dist_return  == True:
    #     return unique_grid_indices, masked_particle_indices[cutoff], r_ri[cutoff], r_ri_dist[cutoff]
    # elif r_ri_return  == False and r_ri_dist_return  == False:
    #     return unique_grid_indices, masked_particle_indices[cutoff]
    # else:
    #     raise ValueError("Please select either r_ri or r_ri_dist.")