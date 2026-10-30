r"""
Module for efficiently matching particles to grid points within a cutoff radius and calculating displacement vectors and distances between them.

This module contains three main functions:

    - :func:`particle_node_match`: Uses kd-tree spatial queries to find particles within a cutoff radius of each grid point, returning
      start indices and a flattened array of matching particle indices.

    - :func:`calc_displacement`: Computes displacement vectors and distances between each grid point and its neighboring particles,
      using the output of :func:`particle_node_match`.

    - :func:`branch_node_match`: Finds the contact halves whose branch, from the particle centre to the contact point,
      comes within the cutoff radius of each grid point, with their displacement vectors.

These functions facilitate coarse-graining operations by quickly associating particles to grid points
and calculating relative positional data in an optimized manner.

Along periodic axes, these functions measure distances to the nearest periodic image of each particle,
so that grid points near a periodic boundary also find the particles across it.

**Terminology:**
    - :math:`N_{points}`: Number of coarse-graining grid points.
    - :math:`N_{particles}`: Number of particles in the system.
    - :math:`N_{total\_neighbors}`: Total number of particle-grid point neighbor pairs found within the cutoff distance.
    - :math:`c`: Cutoff distance for neighbor searching.

"""


# import relevant libraries
import numpy as np
from numba.types import Tuple as NumbaTuple 
from typing import Optional, Tuple as TypingTuple
from numba import njit, prange, float64, float32, int32
from scipy.spatial import cKDTree
from itertools import accumulate
# =======================================================================

def wrap_into_box(Points:np.ndarray, box_lengths:np.ndarray)-> np.ndarray:
    """
    Copy of the points with their coordinates along the periodic axes brought into [0, L).

    Inputs
    ------
    Points : np.ndarray, shape(N, 3)
        Coordinates of the points.
    box_lengths : np.ndarray, shape(3,)
        Length L of the simulation box along each periodic axis, 0 along the non-periodic axes.

    Outputs
    -------
    wrapped : np.ndarray, shape(N, 3)
        Coordinates (float64) folded into the box along the periodic axes, unchanged along the others.
    """
    wrapped = np.array(Points, dtype=np.float64)
    for k in np.flatnonzero(box_lengths > 0):
        wrapped[:, k] = np.mod(wrapped[:, k], box_lengths[k])
        wrapped[wrapped[:, k] >= box_lengths[k], k] = 0.0 # np.mod rounds tiny negative values up to L
    return wrapped

def particle_node_match(GridPoints:np.ndarray, Particle_Position:np.ndarray, c:float,
                        box_lengths:Optional[np.ndarray]=None)-> TypingTuple[np.ndarray, np.ndarray]:
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
    box_lengths : np.ndarray or None, shape(3,), optional
        Length of the simulation box along each periodic axis, 0 along the non-periodic axes.
        Along a periodic axis the distance is measured to the nearest periodic image of the particle,
        which requires c to be less than half the box length. Default is None (no periodic axis).

    Outputs
    -------
    start_indices : np.ndarray, shape(N_points + 2,) 
        Array of start indices into the flattened particle indices array for each grid point.
        Includes padding: start_indices[0] = 0 and start_indices[-1] = total number of matched particles.
    particle_indices_flat : np.ndarray, shape(N_total_neighbors,)
        Flattened array of particle indices that lie within cutoff distance of each grid point.
        The neighbors of grid point i are located in particle_indices_flat[start_indices[i]:start_indices[i+1]].
    """
    if box_lengths is None or not np.any(np.asarray(box_lengths) > 0):
        tree = cKDTree(GridPoints)
        particle_indices = tree.query_ball_tree(cKDTree(Particle_Position), c)
    else:
        # periodic kd-trees: both point sets folded into the box, distances to the nearest image
        box = np.asarray(box_lengths, dtype=np.float64)
        if np.any(2 * c >= box[box > 0]):
            raise ValueError(f"The search radius {c:.6g} must be less than half the box length along every periodic axis "
                             f"(box lengths {box}); otherwise a grid point would reach some particles through two periodic images.")
        tree = cKDTree(wrap_into_box(GridPoints, box), boxsize=box)
        particle_indices = tree.query_ball_tree(cKDTree(wrap_into_box(Particle_Position, box), boxsize=box), c)

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

def calc_displacement(GridPoints:np.ndarray, Particle_Position:np.ndarray, start_indices:np.ndarray, visibility:np.ndarray,
                      box_lengths:Optional[np.ndarray]=None)-> TypingTuple[np.ndarray, np.ndarray]:
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
    box_lengths : np.ndarray or None, shape(3,), optional
        Length of the simulation box along each periodic axis, 0 along the non-periodic axes.
        Along a periodic axis the displacement is taken to the nearest periodic image of the particle.
        Default is None (no periodic axis).

    Outputs
    -------
    displacement_vectors : np.ndarray, shape(N_total_neighbors, 3) 
        Displacement vectors from each particle to the corresponding grid point.
    distances : np.ndarray, shape(N_total_neighbors,)
        Euclidean distances corresponding to displacement vectors.
    """
    box = np.zeros(3) if box_lengths is None else np.asarray(box_lengths, dtype=np.float64)
    return _calc_displacement(GridPoints, Particle_Position, start_indices, visibility, box)

@njit(NumbaTuple((float64[:,:], float64[:]))(float64[:,:], float32[:,:], int32[:], int32[:], float64[:]),parallel=True)
def _calc_displacement(GridPoints, Particle_Position, start_indices, visibility, box_lengths):
    """
    Numba kernel of :func:`calc_displacement`, with box_lengths given (0 along the non-periodic axes).
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
        
        # Nearest periodic image along the periodic axes
        for k in range(3):
            if box_lengths[k] > 0.0:
                r_ri[:, k] -= box_lengths[k] * np.rint(r_ri[:, k] / box_lengths[k])

        # Calculate squared distances manually (to replace np.einsum)
        r_ri_dist_squared = r_ri[:, 0]**2 + r_ri[:, 1]**2 + r_ri[:, 2]**2
        r_ri_dist = np.sqrt(r_ri_dist_squared)
        
        # Store results in preallocated arrays
        displacement_vectors[start:end] = r_ri
        distances[start:end] = r_ri_dist
    
    return displacement_vectors, distances
    
def branch_node_match(GridPoints:np.ndarray, Particle_Position:np.ndarray, BranchVector:np.ndarray, c:float,
                      box_lengths:Optional[np.ndarray]=None,
                      chunk_size:int=20000)-> TypingTuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Find the contact halves whose branch comes within a cutoff distance of each grid point.

    A contact half contributes to the contact stress at a grid point along the part of its branch,
    from the particle centre to the contact point, that lies within c of the grid point. That part
    exists for some halves whose particle centre is farther than c. The candidates of each half are the
    grid points within c plus half its branch length of the middle of its branch (the smallest such ball
    holding every point within c of the branch), and the grid points farther than c from the branch are
    then dropped. The halves are searched in chunks, so that only the candidate pairs of one chunk are
    held at a time. For each grid point the halves come in increasing index order, as in
    :func:`particle_node_match`, so a grid point that no branch reaches from beyond c gets the same
    halves, in the same order and with the same displacements, as a search of the particle centres within c.

    Inputs
    ------
    GridPoints : np.ndarray, shape(N_points, 3)
        Coordinates of the coarse-graining grid points.
    Particle_Position : np.ndarray, shape(N_halves, 3)
        Centre of the particle of each contact half.
    BranchVector : np.ndarray, shape(N_halves, 3)
        Branch vector of each contact half, from the contact point to the particle centre.
    c : float
        Cutoff distance of the kernel.
    box_lengths : np.ndarray or None, shape(3,), optional
        Length of the simulation box along each periodic axis, 0 along the non-periodic axes.
        Default is None (no periodic axis).
    chunk_size : int, optional
        Number of halves searched at once. Default is 20000.

    Outputs
    -------
    start_indices : np.ndarray, shape(N_points + 2,)
        Start indices into half_indices for each grid point, as returned by :func:`particle_node_match`.
    half_indices : np.ndarray, shape(N_total_neighbors,)
        Indices of the contact halves kept for each grid point.
    displacement_vectors : np.ndarray, shape(N_total_neighbors, 3)
        Displacement vectors from the particle centre of each half to the grid point.
    """
    Centre = np.asarray(Particle_Position)
    Branch = np.asarray(BranchVector, dtype=np.float64)
    branch_length = np.sqrt(Branch[:, 0]**2 + Branch[:, 1]**2 + Branch[:, 2]**2)
    box = np.zeros(3) if box_lengths is None else np.asarray(box_lengths, dtype=np.float64)
    periodic = np.flatnonzero(box > 0)
    n_points, n_halves = GridPoints.shape[0], len(Branch)

    # candidate grid points of each half: within c + |B|/2 of the middle of its branch
    middle = Centre.astype(np.float64) - 0.5 * Branch
    radius = (c + 0.5 * branch_length) * (1 + 1e-12) # margin for rounding, the exact test follows
    if len(periodic):
        if n_halves and np.any(2 * (c + branch_length.max()) >= box[periodic]):
            raise ValueError(f"The cutoff ({c:.6g}) plus the longest branch ({branch_length.max():.6g}) must be less than half "
                             f"the box length along every periodic axis (box lengths {box}).")
        boxsize = box
        tree = cKDTree(wrap_into_box(GridPoints, box), boxsize=boxsize)
        middle = wrap_into_box(middle, box)
    else:
        boxsize = None
        tree = cKDTree(GridPoints)

    nodes, halves, displacements = [], [], []
    for start in range(0, n_halves, chunk_size):
        stop = min(start + chunk_size, n_halves)
        pairs = tree.sparse_distance_matrix(cKDTree(middle[start:stop], boxsize=boxsize), radius[start:stop].max(),
                                            output_type="ndarray")
        node = pairs["i"].astype(np.int32)
        half = pairs["j"].astype(np.int32) + np.int32(start)
        near = pairs["v"] <= radius[half]
        node, half = node[near], half[near]
        # displacement from the particle centre to the grid point, to the nearest periodic image (as calc_displacement)
        r = GridPoints[node] - Centre[half]
        for k in periodic:
            r[:, k] -= box[k] * np.rint(r[:, k] / box[k])
        # closest point of the branch to the grid point: r + s * B with 0 <= s <= 1
        B = Branch[half]
        BB = B[:, 0]**2 + B[:, 1]**2 + B[:, 2]**2
        s = np.clip(-(r[:, 0]*B[:, 0] + r[:, 1]*B[:, 1] + r[:, 2]*B[:, 2]) / np.where(BB > 0, BB, 1.0), 0.0, 1.0)
        closest = r + s[:, np.newaxis] * B
        keep = ((r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2 <= c*c) # centre within c
                | (closest[:, 0]**2 + closest[:, 1]**2 + closest[:, 2]**2 <= c*c)) # or part of the branch
        nodes.append(node[keep])
        halves.append(half[keep])
        displacements.append(r[keep])

    node = np.concatenate(nodes) if nodes else np.zeros(0, dtype=np.int32)
    half = np.concatenate(halves) if halves else np.zeros(0, dtype=np.int32)
    r = np.concatenate(displacements) if displacements else np.zeros((0, 3))
    del nodes, halves, displacements # the chunks, now copied
    order = np.lexsort((half, node)) # grouped by grid point, halves in increasing order
    return _start_indices(node[order], n_points), half[order], r[order]

def _start_indices(nodes:np.ndarray, n_points:int)-> np.ndarray:
    """
    Start indices, padded as those of :func:`particle_node_match`, of pairs sorted by grid point index.
    """
    start_indices = np.zeros(n_points + 2, dtype=np.int64)
    start_indices[1:-1] = np.cumsum(np.bincount(nodes, minlength=n_points))
    start_indices[-1] = start_indices[-2]
    return start_indices.astype(np.int32)
    