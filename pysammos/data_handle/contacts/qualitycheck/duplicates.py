"""
Duplicate Particle Pairs Handling
========================
This module provides functionality to identify and remove duplicate particle pairs
from contact data. It ensures that each unique pair of particles is represented only once,
regardless of their order (i.e., (A, B) is considered the same as (B, A)).
It is particularly useful in DEM simulations computed in parallel, where duplicate pairs
may arise due to the use of ghost particles or other parallelization artifacts.

This module contains two main functions:
1. `get_unique_pairs`: Identifies and filters duplicate particle pairs from two arrays.
2. `delete`: Removes duplicate particle pairs and their associated data from the original contact list.

"""


# import necessary libraries
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional

# Get unique pairs of two arrays 
def get_unique_pairs(LL_py:np.ndarray, I_py:np.ndarray)-> np.ndarray:
    r"""
    
    Identify and filter duplicate particle pairs.

    Given two arrays representing particle interactions, this function identifies
    all unique unordered pairs (i.e., (A, B) and (B, A) are treated as the same)
    and returns the indices of the first occurrence of each pair.

    Parameters
    ----------
    LL_py : ndarray, shape (N,).
        First particle ID array.

    I_py : ndarray, shape (N,).
        Particle that particle LL is interacting with.

    Returns
    -------
    keep_indices : ndarray, shape (M,).
        Indices of the first occurrence of each unique pair.

    """
    pair_indices = defaultdict(list) # Store indices of each pair, treating (A,B) and (B,A) as the same
    for idx, (ll, i) in enumerate(zip(LL_py, I_py)):
        sorted_pair = tuple(sorted((ll, i))) # Sorting ensures (2,6) == (6,2)
        pair_indices[sorted_pair].append(idx)
    keep_mask = np.ones(len(LL_py), dtype=bool) # Create a boolean mask to track which elements to keep
    for pair, indices in pair_indices.items(): 
        if len(indices) > 1: # If duplicates exist
            #print(f"First Pair: {LL_py[indices[0]], I_py[indices[0]]}. Second Pair: {LL_py[indices[1]], I_py[indices[1]]}")
            for idx in indices[1:]:# Mark duplicate occurrences for removal (except the first one)
                keep_mask[idx] = False
    keep_indices = np.where(keep_mask)[0] ; #print('Keep indices: ', len(keep_indices))
    #remove_indices = np.where(~keep_mask)[0] ; #print('Remove indices: ', len(remove_indices))  
    return keep_indices


# Check if there is duplicate pairs
def delete(Particle_LL_OG:np.ndarray, Particle_I_OG:np.ndarray, 
           Fij_OG:np.ndarray, Contacts_OG:Optional[np.ndarray])->Tuple[np.ndarray,np.ndarray,Optional[np.ndarray]]:
    r"""
    Remove duplicate particle interaction pairs and associated data.

    Uses `get_unique_pairs()` to retain only the first occurrence of each
    particle pair (treating (A, B) and (B, A) as duplicates).

    Parameters
    ----------
    Particle_LL_OG : ndarray, shape (N,).
        Particle A IDs for original contact list.

    Particle_I_OG : ndarray, shape (N,).
        Particle B IDs for original contact list.

    Fij_OG : ndarray, shape (N, 3).
        Contact force vectors corresponding to particle pairs.

    Contacts_OG : ndarray or None, optional, shape (N, 3).
        Contact point coordinates. If `None`, no contact points are returned.


    Returns
    -------
    Particle_LL : ndarray, shape (M,).
        Filtered Particle A IDs.

    Particle_I : ndarray, shape (M,).
        Filtered Particle B IDs.

    Fij : ndarray, shape (M, 3).
        Filtered contact forces.

    Contacts : ndarray or None, shape (M, 3), optional.
        Filtered contact points if provided.
    """
    # Find Unique Pairs
    keep_these = get_unique_pairs(Particle_LL_OG, Particle_I_OG)
    #Select the Contact data corresponding to Unique Pairs
    Particle_LL = Particle_LL_OG[keep_these]
    Particle_I = Particle_I_OG[keep_these]
    Fij = Fij_OG[keep_these]
    Contacts = Contacts_OG[keep_these] if Contacts_OG is not None else None
    print('  Repeated pairs in contact data: ', len(Particle_LL_OG) - len(Particle_LL))
    
    return Particle_LL, Particle_I, Fij, Contacts
