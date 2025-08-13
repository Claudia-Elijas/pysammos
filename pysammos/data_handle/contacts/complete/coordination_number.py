"""
Coordination Number Calculation
===============================

This module provides functionality to calculate the coordination number of particles
based on their contacts. It counts the number of contacts for each particle and
excludes isolated particles (rattlers) that have fewer than two contacts.

It uses Numba for efficient computation, especially for large datasets.

Functions
--------
- `count`: Counts the number of contacts per particle and excludes isolated ones.   
Returns both the full list and the filtered one (excluding particles with fewer than two contacts). 
"""

# import necessary libraries
import numpy as np
from numba import njit, int64
from numba.types import Tuple as NumbaTuple 
from typing import Tuple as TypingTuple

# Calculate coordination number
@njit(NumbaTuple((int64[:], int64[:]))(int64[:],int64[:]))
def count(particle_inContacts_dup:np.ndarray, global_id_all:np.ndarray)-> TypingTuple[np.ndarray, np.ndarray]:
    r"""
    
    Count the number of contacts per particle and exclude isolated ones.

    Counts the number of times each particle ID appears in the contact list,
    returning both the full list and the filtered one (excluding particles with
    fewer than two contacts).

    Parameters
    ----------
    particle_inContacts_dup : ndarray, shape (N,).
        Array of particle IDs involved in contacts (may contain duplicates).

    global_id_all : ndarray, shape (M,).
        List of all particle IDs for which coordination numbers are computed.

    Returns
    -------
    CN : ndarray, shape (M,).
        Coordination numbers (number of contacts) for each particle in `global_id_all`.

    CN_no_rattlers : ndarray, shape (K,).
        Coordination numbers excluding "rattlers" (particles with <= 1 contact).
    """
    max_id = int(np.max(particle_inContacts_dup))
    count_array = np.zeros(max_id + 1, dtype=np.int64)
    bincounts = np.bincount(particle_inContacts_dup)
    count_array[:len(bincounts)] = bincounts

    CN = np.zeros(len(global_id_all), dtype=np.int64)
    for i in range(len(global_id_all)):
        pid = global_id_all[i]
        if pid <= max_id:
            CN[i] = count_array[pid]

    CN_no_rattlers = CN[CN > 1]
    return CN, CN_no_rattlers
