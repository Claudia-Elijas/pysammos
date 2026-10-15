"""
This module provides functionality to identify and remove duplicate particle pairs
from contact data. It ensures that each contact between two particles is represented only once,
regardless of their order (i.e., (A, B) is considered the same as (B, A)).
It is particularly useful in DEM simulations computed in parallel, where duplicate pairs
may arise due to the use of ghost particles or other parallelization artifacts.
When contact points are available, a pair that touches at several points (clumps,
concave particles) keeps one row per contact.

This module contains two main functions:
    1. :func:`get_unique_pairs`: Identifies and filters duplicate particle pairs from two arrays.
    2. :func:`delete`: Removes duplicate particle pairs and their associated data from the original contact list.
"""


# import necessary libraries
import numpy as np
from typing import Tuple, Optional

# Get unique pairs of two arrays 
def get_unique_pairs(LL_py:np.ndarray, I_py:np.ndarray, Contact_points:Optional[np.ndarray]=None,
                     tolerance:float=0.0, box_lengths:Optional[np.ndarray]=None,
                     periodic:Optional[np.ndarray]=None, Forces:Optional[np.ndarray]=None,
                     tolerance_with_forces:float=0.0, force_tolerance:float=1e-4,
                     box_known:Optional[np.ndarray]=None)-> np.ndarray:
    r"""
    
    Identify and filter duplicate particle pairs.

    Given two arrays representing particle interactions, this function identifies
    all unique unordered pairs (i.e., (A, B) and (B, A) are treated as the same)
    and returns the indices of the rows to keep. Without contact points, only the
    first occurrence of each pair is kept. With contact points, a repeated row is a
    duplicate only if it is the same contact as a row already kept for the pair: its
    contact point lies within `tolerance` of that row's or, when `Forces` are given,
    within `tolerance_with_forces` with the same or the opposite force (the same contact
    written from either particle, each copy placed next to its own particle). A pair
    with several contacts keeps one row per contact.

    Inputs
    ------
    LL_py : ndarray, shape (N,).
        First particle ID array.

    I_py : ndarray, shape (N,).
        Particle that particle LL is interacting with.

    Contact_points : ndarray or None, optional, shape (N, 3).
        Contact point of each row. If `None`, every repeated pair is a duplicate.

    tolerance : float, optional.
        Largest distance between the contact points of two rows of the same contact.

    box_lengths : ndarray or None, optional, shape (3,).
        Size of the domain along each axis, used with `periodic`.

    periodic : ndarray of bool or None, optional, shape (3,).
        Periodic axes. Along those whose box length is known (`box_known`), contact points
        are compared through the nearest periodic image. Along the others the length may be
        an estimate; the contacts of one pair lie within a particle size of each other, so
        two contact points more than half the box apart are taken as periodic images of one
        contact, and that component is not compared.

    Forces : ndarray or None, optional, shape (N, 3).
        Contact force of each row, compared for the rows whose contact points lie farther
        apart than `tolerance` but within `tolerance_with_forces`.

    tolerance_with_forces : float, optional.
        Largest distance between the contact points of two rows of the same contact that
        have the same or opposite forces.

    force_tolerance : float, optional.
        Largest difference between the force of one row and the force of another, or its
        opposite, relative to the larger of the two, for rows of the same contact.

    box_known : ndarray of bool or None, optional, shape (3,).
        Periodic axes whose box length is known exactly. Default is None (none).

    Outputs
    -------
    keep_indices : ndarray, shape (M,).
        Indices of the rows to keep, in their original order.

    Examples
    --------
    >>> LL_py = np.array([1, 2, 1, 4, 1, 3, 7])  # Particle A IDs with duplicates
    >>> I_py =  np.array([2, 1, 2, 3, 3, 1, 4])  # Particle B IDs with duplicates
    >>> keep_indices = get_unique_pairs(LL_py, I_py)
    >>> print(keep_indices)
    [0 3 4 6]
    This indicates that the unique pairs are found at indices 0, 3, 4, and 6 of the original arrays.

    >>> points = np.array([[0., 0., 0.], [1., 0., 0.], [1e-3, 0., 0.]])  # contact points
    >>> forces = np.array([[1., 0., 0.], [2., 0., 0.], [-1., 0., 0.]])  # force on the first particle of each row
    >>> print(get_unique_pairs(np.array([1, 1, 2]), np.array([2, 2, 1]), points, tolerance=1e-6,
    ...                        Forces=forces, tolerance_with_forces=1e-2))
    [0 1]

    Rows 0 and 1 are two contacts of the pair (1, 2) at different points, and both are kept.
    Row 2 is the reversed copy of row 0, 1e-3 away with the opposite force, and is removed.

    """
    keep_indices, _ = _unique_rows(LL_py, I_py, Contact_points, tolerance, box_lengths, periodic, Forces,
                                   tolerance_with_forces, force_tolerance, box_known)
    return keep_indices


def _unique_rows(LL_py:np.ndarray, I_py:np.ndarray, Contact_points:Optional[np.ndarray], tolerance:float,
                 box_lengths:Optional[np.ndarray], periodic:Optional[np.ndarray], Forces:Optional[np.ndarray],
                 tolerance_with_forces:float, force_tolerance:float,
                 box_known:Optional[np.ndarray])-> Tuple[np.ndarray, int]:
    """Indices of the rows to keep, see `get_unique_pairs()`, and the number of distinct pairs."""
    LL_py, I_py = np.asarray(LL_py), np.asarray(I_py)
    n = len(LL_py)
    low, high = np.minimum(LL_py, I_py), np.maximum(LL_py, I_py) # (2,6) and (6,2) give the same pair
    order = np.lexsort((np.arange(n), high, low)) # rows of one pair become neighbours, in their original order
    low, high = low[order], high[order]
    first = np.ones(n, dtype=bool) # first row of each pair (in sorted order)
    first[1:] = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[order[first]] = True

    if Contact_points is not None and not first.all():
        rule = (Contact_points, Forces, tolerance, tolerance_with_forces, force_tolerance, box_lengths, periodic, box_known)
        starts = np.flatnonzero(first)
        sizes = np.diff(np.append(starts, n))
        # pairs found twice (the usual case): compare the second row with the first
        two = starts[sizes == 2]
        differ = ~_same_contact(order[two + 1], order[two], *rule)
        keep_mask[order[two + 1][differ]] = True
        # pairs found three times or more: compare each row with the rows kept so far
        for s, size in zip(starts[sizes > 2], sizes[sizes > 2]):
            kept = [order[s]]
            for idx in order[s + 1:s + size]:
                if not _same_contact(np.full(len(kept), idx), np.array(kept), *rule).any():
                    kept.append(idx)
            keep_mask[kept] = True

    return np.flatnonzero(keep_mask), int(np.count_nonzero(first))


def _same_contact(a:np.ndarray, b:np.ndarray, Contact_points:np.ndarray, Forces:Optional[np.ndarray],
                  tolerance:float, tolerance_with_forces:float, force_tolerance:float,
                  box_lengths:Optional[np.ndarray], periodic:Optional[np.ndarray],
                  box_known:Optional[np.ndarray])-> np.ndarray:
    """True where rows a[k] and b[k] hold the same contact, see `get_unique_pairs()`."""
    distance = _point_distance(Contact_points[a], Contact_points[b], box_lengths, periodic, box_known)
    same = distance <= tolerance
    if Forces is not None and tolerance_with_forces > tolerance:
        fa = np.asarray(Forces[a], dtype=np.float64)
        fb = np.asarray(Forces[b], dtype=np.float64)
        difference = np.minimum(np.linalg.norm(fa - fb, axis=-1), np.linalg.norm(fa + fb, axis=-1))
        scale = np.maximum(np.linalg.norm(fa, axis=-1), np.linalg.norm(fb, axis=-1))
        same |= (distance <= tolerance_with_forces) & (difference <= force_tolerance * scale)
    return same


def _point_distance(p:np.ndarray, q:np.ndarray, box_lengths:Optional[np.ndarray], periodic:Optional[np.ndarray],
                    box_known:Optional[np.ndarray])-> np.ndarray:
    """Distance between contact points p and q (shape (..., 3)), see `get_unique_pairs()`."""
    d = np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)
    if box_lengths is not None and periodic is not None:
        L = np.asarray(box_lengths, dtype=np.float64)
        along = np.asarray(periodic, dtype=bool) & (L > 0)
        known = along & (np.zeros(3, dtype=bool) if box_known is None else np.asarray(box_known, dtype=bool))
        L_known = np.where(known, L, 1.0)
        d = np.where(known, d - L_known * np.rint(d / L_known), d) # nearest periodic image
        d = np.where(along & ~known & (np.abs(d) > 0.5 * L), 0.0, d) # component of a periodic image, not compared
    return np.sqrt(np.sum(d * d, axis=-1))


# Check if there is duplicate pairs
def delete(Particle_LL_OG:np.ndarray, Particle_I_OG:np.ndarray, 
           Fij_OG:np.ndarray, Contacts_OG:Optional[np.ndarray],
           tolerance:float=0.0, box_lengths:Optional[np.ndarray]=None,
           periodic:Optional[np.ndarray]=None, tolerance_with_forces:float=0.0, force_tolerance:float=1e-4,
           box_known:Optional[np.ndarray]=None)->Tuple[np.ndarray,np.ndarray,Optional[np.ndarray]]:
    r"""
    Remove duplicate particle interaction pairs and associated data.

    Uses `get_unique_pairs()` to remove repeated rows of a particle pair (treating
    (A, B) and (B, A) as the same pair). Without contact points only the first
    occurrence of each pair is kept; with contact points, a repeated row is removed
    only if it is the same contact as a row already kept for the pair: the same contact
    point, within `tolerance`, or within `tolerance_with_forces` with the same or the
    opposite force (the forces are Fij_OG).

    Inputs
    ------
    Particle_LL_OG : ndarray, shape (N,).
        Particle A IDs for original contact list.

    Particle_I_OG : ndarray, shape (N,).
        Particle B IDs for original contact list.

    Fij_OG : ndarray, shape (N, 3).
        Contact force vectors corresponding to particle pairs.

    Contacts_OG : ndarray or None, optional, shape (N, 3).
        Contact point coordinates. If `None`, no contact points are returned.

    tolerance, box_lengths, periodic, tolerance_with_forces, force_tolerance, box_known : optional.
        How the rows of a pair are compared, see `get_unique_pairs()`.

    Outputs
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
    keep_these, n_pairs = _unique_rows(Particle_LL_OG, Particle_I_OG, Contacts_OG, tolerance, box_lengths, periodic,
                                       Fij_OG, tolerance_with_forces, force_tolerance, box_known)
    #Select the Contact data corresponding to Unique Pairs
    Particle_LL = Particle_LL_OG[keep_these]
    Particle_I = Particle_I_OG[keep_these]
    Fij = Fij_OG[keep_these]
    Contacts = Contacts_OG[keep_these] if Contacts_OG is not None else None
    print('  Repeated pairs in contact data: ', len(Particle_LL_OG) - len(Particle_LL))
    if Contacts_OG is not None:
        print('  Repeated pairs kept as separate contacts (different contact points or forces): ', len(Particle_LL) - n_pairs)
    
    return Particle_LL, Particle_I, Fij, Contacts
