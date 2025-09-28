"""
This module provides functions to calculate branch vectors and center-to-center vectors
between particles in a simulation, considering periodic boundary conditions.
It includes two main functions:

    1. :func:`from_contacts`: Computes branch vectors from contact positions.

    2. :func:`from_diameters`: Computes branch vectors based on particle diameters.

These functions handle periodic boundary corrections to ensure accurate vector calculations
in simulations with periodic boundaries.
"""

import numpy as np
from typing import Tuple

# Calculate branch vectors
def from_contacts(r_A: np.ndarray, r_B: np.ndarray, contact_A: np.ndarray, 
                  L: np.ndarray, periodic: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    r"""

    Computes the branch vector :math:`\mathbf{B_v}` from the center of particle A
    to its contact point, as well as the center-to-center vector :math:`\mathbf{d}`
    between particles A and B. It uses the position of the contact between the two particles.
    The function handles periodic boundary corrections along each axis.

    Let:

    - :math:`\mathbf{r}_A`, :math:`\mathbf{r}_B` be the center positions of particles A and B.
    - :math:`\mathbf{c}_A` be the contact point on particle A.
    - :math:`\mathbf{L}` be the domain size.
    - :math:`\mathbf{p}` be the periodicity flags (True/False) per dimension.

    The branch vector is computed as:

    .. math::

        \mathbf{B_v} = \mathbf{r}_A - \mathbf{c}_A

    And the center-to-center vector is:

    .. math::

        \mathbf{d} = \mathbf{r}_A - \mathbf{r}_B

    Periodic corrections are applied:

    .. math::

        \mathbf{d}_i \leftarrow \mathbf{d}_i - L_i \cdot \mathrm{round}\left(\frac{\mathbf{d}_i}{L_i}\right)

    for each dimension :math:`i` where periodicity is enabled.
   
    
    Inputs
    -------
    r_A : ndarray, shape (N, 3)
        Center positions of particle A.
    r_B : ndarray, shape (N, 3)
        Center positions of particle B.
    contact_A : ndarray, shape (N, 3)
        Contact points on particle A.
    L : ndarray, shape (3,)
        Domain dimensions.
    periodic : ndarray, shape (3,)
        Boolean array indicating periodicity in each spatial dimension.

    Outputs
    -------
    BV : ndarray, shape (N, 3)
        Branch vectors from particle A center to contact point.
    d : ndarray, shape (N, 3)
        Center-to-center vectors between particles A and B.

    
    Examples
    --------

    Given two particles A and B with positions and a contact point on A, the branch vector
    and center-to-center vector can be computed as follows.

        >>> r_A = np.array([[1, 2.0]])
        >>> r_B = np.array([[2, 2.0]])
        >>> contact_A = np.array([[1.5, 2.0]])
        >>> L = np.array([3.0, 3.0])
        >>> periodic = np.array([True, True])
        >>> BV, d = from_contacts(r_A, r_B, contact_A, L, periodic)
        >>> print("Branch Vectors:\n", BV)
        Branch Vectors:
        [[-0.5  0. ]]
        >>> print("Center-to-Center Vectors:\n", d)
        Center-to-Center Vectors:
        [[-1.  0. ]]    

    """


    # Compute displacement vector from center to contact point
    BV = r_A - contact_A
    d = r_A - r_B  # Center-to-center vector
    
    # Adjust displacement for periodic boundaries
    for i in range(len(L)):  # Loop over each dimension
        if periodic[i]:  
            BV[:, i] = BV[:, i] - L[i] * np.round(BV[:, i] / L[i])  # Periodic correction
            d[:,i] = d[:,i] - L[i] * np.round(d[:,i] / L[i]) # correct for particles close to boundary
    
    return BV, d

# Calculate branch vectors from diameters
def from_diameters(r_A:np.ndarray, r_B:np.ndarray, d_A:np.ndarray, d_B:np.ndarray, 
                   L:np.ndarray, periodic:np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    r"""
    

    Computes the branch vector :math:`\mathbf{B_v}` from the center of particle A
    to its contact point, as well as the center-to-center vector :math:`\mathbf{d}`
    between particles A and B. It uses the position of the two particles and their diameters.
    The function handles periodic boundary corrections along each axis.

    Given:

    - :math:`\mathbf{r}_A, \mathbf{r}_B` — positions of particles A and B
    - :math:`d_A, d_B` — diameters of particles A and B
    - :math:`\mathbf{L}` — domain lengths
    - :math:`\mathbf{p}` — periodic boundary flags

    The center-to-center vector is:

    .. math::

        \mathbf{d} = \mathbf{r}_A - \mathbf{r}_B

    After applying periodic corrections:

    .. math::

        \mathbf{d}_i \leftarrow \mathbf{d}_i - L_i \cdot \mathrm{round}\left(\frac{\mathbf{d}_i}{L_i}\right)

    The branch vector from the center of particle A to the contact point is:

    .. math::

        \mathbf{BV} = \mathbf{d} \cdot \frac{d_A}{d_A + d_B}

    Inputs
    ------
    r_A : ndarray, shape (N, 3).
        Center positions of particle A.

    r_B : ndarray, shape (N, 3).
        Center positions of particle B.

    d_A : ndarray, shape (N,).
        Diameters of particle A.

    d_B : ndarray, shape (N,).
        Diameters of particle B.

    L : array_like, shape (3,).
        Domain dimensions.

    periodic : array_like, shape (3,).
        Periodicity flags (True/False) for each axis.

    Outputs
    -------
    BV : ndarray, shape (N, 3).
        Branch vectors from particle A to contact points.

    d : ndarray, shape (N, 3).
        Periodically corrected center-to-center vectors.
    """
    # Initialize displacement vector
    d = r_A - r_B
    
    # Adjust displacement for periodic boundaries selectively
    for i in range(len(L)):  # Loop over each dimension
        if periodic[i]:  # Apply periodic adjustment if enabled for the dimension
            d[:,i] = d[:,i] - L[i] * np.round(d[:,i] / L[i]) # correct for particles close to boundary
    BV =  d * d_A[:, np.newaxis] / (d_A[:, np.newaxis] + d_B[:, np.newaxis])
  
    return BV , d    