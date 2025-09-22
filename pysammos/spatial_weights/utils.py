"""
This module provides utility functions commonly used in numerical computations, particularly in physics and
engineering simulations involving integration along branches or paths, and geometric distance calculations.

This module includes the following functions:

    - :func:`first_significant_figure_position`: Computes the positional value of the first significant figure in a given number.
    - :func:`integration_scalar`: Generates a linearly spaced scalar array for integration purposes.
    - :func:`trapezoidal_integration`: Performs trapezoidal numerical integration on a sampled function W over an interval.
    - :func:`compute_dist_along_branch`: Computes Euclidean distances along a branch for multiple contact points and scalar steps.
"""

# import relevant libraries
import numpy as np
from numba import njit, prange, float64, float32, int32

# Find the position of the first significant figure
def first_significant_figure_position(number:float)->float:
    """
    Calculate the position (value) of the first significant figure of a number.

    Inputs
    ------
    number : float
        Input number to analyze.

    Outputs
    -------
    float
        The value of the first significant figure's position.
        For example, if number = 345.6, returns 100.0.
        Special case: returns 0 if number is zero.

    Examples
    --------
    >>> first_significant_figure_position(345.6)
    100.0
    >>> first_significant_figure_position(0.00789)
    0.001
    >>> first_significant_figure_position(0)
    0
    """
    if number == 0:
        return 0  # Special case for zero
    order_of_magnitude = np.floor(np.log10(abs(number))) # Get the order of magnitude of the number
    first_significant_position = 10 ** order_of_magnitude # Calculate the position of the first significant figure
    return first_significant_position

# Creating the scalar for integration
def integration_scalar(s0:float, s1:float, n:float)->np.ndarray: 
    """
    Create a linearly spaced scalar array for integration over [s0, s1].

    Inputs
    ------
    s0 : float
        Start of the integration interval.
    s1 : float
        End of the integration interval.
    n : int
        Number of sample points.

    Outputs
    -------
    np.ndarray, shape (n,)
        Linearly spaced array of scalars from s0 to s1.

    Examples
    --------
    >>> integration_scalar(0, 1, 5)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    s = np.linspace(s0, s1, n)
    return s

# Trapezoidal integration
def trapezoidal_integration(s0, s1, n, W):
    r"""
    Compute numerical integral of sampled function W using the trapezoidal rule.

    .. math::

        I_m = \int_{s_0}^{s_1} W_m(s) \, ds
        \approx \frac{\Delta s}{2} \left( W_m(s_0) + 2 \sum_{i=1}^{n-2} W_m(s_i) + W_m(s_{n-1}) \right) \\
        \text{for } m = 0, \ldots, M-1

    where:
    :math:`I_m` is the integral of the m-th component of W,
    :math:`W_m(s)` is the m-th component of the function W evaluated at s,
    :math:`s_0` and :math:`s_1` are the integration bounds,
    :math:`n` is the number of sample points,
    :math:`\Delta s = \frac{s_1 - s_0}{n-1}` is the step size,
    and :math:`M` is the number of components (columns) in W.


    Inputs
    ------
    s0 : float
        Start of the integration interval.
    s1 : float
        End of the integration interval.
    n : int
        Number of sample points.
    W : np.ndarray, shape (n, m)
        Values of the function sampled at the points defined by s0, s1, and n.
        Integration is performed along the first axis.

    Outputs
    -------
    np.ndarray, shape (m,)
        Resulting integral values for each column of W.

    Examples
    --------
    >>> s0, s1, n = 0, 1, 5
    >>> W = np.array([[0], [1], [4], [9], [16]])
    >>> trapezoidal_integration(s0, s1, n, W)
    array([5.0])
    """
    ds = (s1 - s0) / (n - 1)
    Wint = (ds / 2) * (W[0, :] + 2 * np.sum(W[1:n-1, :], axis=0) + W[n-1, :])
    return Wint

# Numba version
@njit(float64[:,:](float64[:,:], float64[:], float32[:,:], int32[:]),parallel=True)
def compute_dist_along_branch(r_ri_c, s, BranchVector_i, part_ind_c):
    r"""
    Compute Euclidean distances along a branch for multiple contact points and scalar steps.

    .. math::

        d_{ij} = \left\| \mathbf{r}_{ri}^{(j)} + s_i \, \mathbf{B}_{\text{part}}^{(k_j)} \right\|_2
    
    where:

    :math:`d_{ij}` is the distance for scalar step :math:`s_i` and contact point :math:`j`,
    :math:`\mathbf{r}_{ri}^{(j)}` is the displacement vector from the reference point to contact point :math:`j`,
    :math:`s_i` is the scalar step along the branch,
    :math:`\mathbf{B}_{\text{part}}^{(k_j)}` is the branch direction vector associated with particle :math:`k_j`,
    and :math:`k_j = \text{part_ind_c}[j]` is the index mapping contact points to particles.

    Inputs
    ------
    r_ri_c : np.ndarray, shape (n_contacts, 3)
        Displacement vectors from reference points to contact points.
    s : np.ndarray, shape (n_s,)
        Scalars along the branch for which distances are computed.
    BranchVector_i : np.ndarray, shape (n_particles, 3)
        Branch direction vectors associated with each particle.
    part_ind_c : np.ndarray, shape (n_contacts,)
        Indices mapping contacts to corresponding particles in BranchVector_i.

    Outputs
    -------
    np.ndarray, shape (n_s, n_contacts)
        Euclidean distances along the branch for each scalar step and contact.

    Notes
    -----
    This function is JIT-compiled with Numba and parallelized for efficiency.

    Examples
    --------
    >>> r_ri_c = np.array([[1,0,0], [0,1,0]], dtype=np.float64)
    >>> s = np.array([0, 1], dtype=np.float64)
    >>> BranchVector_i = np.array([[1,0,0], [0,1,0]], dtype=np.float32)
    >>> part_ind_c = np.array([0, 1], dtype=np.int32)
    >>> compute_dist_along_branch(r_ri_c, s, BranchVector_i, part_ind_c)
    array([[1., 1.],
           [2., 2.]])
    """
    n_s = s.shape[0]
    n_contacts = r_ri_c.shape[0]
    out = np.empty((n_s, n_contacts), dtype=np.float64)
    for i in prange(n_s):
        for j in range(n_contacts):
            vec = r_ri_c[j] + s[i] * BranchVector_i[part_ind_c[j]]
            out[i, j] = np.sqrt(np.sum(vec ** 2))
    return out