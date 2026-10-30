"""
This module provides utility functions commonly used in numerical computations, particularly in physics and
engineering simulations involving integration along branches or paths, and geometric distance calculations.

This module includes the following functions:

    - :func:`first_significant_figure_position`: Computes the positional value of the first significant figure in a given number.
    - :func:`integration_scalar`: Generates a linearly spaced scalar array for integration purposes.
    - :func:`trapezoidal_integration`: Performs trapezoidal numerical integration on a sampled function W over an interval.
    - :func:`compute_dist_along_branch`: Computes Euclidean distances along a branch for multiple contact points and scalar steps.
    - :func:`gauss_legendre_points`: Returns Gauss-Legendre points and weights on [0, 1].
    - :func:`compute_dist_along_branch_in_support`: Computes distances along the part of each branch inside the kernel support
      and the length of that part.
"""

# import relevant libraries
import numpy as np
from numba import njit, prange, float64, float32, int32, types
from typing import Tuple

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
    :math:`\mathbf{B}_{\text{part}}^{(k_j)}` is the branch direction vector associated with particle :math:`k_j`, and :math:`k_j = \mathbf{part\_ind\_c}_{j}` is the index mapping contact points to particles.

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

def gauss_legendre_points(n:int)->Tuple[np.ndarray, np.ndarray]:
    r"""
    Gauss-Legendre points and weights on [0, 1].

    .. math::

        \int_0^1 f(s)\,ds \approx \sum_{k=1}^{n} w_k\, f(s_k), \qquad \sum_{k=1}^{n} w_k = 1

    Inputs
    ------
    n : int
        Number of points.

    Outputs
    -------
    s : np.ndarray, shape (n,)
        Points in (0, 1).
    w : np.ndarray, shape (n,)
        Weights, summing to 1.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w

# Numba version
@njit(types.Tuple((float64[:,:], float64[:]))(float64[:,:], float64[:], float32[:,:], int32[:], float64), parallel=True)
def compute_dist_along_branch_in_support(r_ri_c, s, BranchVector_i, part_ind_c, c):
    r"""
    Compute distances along the part of each branch that lies inside the kernel support.

    The points of a branch are :math:`\mathbf{r}_{ri} + t\,\mathbf{B}`, :math:`0 \le t \le 1`. The kernel
    vanishes beyond c, so the integral over t is taken over the interval
    :math:`[t_{lo}, t_{lo} + \ell]` where :math:`\|\mathbf{r}_{ri} + t\,\mathbf{B}\| \le c` (between the
    roots of a quadratic, clipped to [0, 1]), at the points :math:`t_i = t_{lo} + \ell\, s_i`:

    .. math::

        \int_0^1 W(\|\mathbf{r}_{ri} + t\,\mathbf{B}\|)\,dt \approx \ell \sum_i w_i\, W(d_{ij})

    No quadrature point then straddles the cutoff, where the truncated Gaussian and the HeavySide
    kernels jump. For the Lucy kernel the integrand has a jump in its third derivative where a
    branch passes through the grid point, which limits a single weight there to within 2.5e-6 of
    W(0) for a branch of c/3 (spheres at the default resolution), 3.4e-5 for 0.8 c, and 5e-4 at
    most for longer branches.

    Inputs
    ------
    r_ri_c : np.ndarray, shape (n_contacts, 3)
        Displacement vectors from the particle centres to the grid points.
    s : np.ndarray, shape (n_s,)
        Quadrature points on [0, 1] (see :func:`gauss_legendre_points`).
    BranchVector_i : np.ndarray, shape (n_particles, 3)
        Branch vectors, from the contact point to the particle centre.
    part_ind_c : np.ndarray, shape (n_contacts,)
        Indices mapping contacts to corresponding particles in BranchVector_i.
    c : float
        Cutoff distance of the kernel.

    Outputs
    -------
    dist : np.ndarray, shape (n_s, n_contacts)
        Distances to the grid point at the quadrature points of the part of each branch inside the support.
    length : np.ndarray, shape (n_contacts,)
        Length of that part as a fraction of the branch, 0 where no part of the branch lies within c.

    Examples
    --------
    >>> r_ri_c = np.array([[0.5, 0, 0]], dtype=np.float64)   # grid point 0.5 from the centre
    >>> B = np.array([[1, 0, 0]], dtype=np.float32)          # branch of length 1 pointing away from it
    >>> dist, length = compute_dist_along_branch_in_support(r_ri_c, np.array([0.5]), B, np.array([0], dtype=np.int32), 1.0)
    >>> length                                                # only t <= 0.5 lies within c = 1
    array([0.5])
    >>> dist                                                  # midpoint of that part, t = 0.25
    array([[0.75]])
    """
    n_s = s.shape[0]
    n_contacts = r_ri_c.shape[0]
    dist = np.empty((n_s, n_contacts), dtype=np.float64)
    length = np.zeros(n_contacts, dtype=np.float64)
    for j in prange(n_contacts):
        k = part_ind_c[j]
        b0 = np.float64(BranchVector_i[k, 0])
        b1 = np.float64(BranchVector_i[k, 1])
        b2 = np.float64(BranchVector_i[k, 2])
        r0 = r_ri_c[j, 0]
        r1 = r_ri_c[j, 1]
        r2 = r_ri_c[j, 2]
        a = b0*b0 + b1*b1 + b2*b2
        h = r0*b0 + r1*b1 + r2*b2 # a t^2 + 2 h t + q = 0 where the branch crosses the cutoff
        q = r0*r0 + r1*r1 + r2*r2 - c*c
        t_lo = 0.0
        t_hi = 0.0
        if a > 0.0:
            disc = h*h - a*q
            if disc > 0.0:
                root = np.sqrt(disc)
                if h >= 0.0: # roots in the form that avoids cancellation
                    t1 = (-h - root) / a
                    t2 = q / (-h - root)
                else:
                    t2 = (-h + root) / a
                    t1 = q / (-h + root)
                t_lo = max(min(t1, t2), 0.0)
                t_hi = min(max(t1, t2), 1.0)
        elif q <= 0.0: # branch of zero length, with its centre within c
            t_hi = 1.0
        if t_hi < t_lo:
            t_hi = t_lo
        ell = t_hi - t_lo
        length[j] = ell
        for i in range(n_s):
            t = t_lo + ell * s[i]
            d0 = r0 + t*b0
            d1 = r1 + t*b1
            d2 = r2 + t*b2
            dist[i, j] = np.sqrt(d0*d0 + d1*d1 + d2*d2)
    return dist, length
