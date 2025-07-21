import numpy as np
from numba import njit, prange

# Find the position of the first significant figure
def first_significant_figure_position(number):
    if number == 0:
        return 0  # Special case for zero
    order_of_magnitude = np.floor(np.log10(abs(number))) # Get the order of magnitude of the number
    first_significant_position = 10 ** order_of_magnitude # Calculate the position of the first significant figure
    return first_significant_position

# Creating the scalar for integration
def integration_scalar(s0, s1, n): 
    s = np.linspace(s0, s1, n)
    return s

# Trapezoidal integration
def trapezoidal_integration(s0, s1, n, W):
    ds = (s1 - s0) / (n - 1)
    Wint = (ds / 2) * (W[0, :] + 2 * np.sum(W[1:n-1, :], axis=0) + W[n-1, :])
    return Wint

# Numba version
@njit(parallel=True)
def compute_dist_along_branch(r_ri_c, s, BranchVector_i, part_ind_c):
    n_s = s.shape[0]
    n_contacts = r_ri_c.shape[0]
    out = np.empty((n_s, n_contacts), dtype=np.float64)
    for i in prange(n_s):
        for j in range(n_contacts):
            vec = r_ri_c[j] + s[i] * BranchVector_i[part_ind_c[j]]
            out[i, j] = np.sqrt(np.sum(vec ** 2))
    return out