import numpy as np
from numba import njit, prange

# Find the position of the first significant figure
def first_significant_figure_position(number):
    if number == 0:
        return 0  # Special case for zero
    order_of_magnitude = np.floor(np.log10(abs(number))) # Get the order of magnitude of the number
    first_significant_position = 10 ** order_of_magnitude # Calculate the position of the first significant figure
    return first_significant_position
# Hash table search
def make_hash_table(func, max_value, sensitivity):
    step_size = first_significant_figure_position(max_value) / sensitivity  # Step size for the hash table
    hash_table_inputs = np.arange(step_size, max_value + step_size, step_size)  # Create an array of input data
    vectorized_func = np.vectorize(func)
    hash_table_outputs = vectorized_func(max_value,hash_table_inputs)  # Create an array of output data
    return hash_table_outputs, step_size

# Hash table search
@njit(parallel=True)
def hash_table_search(query_values, hash_table_outputs, step_size):
    # 2. match your data to the table outputs
    query_shape = query_values.shape; #print(f"query_shape: {query_shape}")
    query_flat = query_values.reshape(-1); #print(f"query_flat: {query_flat.shape}")
    #print(f"query flattened")
    query_results = np.empty(len(query_flat)); #print(f"query_results: {query_results.shape}")
    #print(f"query result made")
    max_index = len(hash_table_outputs) - 1; #print(f"max_index: {max_index}")
    for q in prange(len(query_flat)):
        hash_index = int(np.floor(query_flat[q] * (1 / step_size)))
        hash_index = min(max(hash_index, 0), max_index)
        query_results[q] = hash_table_outputs[hash_index]
    #print(f"loop done")
    query_results = query_results.reshape(query_shape); #print(f"query_results: {query_results.shape}")
    #print(f"query reshaped")
    # query_indices_old = np.floor(query_values * (1 / step_size)).astype(int)  # Find the index of the query values in the hash table
    # if clip == True:
    #     query_indices_old = np.clip(query_indices_old, 0, len(hash_table_inputs) - 1) # Ensure query_indices are within bounds
    # query_results_old = hash_table_outputs[query_indices_old]

    return query_results  #, query_results_old

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
def compute_dist_along_branch_numba(r_ri_c, s, BranchVector_i, part_ind_c):
    n_s = s.shape[0]
    n_contacts = r_ri_c.shape[0]
    out = np.empty((n_s, n_contacts), dtype=np.float64)
    for i in prange(n_s):
        for j in range(n_contacts):
            vec = r_ri_c[j] + s[i] * BranchVector_i[part_ind_c[j]]
            out[i, j] = np.sqrt(np.sum(vec ** 2))
    return out