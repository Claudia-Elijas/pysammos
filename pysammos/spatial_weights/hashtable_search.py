import numpy as np
from numba import njit, prange, float64, float32, int64, int32
from .utils import *


# Build hash table
def make_hash_table(func, max_value, sensitivity):
    step_size = first_significant_figure_position(max_value) / sensitivity  # Step size for the hash table
    hash_table_inputs = np.arange(step_size, max_value + step_size, step_size)  # Create an array of input data
    vectorized_func = np.vectorize(func)
    hash_table_outputs = vectorized_func(max_value,hash_table_inputs)  # Create an array of output data
    return hash_table_outputs, step_size

# Hash table search algorithm
@njit(float64[:](float64[:], float64[:], float64), parallel=True)
def hash_table_search_1d(query_values, hash_table_outputs, step_size):
    query_results = np.empty_like(query_values)
    max_index = len(hash_table_outputs) - 1

    for q in prange(query_values.shape[0]):
        hash_index = int(np.floor(query_values[q] * (1 / step_size)))
        hash_index = min(max(hash_index, 0), max_index)
        query_results[q] = hash_table_outputs[hash_index]

    return query_results

@njit(float64[:,:](float64[:,:], float64[:], float64), parallel=True)
def hash_table_search_2d(query_values, hash_table_outputs, step_size):
    out = np.empty_like(query_values)
    max_index = len(hash_table_outputs) - 1

    for i in prange(query_values.shape[0]):
        for j in range(query_values.shape[1]):
            val = query_values[i, j]
            hash_index = int(np.floor(val * (1 / step_size)))
            hash_index = min(max(hash_index, 0), max_index)
            out[i, j] = hash_table_outputs[hash_index]

    return out

# Dispatcher (pure Python)
def hash_table_search(query_values, hash_table_outputs, step_size):
    if query_values.ndim == 1:
        return hash_table_search_1d(query_values, hash_table_outputs, step_size)
    elif query_values.ndim == 2:
        return hash_table_search_2d(query_values, hash_table_outputs, step_size)
    else:
        raise ValueError("query_values must be 1D or 2D")



