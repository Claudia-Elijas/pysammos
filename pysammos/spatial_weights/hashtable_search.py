"""
This module provides functions to build and query hash tables for fast approximate evaluation of expensive functions.
A hash table is constructed by precomputing function outputs at discrete input intervals. Queries on arbitrary inputs
are answered by linear interpolation between the entries of this table, trading some precision for speed.

This module contains the following functions:
    - :func:`make_hash_table`: Constructs a hash table of function values by sampling a given function over a specified range.
    - :func:`hash_table_search_1d`: Efficiently searches the hash table for 1D arrays of query values,
      interpolating linearly between its entries.
    - :func:`hash_table_search_2d`: Efficiently searches the hash table for 2D arrays of query values,
      interpolating linearly between its entries.
    - :func:`hash_table_search`: Dispatcher that routes queries to the correct 1D or 2D search function based on input dimensionality.

"""

# import relevant libraries 
import numpy as np
from numba import njit, prange, float64, float32, int64, int32
from .utils import *

 
# Build hash table
def make_hash_table(func, max_value, sensitivity):
    """
    Construct a hash table by sampling a function at discrete input intervals.

    Inputs
    ------
    func : callable
        The function to sample. Must accept two arguments: max_value and input_value.
    max_value : float
        The maximum input value to sample up to.
    sensitivity : float
        Controls the granularity of the hash table sampling by dividing the
        position of the first significant digit of max_value.

    Outputs
    -------
    hash_table_outputs : np.ndarray
        Function values at k*step_size for k = 0, 1, ... below max_value, then the value just
        below max_value, then zero.
    step_size : float
        The sampling interval between hash table inputs.

    Notes
    -----
    The step_size is calculated as the position of the first significant figure of max_value
    divided by sensitivity. The function is sampled at the n points k*step_size that lie below
    max_value, and :func:`hash_table_search` interpolates linearly between the samples, so that
    the error of a look-up is of second order in step_size. The value just below max_value (the
    limit from inside, where a function such as the HeavySide or the truncated Gaussian jumps) is
    stored at n*step_size, the end of the bin that holds max_value, so that bin interpolates
    towards it (to first order in step_size, since max_value is in general not at the end of
    the bin). A zero follows, which the search returns from (n+1)*step_size on; the
    coarse-graining only looks up distances up to max_value. The function is vectorized over
    these inputs to produce the outputs.
    
    """
    step_size = first_significant_figure_position(max_value) / sensitivity  # Step size for the hash table
    edges = np.arange(int(np.ceil(max_value / step_size)) + 1) * step_size
    edges = edges[edges < max_value] # k*step_size below max_value
    hash_table_inputs = np.append(edges, np.nextafter(max_value, 0.0))  # Create an array of input data, the last just below max_value
    vectorized_func = np.vectorize(func)
    hash_table_outputs = vectorized_func(max_value,hash_table_inputs)  # Create an array of output data
    hash_table_outputs = np.append(np.asarray(hash_table_outputs, dtype=np.float64), 0.0) # zero beyond max_value
    return hash_table_outputs, step_size

# Hash table search algorithm
@njit(float64[:](float64[:], float64[:], float64), parallel=True)
def hash_table_search_1d(query_values, hash_table_outputs, step_size):
    """
    Approximate function evaluation for 1D query arrays by linear interpolation in the precomputed hash table.

    Inputs
    ------
    query_values : np.ndarray, shape (N,)
        1D array of input values to query.
    hash_table_outputs : np.ndarray, shape (M,)
        Precomputed function values stored in the hash table.
    step_size : float
        Sampling interval used to build the hash table.

    Outputs
    -------
    query_results : np.ndarray, shape (N,)
        Approximated function values corresponding to each input query: the last entry of the
        table beyond its end, the first for negative values, and NaN for a NaN query.
    """
    query_results = np.empty_like(query_values)
    max_index = len(hash_table_outputs) - 1

    for q in prange(query_values.shape[0]):
        position = query_values[q] * (1 / step_size)
        if position < max_index:
            position = max(position, 0.0)
            hash_index = int(np.floor(position))
            fraction = position - hash_index
            query_results[q] = hash_table_outputs[hash_index] + fraction * (hash_table_outputs[hash_index + 1] - hash_table_outputs[hash_index])
        elif position >= max_index:
            query_results[q] = hash_table_outputs[max_index]
        else: # NaN query, which has no index in the table
            query_results[q] = np.nan

    return query_results

@njit(float64[:,:](float64[:,:], float64[:], float64), parallel=True)
def hash_table_search_2d(query_values, hash_table_outputs, step_size):
    """
    Approximate function evaluation for 2D query arrays by linear interpolation in the precomputed hash table.

    Inputs
    ------
    query_values : np.ndarray, shape (N, M)
        2D array of input values to query.
    hash_table_outputs : np.ndarray, shape (K,)
        Precomputed function values stored in the hash table.
    step_size : float
        Sampling interval used to build the hash table.

    Outputs
    -------
    out : np.ndarray, shape (N, M)
        Approximated function values corresponding to each input query: the last entry of the
        table beyond its end, the first for negative values, and NaN for a NaN query.
    
    """

    out = np.empty_like(query_values)
    max_index = len(hash_table_outputs) - 1

    for i in prange(query_values.shape[0]):
        for j in range(query_values.shape[1]):
            position = query_values[i, j] * (1 / step_size)
            if position < max_index:
                position = max(position, 0.0)
                hash_index = int(np.floor(position))
                fraction = position - hash_index
                out[i, j] = hash_table_outputs[hash_index] + fraction * (hash_table_outputs[hash_index + 1] - hash_table_outputs[hash_index])
            elif position >= max_index:
                out[i, j] = hash_table_outputs[max_index]
            else: # NaN query, which has no index in the table
                out[i, j] = np.nan

    return out

# Dispatcher (pure Python)
def hash_table_search(query_values, hash_table_outputs, step_size):
    """
    Dispatch function to route query_values to the appropriate hash table search function
    based on dimensionality.

    Inputs
    ------
    query_values : np.ndarray
        Input values to query. Must be 1D or 2D array.
    hash_table_outputs : np.ndarray
        Precomputed hash table outputs.
    step_size : float
        Sampling interval used in hash table construction.

    Outputs
    -------
    np.ndarray
        Approximated function values with same shape as query_values.

    Raises
    ------
    ValueError
        If query_values is not 1D or 2D.
    """
    if query_values.ndim == 1:
        return hash_table_search_1d(query_values, hash_table_outputs, step_size)
    elif query_values.ndim == 2:
        return hash_table_search_2d(query_values, hash_table_outputs, step_size)
    else:
        raise ValueError("query_values must be 1D or 2D")



