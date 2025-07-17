import numpy as np
from numba import njit, prange
from .utils import *


# Build hash table
def make_hash_table(func, max_value, sensitivity):
    step_size = first_significant_figure_position(max_value) / sensitivity  # Step size for the hash table
    hash_table_inputs = np.arange(step_size, max_value + step_size, step_size)  # Create an array of input data
    vectorized_func = np.vectorize(func)
    hash_table_outputs = vectorized_func(max_value,hash_table_inputs)  # Create an array of output data
    return hash_table_outputs, step_size

# Hash table search algorithm
@njit(parallel=True)
def hash_table_search(query_values, hash_table_outputs, step_size):
    # Match your data to the table outputs
    query_shape = query_values.shape; 
    query_flat = query_values.reshape(-1); 
    query_results = np.empty(len(query_flat)); 

    # Calculate the hash index for each query value
    max_index = len(hash_table_outputs) - 1; 
    for q in prange(len(query_flat)):
        hash_index = int(np.floor(query_flat[q] * (1 / step_size))) # Calculate the hash index based on the step size
        hash_index = min(max(hash_index, 0), max_index) # Ensure the index is within bounds
        query_results[q] = hash_table_outputs[hash_index] # Get the corresponding output from the hash table
    
    # Reshape the results to match the original query shape
    query_results = query_results.reshape(query_shape); 

    return query_results 


