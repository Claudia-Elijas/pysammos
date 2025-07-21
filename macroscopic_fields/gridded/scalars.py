import numpy as np
from numba import njit, prange, int32, float32, float64


# Polydisperse
@njit(float64[:](float64[:],int32[:], int32[:], float32[:], float32[:], int32[:]), parallel=True)
def scalar_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase):
    Ngridpoints = len(grid_indices) - 2  # because it's padded [0], [len(flat_visibility)]
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_sum = np.zeros(Nphases)
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                ds = Data_scale[idx]
                phase_sum[phase] += w * d * ds
        for p in range(Nphases):
            CG_Field[g, p + 1] = phase_sum[p]
            CG_Field[g, 0] += phase_sum[p]
    return CG_Field

@njit(float64[:](float64[:],int32[:], int32[:], float32[:], int32[:]),parallel=True)
def scalar_polydisperse(weights, visibility, grid_indices, Data, Phase):
    Ngridpoints = len(grid_indices) - 2  # because it's padded [0], [len(flat_visibility)]
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_sum = np.zeros(Nphases)  # Local buffer for each grid point
        for i in range(start, end): # loop over particles
            idx = visibility[i] # get the particle index
            phase = Phase[idx] # get the phase
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                phase_sum[phase] += w * d # weighted sum
        for p in range(Nphases): # store the weighted sum in the CG_Field
            CG_Field[g, p + 1] = phase_sum[p]
            CG_Field[g, 0] += phase_sum[p]
    return CG_Field


# Monodisperse
@njit(float64[:](float64[:],int32[:], int32[:], float32[:], float32[:]),parallel=True)
def scalar_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale):
    Ngridpoints = len(grid_indices) - 2  # Padding assumed: grid_indices[0] = 0, grid_indices[-1] = len(visibility)
    CG_Field = np.zeros(Ngridpoints, dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        val = 0.0
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            ds = Data_scale[idx]
            val += w * d * ds
        CG_Field[g] = val
    return CG_Field

@njit(float64[:](float64[:],int32[:], int32[:], float32[:]),parallel=True)
def scalar_monodisperse(weights, visibility, grid_indices, Data):
    Ngridpoints = len(grid_indices) - 2  # Padding assumed: grid_indices[0] = 0, grid_indices[-1] = len(visibility)
    CG_Field = np.zeros(Ngridpoints, dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        val = 0.0
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            val += w * d
        CG_Field[g] = val
    return CG_Field


@njit(parallel=True)
def mean_grainsize(weights, visibility, grid_indices, Data, n_flag):
    # nflag = 3 for VOLUME, nflag = 2 for AREA
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros(Ngridpoints)

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        numerator = 0.0
        denominator = 0.0

        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            numerator += w * d**(n_flag + 1)
            denominator += w * d**n_flag

        if denominator != 0.0:
            CG_Field[g] = numerator / denominator
        else:
            CG_Field[g] = 0.0

    return CG_Field

@njit(parallel=True)
def scalar_x_volume(weights, visibility, grid_indices, Data):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros(Ngridpoints)

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        numerator = 0.0
        denominator = 0.0

        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            numerator += w * d
            denominator += w

        if denominator != 0.0:
            CG_Field[g] = numerator / denominator
        else:
            CG_Field[g] = 0.0

    return CG_Field