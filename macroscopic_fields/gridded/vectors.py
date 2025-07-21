import numpy as np
from numba import njit, prange, int32, float32, float64

# polydisperse
@njit(float64[:](float64[:],int32[:], int32[:], float32[:], float32[:], int32[:]),parallel=True)
def vector_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase):
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_vector = np.zeros((Nphases, 3))
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                ds = Data_scale[idx]
                phase_vector[phase, 0] += w * d[0] * ds
                phase_vector[phase, 1] += w * d[1] * ds
                phase_vector[phase, 2] += w * d[2] * ds
        for p in range(Nphases):
            vec = phase_vector[p]
            for d in range(3):
                CG_Field[g, p + 1, d] = vec[d]
                CG_Field[g, 0, d] += vec[d]
    return CG_Field

@njit(float64[:](float64[:],int32[:], int32[:], float32[:], int32[:]),parallel=True)
def vector_polydisperse(weights, visibility, grid_indices, Data, Phase):
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3), dtype=np.float64)
    for g in prange(Ngridpoints):
        start = grid_indices[g] ; end = grid_indices[g+1]
        phase_vector = np.zeros((Nphases, 3))
        for i in range(start, end):
            idx = visibility[i]
            phase = Phase[idx]
            if 0 <= phase < Nphases:
                w = weights[i]
                d = Data[idx]
                phase_vector[phase, 0] += w * d[0]
                phase_vector[phase, 1] += w * d[1]
                phase_vector[phase, 2] += w * d[2]
        for p in range(Nphases):
            vec = phase_vector[p]
            for d in range(3):
                CG_Field[g, p + 1, d] = vec[d]
                CG_Field[g, 0, d] += vec[d]
    return CG_Field


# monodisperse
@njit(float64[:](float64[:],int32[:], int32[:], float32[:], float32[:]), parallel=True)
def vector_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3), dtype=np.float64)  # No phase info in monodisperse case
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        vec = np.zeros(3)
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            ds = Data_scale[idx]
            vec[0] += w * d[0] * ds
            vec[1] += w * d[1] * ds
            vec[2] += w * d[2] * ds
        for d in range(3):
            CG_Field[g, d] = vec[d]

    return CG_Field

@njit(float64[:](float64[:],int32[:], int32[:], float32[:]),parallel=True)
def vector_monodisperse(weights, visibility, grid_indices, Data):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3), dtype=np.float64)  # No phase info in monodisperse case
    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        vec = np.zeros(3)
        for i in range(start, end):
            idx = visibility[i]
            w = weights[i]
            d = Data[idx]
            vec[0] += w * d[0]
            vec[1] += w * d[1]
            vec[2] += w * d[2]
        for d in range(3):
            CG_Field[g, d] = vec[d]
    return CG_Field