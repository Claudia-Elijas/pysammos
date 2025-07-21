import numpy as np
from numba import njit, prange
from numba import get_num_threads
# print the number of threads used by numba
print(" ......................... Numba is using", get_num_threads(), "threads .........................")
   
# Monodisperse vs Polydisperse functions =========================================================
# ---------------- SCALAR ----------------
@njit(parallel=True)
def CG_Scalar_Polydisperse(weights, visibility, grid_indices, Data, Data_scale, Phase):
    Ngridpoints = len(grid_indices) - 2  # because it's padded [0], [len(flat_visibility)]
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1))
    if Data_scale is None:
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
    else:
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
@njit(parallel=True)
def CG_Scalar_Monodisperse(weights, visibility, grid_indices, Data, Data_scale):
    Ngridpoints = len(grid_indices) - 2  # Padding assumed: grid_indices[0] = 0, grid_indices[-1] = len(visibility)
    CG_Field = np.zeros(Ngridpoints)
    if Data_scale is None:
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
    else:
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
# ---------------- VECTOR ----------------
@njit(parallel=True)
def CG_Vector_Polydisperse(weights, visibility, grid_indices, Data, Data_scale, Phase):
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3))
    if Data_scale is None:
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

    else:
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
@njit(parallel=True)
def CG_Vector_Monodisperse(weights, visibility, grid_indices, Data, Data_scale):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3))  # No phase info in monodisperse case

    if Data_scale is None:
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
    else:
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
# ---------------- TENSOR ----------------
@njit(parallel=True)
def CG_Tensor_Polydisperse(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase):
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3, 3))
    if Data_scale is None:
        for g in prange(Ngridpoints):
            start = grid_indices[g] ; end = grid_indices[g+1]
            phase_tensor = np.zeros((Nphases, 3, 3))
            for i in range(start, end):
                idx = visibility[i]
                phase = Phase[idx]
                if 0 <= phase < Nphases:
                    w = weights[i]
                    d1 = Data1[idx]
                    d2 = Data2[idx]
                    for j in range(3):
                        for k in range(3):
                            phase_tensor[phase, j, k] += w * d1[j] * d2[k]
            for p in range(Nphases):
                tensor = phase_tensor[p]
                for j in range(3):
                    for k in range(3):
                        CG_Field[g, p + 1, j, k] = tensor[j, k]
                        CG_Field[g, 0, j, k] += tensor[j, k]

    else:
        for g in prange(Ngridpoints):
            start = grid_indices[g] ; end = grid_indices[g+1]
            phase_tensor = np.zeros((Nphases, 3, 3))
            for i in range(start, end):
                idx = visibility[i]
                phase = Phase[idx]
                if 0 <= phase < Nphases:
                    w = weights[i]
                    d1 = Data1[idx]
                    d2 = Data2[idx]
                    ds = Data_scale[idx]
                    for j in range(3):
                        for k in range(3):
                            phase_tensor[phase, j, k] += w * d1[j] * d2[k] * ds
            for p in range(Nphases):
                tensor = phase_tensor[p]
                for j in range(3):
                    for k in range(3):
                        CG_Field[g, p + 1, j, k] = tensor[j, k]
                        CG_Field[g, 0, j, k] += tensor[j, k]
    return CG_Field
@njit(parallel=True)
def CG_Tensor_Monodisperse(weights, visibility, grid_indices, Data1, Data2, Data_scale):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3, 3))  # Only one tensor per grid point

    if Data_scale is None:
        for g in prange(Ngridpoints):
            start = grid_indices[g]
            end = grid_indices[g + 1]
            tensor = np.zeros((3, 3))
            for i in range(start, end):
                idx = visibility[i]
                w = weights[i]
                d1 = Data1[idx]
                d2 = Data2[idx]
                for j in range(3):
                    for k in range(3):
                        tensor[j, k] += w * d1[j] * d2[k]
            for j in range(3):
                for k in range(3):
                    CG_Field[g, j, k] = tensor[j, k]
    else:
        for g in prange(Ngridpoints):
            start = grid_indices[g]
            end = grid_indices[g + 1]
            tensor = np.zeros((3, 3))
            for i in range(start, end):
                idx = visibility[i]
                w = weights[i]
                d1 = Data1[idx]
                d2 = Data2[idx]
                ds = Data_scale[idx]
                for j in range(3):
                    for k in range(3):
                        tensor[j, k] += w * d1[j] * d2[k] * ds
            for j in range(3):
                for k in range(3):
                    CG_Field[g, j, k] = tensor[j, k]
    

    return CG_Field
# ---------------- KINETIC TENSOR ----------------
@njit(parallel=True)
def KineticTensor_vInterpolated_Polydisperse(weights, visibility, grid_indices, displacement,
                                     Particle_Velocity, Particle_Mass, 
                                     Velocity_Field, Velocity_Field_Gradient, 
                                     phase_array): 
    
    Ngridpoints = len(grid_indices) - 2
    Nphases = 0
    for i in range(len(phase_array)):
        if phase_array[i] > Nphases:
            Nphases = phase_array[i]
    Nphases += 1

    KineticTensor = np.zeros((Ngridpoints, Nphases + 1, 3, 3), dtype=np.float64)

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        Velocity_Field_g = Velocity_Field[g]
        GradV_g = Velocity_Field_Gradient[g]

        phase_tensor = np.zeros((Nphases, 3, 3), dtype=np.float64)

        for i in range(start, end):
            idx = visibility[i]
            phase = phase_array[idx]

            if 0 <= phase < Nphases:
                r_ri = displacement[i]
                w = weights[i]
                v_particle = Particle_Velocity[idx]
                m_particle = Particle_Mass[idx]

                # interpolated_velocity = np.zeros(3, dtype=np.float64)
                # for j in range(3):
                #     interpolated_velocity[j] = Velocity_Field_g[j]
                #     for k in range(3):
                #         interpolated_velocity[j] += r_ri[k] * GradV_g[k, j]
    
                interpolated_velocity = np.zeros(3, dtype=np.float64)
                for j in range(3):
                    interpolated_velocity_j = Velocity_Field_g[j]
                    for k in range(3):
                        interpolated_velocity_j += r_ri[k] * GradV_g[k, j]
                    interpolated_velocity[j] = interpolated_velocity_j

                velocity_fluctuation = np.zeros(3, dtype=np.float64)
                for j in range(3):
                    velocity_fluctuation[j] = v_particle[j] - interpolated_velocity[j]

                for j in range(3):
                    for k in range(3):
                        phase_tensor[phase, j, k] += m_particle * velocity_fluctuation[j] * velocity_fluctuation[k] * w

        for p in range(Nphases):
            for j in range(3):
                for k in range(3):
                    KineticTensor[g, p + 1, j, k] = phase_tensor[p, j, k]
                    KineticTensor[g, 0, j, k] += phase_tensor[p, j, k]

    return KineticTensor
@njit(parallel=True)
def KineticTensor_vInterpolated_Monodisperse(weights, visibility, grid_indices, displacement,
                                             Particle_Velocity, Particle_Mass, 
                                             Velocity_Field, Velocity_Field_Gradient):
    Ngridpoints = len(grid_indices) - 2
    KineticTensor = np.zeros((Ngridpoints, 3, 3))  # Only one tensor per grid point

    for g in prange(Ngridpoints):
        start = grid_indices[g]
        end = grid_indices[g + 1]
        tensor = np.zeros((3, 3))

        Velocity_Field_g = Velocity_Field[g]
        GradV_g = Velocity_Field_Gradient[g]

        for i in range(start, end):
            idx = visibility[i]
            r_ri = displacement[i]
            w = weights[i]
            v_particle = Particle_Velocity[idx]
            m_particle = Particle_Mass[idx]

            # Interpolate velocity
            # interpolated_velocity = Velocity_Field_g.copy()
            # for j in range(3):
            #     for k in range(3):
            #         interpolated_velocity[j] += r_ri[k] * GradV_g[k, j]
            interpolated_velocity = np.zeros(3, dtype=np.float64)
            for j in range(3):
                interpolated_velocity_j = Velocity_Field_g[j]
                for k in range(3):
                    interpolated_velocity_j += r_ri[k] * GradV_g[k, j]
                interpolated_velocity[j] = interpolated_velocity_j

            # Velocity fluctuation
            v_fluc = np.zeros(3)
            for j in range(3):
                v_fluc[j] = v_particle[j] - interpolated_velocity[j]

            # Outer product for tensor
            for j in range(3):
                for k in range(3):
                    tensor[j, k] += m_particle * v_fluc[j] * v_fluc[k] * w

        for j in range(3):
            for k in range(3):
                KineticTensor[g, j, k] = tensor[j, k]

    return KineticTensor

# Dispatch function for polydisperse vs monodisperse functions =======================================
def CG_Scalar(weights, visibility, grid_indices, Data, Data_scale, Phase, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        return CG_Scalar_Monodisperse(weights, visibility, grid_indices, Data, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        return CG_Scalar_Polydisperse(weights, visibility, grid_indices, Data, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")  
def CG_Vector(weights, visibility, grid_indices, Data, Data_scale, Phase, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        return CG_Vector_Monodisperse(weights, visibility, grid_indices, Data, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        return CG_Vector_Polydisperse(weights, visibility, grid_indices, Data, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")
def CG_Tensor(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        return CG_Tensor_Monodisperse(weights, visibility, grid_indices, Data1, Data2, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        return CG_Tensor_Polydisperse(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")
def CG_KineticTensor(weights, visibility, grid_indices, Displacement,
                                Particle_Velocity, Particle_Mass, 
                                Velocity_Field, Velocity_Field_Gradient, 
                                phase_array, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        return KineticTensor_vInterpolated_Monodisperse(weights, visibility, grid_indices, Displacement,
                                             Particle_Velocity, Particle_Mass, 
                                             Velocity_Field, Velocity_Field_Gradient)
    elif cg_calc_mode == 'Polydisperse':
        return KineticTensor_vInterpolated_Polydisperse(weights, visibility, grid_indices, Displacement,
                                Particle_Velocity, Particle_Mass, 
                                Velocity_Field, Velocity_Field_Gradient, 
                                phase_array)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")
# ============================================================================================ #

# FUNCTIONS THAT ARE CALCULATED ALWAYS BULK 
@njit
def CG_Weighted_Mean_Grainsize(weights, visibility, grid_indices, Data, n_flag):
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
def CG_Scalar_1D_from_same_scalar(weights, visibility, grid_indices, Data):
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