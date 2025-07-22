import numpy as np
from numba import njit, prange, int32, float32, float64


#ploydisperse
@njit(float64[:,:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:,:], float32[:], int32[:]), parallel=True)
def tensor_polydisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase):
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3, 3), dtype=np.float64)
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

@njit(float64[:,:,:,:](float64[:], int32[:], int32[:], float32[:,:], float32[:,:], int32[:]), parallel=True)
def tensor_polydisperse(weights, visibility, grid_indices, Data1, Data2, Phase):
    Ngridpoints = len(grid_indices) - 2
    Nphases = np.max(Phase) + 1
    CG_Field = np.zeros((Ngridpoints, Nphases + 1, 3, 3), dtype=np.float64)
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
    return CG_Field

# monodisperse
@njit(float64[:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:,:], float32[:]), parallel=True)
def tensor_monodisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3, 3), dtype=np.float64)  # Only one tensor per grid point
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

@njit(float64[:,:,:](float64[:],int32[:], int32[:], float32[:,:], float32[:,:]), parallel=True)
def tensor_monodisperse(weights, visibility, grid_indices, Data1, Data2):
    Ngridpoints = len(grid_indices) - 2
    CG_Field = np.zeros((Ngridpoints, 3, 3), dtype=np.float64)  # Only one tensor per grid point
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

    return CG_Field

# ---------------- KINETIC TENSOR ----------------
@njit(float64[:,:,:,:](float64[:],int32[:], int32[:], float64[:,:], float32[:,:], float32[:], float64[:,:], float64[:,:,:], int32[:]), parallel=True)
def kinetic_tensor_interpolation_polydisperse(weights, visibility, grid_indices, displacement,
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

# kinetic tensor for monodisperse case
@njit(float64[:,:,:](float64[:],int32[:], int32[:], float64[:,:], float32[:,:], float32[:], float64[:,:], float64[:,:,:]),parallel=True)
def kinetic_tensor_interpolation_monodisperse(weights, visibility, grid_indices, displacement,
                                             Particle_Velocity, Particle_Mass, 
                                             Velocity_Field, Velocity_Field_Gradient):
    Ngridpoints = len(grid_indices) - 2
    KineticTensor = np.zeros((Ngridpoints, 3, 3), dtype=np.float64)  # Only one tensor per grid point

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
