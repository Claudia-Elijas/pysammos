import numpy as np
from numba import njit

@njit
def Wm_numba(m, n):
    return 0.5 * (1 + np.cos(np.pi * (m - (n + 1)) / n))

@njit
def Area_numba(particle_radius, distance_to_plane):
    r = np.sqrt(np.maximum(particle_radius ** 2 - distance_to_plane ** 2, 0.0))
    return np.pi * r ** 2

@njit
def ParticlesInBand_numba(y_particle_pos, particle_radius, lower_plane, upper_plane):
    n_particles = y_particle_pos.shape[0]
    result = []
    for i in range(n_particles):
        if (y_particle_pos[i] + particle_radius[i] >= lower_plane) and (y_particle_pos[i] - particle_radius[i] <= upper_plane):
            result.append(i)
    return np.array(result, dtype=np.int64)
