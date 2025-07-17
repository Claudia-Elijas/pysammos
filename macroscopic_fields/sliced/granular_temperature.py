import numpy as np
from numba import njit, prange
from .utils import *

@njit
def GranularTemperature_inSlices(dy, y0, y1, velocity_all, diam_all, density_all, mass_all, particle_positions_all):
    rad = diam_all / 2
    position_y = particle_positions_all[:, 1]
    rho_particle = density_all.flatten()
    mi = mass_all

    W = dy
    n = 5
    m = np.arange(1, 2 * n + 2)  # length 11
    bands = np.arange(y0 - W / 2, y1 + W, W)
    Z_k = (bands[:-1] + bands[1:]) / 2
    Z_k_m = np.zeros((Z_k.shape[0], m.shape[0]))
    for j in range(Z_k.shape[0]):
        for i in range(m.shape[0]):
            Z_k_m[j, i] = Z_k[j] - W / 2 + (m[i] - 1) * W / (2 * n)

    half_sublayer_width = W / (2 * n) / 2

    # Precompute weights
    wm_vals = np.zeros(len(m))
    for i in range(len(m)):
        wm_vals[i] = Wm_numba(m[i], n)

    m_WeightedVelocity = np.zeros((len(Z_k), len(m), 3))
    k_WeightedVelocity = np.zeros((len(Z_k), 3))
    velocity_fluctuation = np.zeros_like(velocity_all)
    m_WeightedT_N = np.zeros_like(m_WeightedVelocity)
    k_WeightedT_N = np.zeros((len(Z_k), 3))
    m_WeightedT_LAMMPS = np.zeros_like(m_WeightedVelocity)
    k_WeightedT_LAMMPS = np.zeros((len(Z_k), 3))

    # Velocity loop
    for j in range(len(Z_k)):
        for i in range(len(m)):
            Zm_low = Z_k_m[j, i] - half_sublayer_width
            Zm_high = Z_k_m[j, i] + half_sublayer_width
            ps = ParticlesInBand_numba(position_y, rad, Zm_low, Zm_high)
            if ps.shape[0] == 0:
                continue
            acc = np.zeros(3)
            total_weight = 0.0
            for idx in ps:
                dist = np.abs(Z_k_m[j, i] - position_y[idx])
                a = Area_numba(rad[idx], dist)
                rho = rho_particle[idx]
                ar = a * rho
                acc += ar * velocity_all[idx]
                total_weight += ar
            if total_weight > 0:
                m_WeightedVelocity[j, i, :] = acc / total_weight
        # Band average velocity
        for d in range(3):
            k_WeightedVelocity[j, d] = np.sum(wm_vals * m_WeightedVelocity[j, :, d]) / np.sum(wm_vals)

    # Particle fluctuation
    Zkm_flat = Z_k_m.flatten()
    interp_CGvel = np.zeros_like(velocity_all)
    for p in range(position_y.shape[0]):
        best_idx = 0
        best_dist = np.abs(position_y[p] - Zkm_flat[0])
        for k in range(1, Zkm_flat.shape[0]):
            d = np.abs(position_y[p] - Zkm_flat[k])
            if d < best_dist:
                best_dist = d
                best_idx = k
        interp_CGvel[p, :] = m_WeightedVelocity.reshape(-1, 3)[best_idx, :]
        velocity_fluctuation[p, :] = velocity_all[p, :] - interp_CGvel[p, :]

    # Temperature loop
    for j in prange(len(Z_k)):
        for i in range(len(m)):
            Zm_low = Z_k_m[j, i] - half_sublayer_width
            Zm_high = Z_k_m[j, i] + half_sublayer_width
            ps = ParticlesInBand_numba(position_y, rad, Zm_low, Zm_high)
            N = ps.shape[0]
            if N == 0:
                continue
            avg_mass = np.mean(mi[ps])
            KE = np.zeros(3)
            area_rho_sum = 0.0
            for idx in ps:
                v_fluc = velocity_fluctuation[idx]
                KE += mi[idx] * v_fluc ** 2
            m_WeightedT_LAMMPS[j, i, :] = np.sum(KE) / (3 * N * avg_mass)
            # KAMRIN temperature
            acc = np.zeros(3)
            total_weight = 0.0
            for idx in ps:
                dist = np.abs(Z_k_m[j, i] - position_y[idx])
                a = Area_numba(rad[idx], dist)
                rho = rho_particle[idx]
                ar = a * rho
                acc += ar * velocity_fluctuation[idx] ** 2
                total_weight += ar
            if total_weight > 0:
                m_WeightedT_N[j, i, :] = acc / total_weight
        # Band averages
        for d in range(3):
            k_WeightedT_LAMMPS[j, d] = np.sum(m_WeightedT_LAMMPS[j, :, d]) / len(m)
            k_WeightedT_N[j, d] = np.sum(wm_vals * m_WeightedT_N[j, :, d]) / np.sum(wm_vals)

    return k_WeightedT_N, k_WeightedT_LAMMPS


