import numpy as np
from numba import njit, prange
from .utils import *

def calc_slices(y0, y1, dy, n=5):
    """
    Calculate the slice positions in the y-direction.

    Inputs
    ------
    y0 : float
        The minimum y position of the domain.
    y1 : float
        The maximum y position of the domain.
    dy : float
        The slice height.
    n : int, optional

    Outputs
    -------
    Z_k : np.ndarray
        The center positions of the slices.
    Z_k_m : np.ndarray
        The sub-slice positions within each slice.
    m : np.ndarray
        The sub-slice indices.
    
    Notes
    -----
    The slices are defined from y0 to y1 with a height of dy. Each slice is further divided into 2n sub-slices for more detailed calculations.
    Each slice center Z_k is calculated as the average of the top and bottom of the slice. The sub-slice positions Z_k_m are calculated based on the sub-slice indices m.

    """
  
    m = np.arange(1, 2 * n + 2)  # length 11
    bands = np.arange(y0 - dy / 2, y1 + dy, dy)
    Z_k = (bands[:-1] + bands[1:]) / 2
    Z_k_m = np.zeros((Z_k.shape[0], m.shape[0]))
    for j in range(Z_k.shape[0]):
        for i in range(m.shape[0]):
            Z_k_m[j, i] = Z_k[j] - dy / 2 + (m[i] - 1) * dy / (2 * n)

    return Z_k, Z_k_m, m

@njit 
def granular_temperature(Z_k, Z_k_m, W, n, m, velocity_all, diam_all, density_all, mass_all, particle_positions_all):
    
    """
    Inputs
    ------
    dy : float
        The slice height.
    y0 : float
        The minimum y position of the domain.
    y1 : float
        The maximum y position of the domain.
    velocity_all : np.ndarray
        The particle velocities.    
    diam_all : np.ndarray
        The particle diameters.
    density_all : np.ndarray
        The particle densities.
    mass_all : np.ndarray
        The particle masses.
    particle_positions_all : np.ndarray
        The particle positions.

    Outputs
    -------
    k_WeightedT_N : np.ndarray
        The granular temperature computed using the Kim & Kamrin (2020) method.
    k_WeightedT_LAMMPS : np.ndarray
        The granular temperature computed using the LAMMPS method.
    
    Notes
    -----
    The granular temperature is computed using two methods:

        1. The Kim & Kamrin (2020) method, which uses a weighted average of the velocity fluctuations.
        2. The LAMMPS method, which uses the average of the squared velocity fluctuations.
    
    .. [1] Zhang & Kamrin (2017), Microscopic Description of the Granular Fluidity Field in Nonlocal Flow Modeling. Phys. Rev. Lett. 118, 058001

    """
    
    rad = diam_all / 2
    position_y = particle_positions_all[:, 1]
    rho_particle = density_all.flatten()
    mi = mass_all

    half_sublayer_width = W / (2 * n) / 2

    # Precompute weights
    wm_vals = np.zeros(len(m))
    for i in range(len(m)):
        wm_vals[i] = Wm(m[i], n)

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
            ps = particles_band(position_y, rad, Zm_low, Zm_high)
            if ps.shape[0] == 0:
                continue
            acc = np.zeros(3)
            total_weight = 0.0
            for idx in ps:
                dist = np.abs(Z_k_m[j, i] - position_y[idx])
                a = area(rad[idx], dist)
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
            ps = particles_band(position_y, rad, Zm_low, Zm_high)
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
                a = area(rad[idx], dist)
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

    return k_WeightedT_N, k_WeightedT_LAMMPS, Z_k


