import numpy as np

def d50_calc(diameter_t0_sort, mass_t0_sort):
    
    total_mass = np.sum(mass_t0_sort)
    cumulative_mass = np.cumsum(mass_t0_sort)
    cumulative_fraction = cumulative_mass / total_mass
    idx = np.searchsorted(cumulative_fraction, 0.5) # find where cumulative mass crosses 0.5 (D50)
    if idx == 0:
        d50 = diameter_t0_sort[0] # Handle edge cases
    else: # Linear interpolation
        x0, x1 = diameter_t0_sort[idx - 1], diameter_t0_sort[idx]
        y0, y1 = cumulative_fraction[idx - 1], cumulative_fraction[idx]
        d50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return d50