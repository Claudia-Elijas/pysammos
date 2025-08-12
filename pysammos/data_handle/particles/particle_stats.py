"""
Particle Statistics Module
===================

Functions
--------
- d50_calc: Computes the D50 (median diameter) from sorted particle diameters and their corresponding masses.

"""

# import necessary libraries
import numpy as np


def d50_calc(diameter_t0_sort:np.ndarray, mass_t0_sort:np.ndarray) -> float:
    r"""
    Calculate the D50 (median diameter) from sorted particle diameters and their corresponding masses.
    The D50 is the diameter at which 50% of the total mass is contained in particles smaller than this diameter.
    The mathematical formula is given by:
    .. math::
        D_{50} = \frac{d_{i-1} + (0.5 - F_{i-1}) \cdot (d_i - d_{i-1})}{F_i - F_{i-1}}
    where :math:`F_i` is the cumulative mass fraction at diameter :math:`d_i`, 
    and :math:`F_{i-1}` is the cumulative mass fraction at diameter :math:`d_{i-1}`.
    
    Parameters
    ----------
    diameter_t0_sort : ndarray, shape (N,).
        Sorted array of particle diameters.
    mass_t0_sort : ndarray, shape (N,).
        Sorted array of particle masses corresponding to the diameters.
    Returns
    -------
    d50 : float
        The D50 diameter value.
   
    
    Examples
    --------
    >>> diameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> masses = np.array([1, 2, 3, 4, 5])
    >>> d50 = d50_calc(diameters, masses)
    >>> print(d50)
    0.3 


    """
    
    total_mass = np.sum(mass_t0_sort) # Calculate total mass
    cumulative_mass = np.cumsum(mass_t0_sort) # Calculate cumulative mass
    cumulative_fraction = cumulative_mass / total_mass # Normalize cumulative mass to get fraction
    idx = np.searchsorted(cumulative_fraction, 0.5) # Find where cumulative mass crosses 0.5 (D50)
    if idx == 0: # If the first index is 0, return the first diameter
        d50 = diameter_t0_sort[0] # Handle edge cases
    else: # Linear interpolation
        x0, x1 = diameter_t0_sort[idx - 1], diameter_t0_sort[idx] # Diameters at indices idx-1 and idx
        y0, y1 = cumulative_fraction[idx - 1], cumulative_fraction[idx] # Cumulative fractions at those indices
        d50 = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) # Interpolate to find D50
    return d50