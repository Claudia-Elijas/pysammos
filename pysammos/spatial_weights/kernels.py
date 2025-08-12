"""
Kernel functions module
=======================

This module provides a collection of commonly used kernel functions for
smoothing, weighting, and interpolation in particle-based methods such as
Smoothed Particle Hydrodynamics (SPH) or coarse-graining techniques.

Included kernels:

- Lucy kernel: A smooth, compactly supported kernel with continuous derivatives.
- Top-hat (step) function: A simple binary indicator function.
- Heaviside kernel: Uniform weighting inside a spherical cutoff volume.
- Gaussian kernel: A smooth, bell-shaped kernel with compact support truncated
  at three standard deviations.

Each kernel function computes a weight based on the distance from a kernel center
and a cutoff radius, ensuring locality and normalization properties as appropriate.

These kernels are useful for weighting contributions of particles or points
within a neighborhood and are foundational in various numerical methods
in computational physics, fluid dynamics, and materials science.
"""

from scipy.special import erf
import numpy as np

# Lucy function
def lucy(c,dist):
    r"""Lucy kernel function.

    Computes the Lucy weight at a given distance within the cutoff radius.

    The Lucy kernel is defined as:

    .. math::

        W(r) =
        \begin{cases}
        \dfrac{105}{16\pi c^3} \left(
            -3\left(\dfrac{r}{c}\right)^4
            + 8\left(\dfrac{r}{c}\right)^3
            - 6\left(\dfrac{r}{c}\right)^2
            + 1
        \right) & \text{if } r \leq c \\
        0 & \text{if } r > c
        \end{cases}

    Parameters
    ----------
    c : float
        Cutoff distance.

    dist : float
        Distance between the evaluation point and the kernel center.

    Returns
    -------
    W : float
        The weight computed from the Lucy kernel.
    """
    if dist > c:
        W = 0
    else:
        W = 105/(16*np.pi*c**3) * (-3*(dist/c)**4+8*(dist/c)**3-6*(dist/c)**2+1)
    return W


# top hat function
def h(dis):
    r"""Top-hat (step) function.

    A basic step function defined as:

    .. math::

        h(x) =
        \begin{cases}
        1 & \text{if } x > 0 \\
        0 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    dis : float
        Input value.

    Returns
    -------
    int
        1 if `dis > 0`, otherwise 0.
    """ 
    if dis > 0:
        return 1
    else:
        return 0

# Heavy side function
def heavySide(c, dist):
    r"""Heaviside kernel function.

    Computes the Heaviside (uniform sphere) kernel value for a given distance.

    The function returns:

    .. math::

        H(r) =
        \begin{cases}
        \dfrac{1}{\Omega} & \text{if } r \leq c \\
        0 & \text{otherwise}
        \end{cases}

    where :math:`\Omega = \dfrac{4}{3}\pi c^3` is the normalization volume of a sphere.

    Parameters
    ----------
    c : float
        Cutoff distance (radius of the uniform sphere).

    dist : float
        Distance from the kernel center.

    Returns
    -------
    HS : float
        The Heaviside kernel value.
    """
    # given that c = w
    if dist > c:
        return 0
    else:
        
        Omega = (4/3)*np.pi*c**3 
        HS  = (1/Omega) * h(c - dist)
        return HS


# Gaussian function
def gaussian(c, dist):
    r"""Gaussian kernel function with compact support.

    Computes the Gaussian kernel value used for continuous interpolation or
    smoothing operations. The kernel is truncated at the cutoff distance `c`,
    which corresponds to three standard deviations.

    .. math::

        w = \dfrac{c}{3}

    .. math::

        W(r) =
        \begin{cases}
        \dfrac{1}{V_w} \exp\left(-\dfrac{r^2}{2w^2}\right) & \text{if } r \leq c \\
        0 & \text{if } r > c
        \end{cases}

    where :math:`V_w` is a normalization constant to ensure the kernel integrates to 1:

    .. math::

        V_w = 2\sqrt{2}w^3\pi^{3/2}
        \operatorname{erf}\left(\dfrac{c\sqrt{2}}{2w}\right)
        - 4cw^2\pi \exp\left(-\dfrac{c^2}{2w^2}\right)

    Parameters
    ----------
    c : float
        Cutoff distance (typically `3σ`).

    dist : float
        Distance between the evaluation point and the kernel center.

    Returns
    -------
    W : float
        The Gaussian kernel value.
    """
    
    if dist > c:
        W = 0
    else:
        w = c/3
        Vw = 2 * np.sqrt(2) * w**3 * np.pi**(3/2) * erf((c*np.sqrt(2))/(2*w)) - 4 * c * w**2 * np.pi * np.exp(-(c**2/(2*w**2))) # Matlab-Carrara
        
        #Vw = 2 * math.sqrt(2) * (c/3)**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*(c/3))) - 4 * c * (c/3)**2 * math.pi * math.exp(-(c**2/(2*(c/3)**2))) # Matlab-Carrara
        
        #Vw = math.erf(c/(math.sqrt(2)*w))- math.sqrt(2/math.pi) * (c/w) * math.exp(-(c**2)/(2*w**2)) # Weinhart 2016
        #Vw = (np.sqrt(2 * np.pi) * w)**3 # Tunuguntla 2016
        if Vw == 0:
            W = 0
        else:
            W = (1/Vw) * np.exp(-((dist**2))/(2*w**2))

    return W


