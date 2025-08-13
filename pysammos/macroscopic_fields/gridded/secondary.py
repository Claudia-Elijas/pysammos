"""
Tensor and Rheological Calculations for Granular Flow
=====================================================

This module provides functions to compute velocity gradients, 
shear rate tensors, deviatoric tensors, tensor invariants, 
pressure, inertial numbers, granular temperature, and bulk fabric tensors 
for both monodisperse and polydisperse particulate systems.

It supports 1D, 2D, and 3D grids and uses `NumPy` for array 
operations and `Numba` for accelerated numerical loops.

Functions
---------
- compute_vector_bulk_gradient : Calculate velocity gradient tensor from bulk velocity.
- compute_shear_rate_tensor : Compute symmetric part of velocity gradient tensor.
- compute_deviatoric_tensor : Compute deviatoric part of a stress tensor.
- compute_second_invariant : Compute the second invariant of a deviatoric stress tensor.
- compute_pressure : Compute pressure from a stress tensor.
- compute_inertial_number : Compute dimensionless inertial number.
- compute_granular_temperature : Compute granular temperature following Weinhart (2016).
- bulk_fabric_tensor_Sun2015 : Compute bulk fabric tensor following Sun et al. (2015).

Notes
-----
- All array shapes are explicitly described in the function docstrings.
- Functions decorated with `@njit` are JIT-compiled with Numba for performance.
"""


import numpy as np
from numba import njit, prange

# ####################################################################################

# BULK VECTOR GRADIENT (with physical units)
def compute_vector_bulk_gradient(
    Velocity_bulk_CG: np.ndarray,
    nodes: tuple[int, ...],
    spacing: tuple[float, ...]
) -> np.ndarray:
    """
    Compute the gradient tensor of a bulk velocity field on a regular grid.

    Parameters
    ----------
    Velocity_bulk_CG : (N, 3) ndarray
        Flattened velocity field with 3 components per grid point.
    nodes : tuple of int
        Shape of the grid. Can be (nx,), (nx, ny), or (nx, ny, nz).
    spacing : tuple of float
        Grid spacings along each dimension (dx,), (dx, dy), or (dx, dy, dz).

    Returns
    -------
    GradTen : (N, 3, ndim) ndarray
        Velocity gradient tensor at each grid point, where `ndim` is 
        the number of spatial dimensions (1, 2, or 3).

    Raises
    ------
    ValueError
        If the dimensionality is not 1, 2, or 3.

    Notes
    -----
    - The function handles 1D, 2D, and 3D grids automatically.
    - Output is flattened back to match the input node order.
    """

    dim = len([x for x in nodes if x > 1])
    if dim == 1:
        nx = nodes[0]
        dx = spacing[0]
        V = Velocity_bulk_CG.reshape(nx, 3)
        dUdx = np.gradient(V[:, 0], dx).reshape(-1, 1)
        dVdx = np.gradient(V[:, 1], dx).reshape(-1, 1)
        dWdx = np.gradient(V[:, 2], dx).reshape(-1, 1)
        # Stack into (nx, 3, 1): [component, grad_dir]
        GradTen = np.zeros((nx, 3, 1))
        GradTen[:, 0, 0] = dUdx[:, 0]
        GradTen[:, 1, 0] = dVdx[:, 0]
        GradTen[:, 2, 0] = dWdx[:, 0]
        GradTen = GradTen.reshape(nx, 3, 1)
    elif dim == 2:
        nx, ny = nodes[:2]
        dx, dy = spacing[:2]
        V = Velocity_bulk_CG.reshape(nx, ny, 3)
        # Each returns (nx, ny)
        dUdx, dUdy = np.gradient(V[..., 0], dx, dy, axis=(0, 1))
        dVdx, dVdy = np.gradient(V[..., 1], dx, dy, axis=(0, 1))
        dWdx, dWdy = np.gradient(V[..., 2], dx, dy, axis=(0, 1))
        # Stack into (nx, ny, 3, 2): [component, grad_dir]
        GradTen = np.zeros((nx, ny, 3, 2))
        GradTen[..., 0, 0] = dUdx
        GradTen[..., 0, 1] = dUdy
        GradTen[..., 1, 0] = dVdx
        GradTen[..., 1, 1] = dVdy
        GradTen[..., 2, 0] = dWdx
        GradTen[..., 2, 1] = dWdy
        GradTen = GradTen.reshape(nx * ny, 3, 2)
    elif dim == 3:
        nx, ny, nz = nodes
        dx, dy, dz = spacing
        V = Velocity_bulk_CG.reshape(nx, ny, nz, 3)
        dUdx, dUdy, dUdz = np.gradient(V[..., 0], dx, dy, dz, axis=(0, 1, 2))
        dVdx, dVdy, dVdz = np.gradient(V[..., 1], dx, dy, dz, axis=(0, 1, 2))
        dWdx, dWdy, dWdz = np.gradient(V[..., 2], dx, dy, dz, axis=(0, 1, 2))
        # Stack into (nx, ny, nz, 3, 3): [component, grad_dir]
        GradTen = np.zeros((nx, ny, nz, 3, 3))
        GradTen[..., 0, 0] = dUdx
        GradTen[..., 0, 1] = dUdy
        GradTen[..., 0, 2] = dUdz
        GradTen[..., 1, 0] = dVdx
        GradTen[..., 1, 1] = dVdy
        GradTen[..., 1, 2] = dVdz
        GradTen[..., 2, 0] = dWdx
        GradTen[..., 2, 1] = dWdy
        GradTen[..., 2, 2] = dWdz
        GradTen = GradTen.reshape(nx * ny * nz, 3, 3)
    else:
        raise ValueError("Unsupported dimensionality. Only 1D, 2D, and 3D grids are supported.")
    return GradTen

# SHEAR RATE TENSOR
@njit
def compute_shear_rate_tensor(
    VelocityGrad: np.ndarray
) -> np.ndarray:
    """
    Compute the symmetric shear rate tensor from a velocity gradient tensor.

    Parameters
    ----------
    VelocityGrad : (N, 3, 3) ndarray
        Velocity gradient tensor for N points.

    Returns
    -------
    ShearRateTensor : (N, 3, 3) ndarray
        Symmetric part of the velocity gradient tensor.
    """
    # case for BULK property only
    Npoints, dims, _ = VelocityGrad.shape
    ShearRateTensor = np.zeros((Npoints, dims, dims))
    for g in range(Npoints):
        ShearRateTensor[g, :, :] = 0.5 * (VelocityGrad[g, :, :] + VelocityGrad[g, :, :].T)
        
    return ShearRateTensor

# TENSOR DEVIATOR 
@njit
def compute_deviatoric_tensor(
    Tensor: np.ndarray
) -> np.ndarray:
    """
    Compute the deviatoric tensor by removing the mean normal stress.

    Parameters
    ----------
    Tensor : (N, 3, 3) or (N, P, 3, 3) ndarray
        Stress tensor(s). P is number of phases.

    Returns
    -------
    DeviatoricTensor : ndarray
        Tensor(s) with isotropic part removed. Shape matches input.
    """
    # 1. case for POLYDISPERSE ---------------------------------------
    if Tensor.ndim == 4:
        Npoints, Nphases, dims, _ = Tensor.shape
        DeviatoricTensor = np.zeros((Npoints, Nphases, dims, dims))
        for g in range(Npoints):
            for p in range(Nphases):
                mean_stress = np.trace(Tensor[g, p, :, :]) / dims
                DeviatoricTensor[g, p, :, :] = Tensor[g, p, :, :] - mean_stress * np.eye(dims)
        return DeviatoricTensor
    # 2. case for MONODISPERSE ---------------------------------------
    elif Tensor.ndim == 3:
        Npoints, dims, _ = Tensor.shape
        DeviatoricTensor = np.zeros((Npoints, dims, dims))
        for g in range(Npoints):
            mean_stress = np.trace(Tensor[g, :, :]) / dims
            DeviatoricTensor[g, :, :] = Tensor[g, :, :] - mean_stress * np.eye(dims)
        return DeviatoricTensor

# TENSOR SECOND INVARIANT
@njit
def compute_second_invariant(
    Tensor: np.ndarray,
    factor: float = 0.5
) -> np.ndarray:
    """
    Compute the second invariant of a deviatoric stress tensor.

    Parameters
    ----------
    Tensor : (N, 3, 3), (N, 2, 2), (N, P, 3, 3), or (N, P, 2, 2) ndarray
        Deviatoric stress tensor(s).
    factor : float, optional
        Multiplicative factor in the invariant calculation (default is 0.5).

    Returns
    -------
    second_invariant : (N,) or (N, P) ndarray
        Second invariant magnitude for each tensor.
    """
    # 1. case for POLYDISPERSE ---------------------------------------
    if Tensor.ndim == 4:
        Npoints, Nphases, dims, _ = Tensor.shape
        second_invariant = np.zeros((Npoints, Nphases))
        if dims == 3:
            for i in range(Npoints):
                for j in range(Nphases):
                    tensor = Tensor[i, j]
                    xx = tensor[0, 0]
                    yy = tensor[1, 1]
                    zz = tensor[2, 2]
                    xy = tensor[0, 1]
                    xz = tensor[0, 2]
                    yz = tensor[1, 2]
                    second_invariant[i, j] = np.sqrt(
                        factor * (xx**2 + yy**2 + zz**2 + 2 * (xy**2 + xz**2 + yz**2))
                    )
        elif dims == 2:
            for i in range(Npoints):
                for j in range(Nphases):
                    tensor = Tensor[i, j]
                    xx = tensor[0, 0]
                    yy = tensor[1, 1]
                    xy = tensor[0, 1]
                    second_invariant[i, j] = np.sqrt(
                        factor * (xx**2 + yy**2 + 2 * xy**2)
                    )
        return second_invariant
    # 2. case for MONODISPERSE ---------------------------------------
    elif Tensor.ndim == 3:
        Npoints = Tensor.shape[0]
        dims = Tensor.shape[1] 
        second_invariant = np.zeros(Npoints)
        if dims == 3:
            for i in range(Npoints):
                tensor = Tensor[i]
                xx = tensor[0, 0]
                yy = tensor[1, 1]
                zz = tensor[2, 2]
                xy = tensor[0, 1]
                xz = tensor[0, 2]
                yz = tensor[1, 2]
                second_invariant[i] = np.sqrt(
                    factor * (xx**2 + yy**2 + zz**2 + 2 * (xy**2 + xz**2 + yz**2))
                )
        elif dims == 2:
            for i in range(Npoints):
                tensor = Tensor[i]
                xx = tensor[0, 0]
                yy = tensor[1, 1]
                xy = tensor[0, 1]
                second_invariant[i] = np.sqrt(
                    factor * (xx**2 + yy**2 + 2 * xy**2)
                )
        return second_invariant

# PRESSURE 
@njit
def compute_pressure(
    StressTensor: np.ndarray
) -> np.ndarray:
    """
    Compute pressure from the stress tensor.

    Parameters
    ----------
    StressTensor : (N, 3, 3) or (N, P, 3, 3) ndarray
        Stress tensor(s), where P is the number of phases.

    Returns
    -------
    Pressure : (N,) or (N, P) ndarray
        Computed pressure(s).
    """
    # 1. POLYDISPERSE case
    if StressTensor.ndim == 4:
        Npoints, Nphases, dims, _ = StressTensor.shape
        Pressure = np.zeros((Npoints, Nphases))
        for g in range(Npoints):
            for p in range(Nphases):
                Pressure[g, p] = np.trace(StressTensor[g, p, :, :]) / dims
        return Pressure
    # 2. MONODISPERSE case
    elif StressTensor.ndim == 3:
        Npoints, dims, _ = StressTensor.shape
        Pressure = np.zeros(Npoints)
        for g in range(Npoints):
            Pressure[g] = np.trace(StressTensor[g, :, :]) / dims
        return Pressure

# INERTIAL NUMBER
@njit
def compute_inertial_number(
    shear_rate_tens_dev_mag: np.ndarray,
    pressure: np.ndarray,
    particle_density_mix: np.ndarray,
    particle_diameter_mix: np.ndarray,
    density_phases: np.ndarray,
    diameter_phases: np.ndarray
) -> np.ndarray:
    """
    Compute the inertial number for monodisperse or polydisperse systems.

    Parameters
    ----------
    shear_rate_tens_dev_mag : (N,) ndarray
        Magnitude of the deviatoric shear rate tensor.
    pressure : (N,) or (N, P) ndarray
        Pressure values.
    particle_density_mix : (N,) or (N, P) ndarray
        Bulk particle density.
    particle_diameter_mix : (N,) ndarray
        Bulk particle diameter.
    density_phases : (P-1,) ndarray
        Densities for individual phases (polydisperse case).
    diameter_phases : (P-1,) ndarray
        Diameters for individual phases (polydisperse case).

    Returns
    -------
    Inertial_Number : (N,) or (N, P) ndarray
        Dimensionless inertial number(s).
    """    
    
    Npoints = pressure.shape[0]
    # 1. Monodisperse case
    if pressure.ndim == 1:
        Inertial_Number = np.zeros(Npoints)
        for g in range(Npoints):
            denom = np.sqrt(np.abs(pressure[g]))
            if denom != 0.0:
                Inertial_Number[g] = shear_rate_tens_dev_mag[g] * particle_diameter_mix[g] * np.sqrt(particle_density_mix[g]) / denom
            else:
                Inertial_Number[g] = 0.0
        return Inertial_Number
    # 2. Polydisperse case
    elif pressure.ndim == 2:
        Nphases = pressure.shape[1]
        Inertial_Number = np.zeros((Npoints, Nphases))
        for g in range(Npoints):
            for p in range(Nphases):
                denom = np.sqrt(np.abs(pressure[g, p]))
                if denom != 0.0:
                    if p == 0:
                        Inertial_Number[g, p] = shear_rate_tens_dev_mag[g] * particle_diameter_mix[g] * np.sqrt(particle_density_mix[g, p]) / denom
                    else:
                        Inertial_Number[g, p] = shear_rate_tens_dev_mag[g] * diameter_phases[p-1] * np.sqrt(density_phases[p-1]) / denom
                else:
                    Inertial_Number[g, p] = 0.0
        return Inertial_Number

# GRANULAR TEMPERATURE WEINHART 2016
@njit
def compute_granular_temperature(
    DensityMixture: np.ndarray,
    KineticTensor: np.ndarray
) -> np.ndarray:
    """
    Compute granular temperature following Weinhart (2016).

    Parameters
    ----------
    DensityMixture : (N,) or (N, P) ndarray
        Bulk or phase-wise particle density.
    KineticTensor : (N, 3, 3), (N, P, 3, 3), (N,), or (N, P) ndarray
        Full kinetic energy tensor or scalar kinetic energy component.

    Returns
    -------
    GranularTemperature : (N,) or (N, P) ndarray
        Granular temperature(s).
    """

    # 1. cases for POLYDISPERSE ---------------------------------------
    # a) full kinetic tensor is provided
    if KineticTensor.ndim == 4:
        Npoints, Nphases, dims, _ = KineticTensor.shape
        GranularTemperature = np.zeros((Npoints, Nphases))
        for g in range(Npoints):
            for p in range(Nphases):
                denom = dims * DensityMixture[g, p]
                if denom != 0.0:
                    GranularTemperature[g, p] = np.trace(KineticTensor[g, p, :, :]) / denom
                else:
                    GranularTemperature[g, p] = 0.0
        return GranularTemperature
    # b) 1 direction of the kinetic tensor is provided (i.e., 00 component)
    elif KineticTensor.ndim == 2: 
        Npoints, Nphases = KineticTensor.shape
        GranularTemperature = np.zeros((Npoints, Nphases))
        for g in range(Npoints):
            for p in range(Nphases):
                denom = DensityMixture[g, p]
                if denom != 0.0:
                    GranularTemperature[g, p] = KineticTensor[g, p] / denom
                else:
                    GranularTemperature[g, p] = 0.0
        return GranularTemperature
    # 2. cases for MONODISPERSE ---------------------------------------
    # a) full kinetic tensor is provided
    elif KineticTensor.ndim == 3:
        Npoints, dims, _ = KineticTensor.shape
        GranularTemperature = np.zeros(Npoints)
        for g in range(Npoints):
            #GranularTemperature[g] = np.trace(KineticTensor[g, :, :]) / (dims * DensityMixture[g])
            denom = dims * DensityMixture[g]
            if denom != 0.0:
                GranularTemperature[g] = np.trace(KineticTensor[g, :, :]) / denom
            else:
                GranularTemperature[g] = 0.0

    # b) 1 direction of the kinetic tensor is provided (i.e., 00 component)
    elif KineticTensor.ndim == 1:
        Npoints = KineticTensor.shape[0]
        GranularTemperature = np.zeros(Npoints)
        for g in range(Npoints):
            #GranularTemperature[g] = KineticTensor[g] / DensityMixture[g]
            denom = DensityMixture[g]
            if denom != 0.0:
                GranularTemperature[g] = KineticTensor[g] /  denom
            else:
                GranularTemperature[g] = 0.0
        return GranularTemperature

# BULK FABRIC TENSOR 
@njit
def bulk_fabric_tensor_Sun2015(
    normalised_center_to_center_vectors: np.ndarray
) -> np.ndarray:
    """
    Compute the bulk fabric tensor following Sun et al. (2015).

    Parameters
    ----------
    normalised_center_to_center_vectors : (Nc, 3) ndarray
        Normalised contact vectors between particle centers.

    Returns
    -------
    A : (3, 3) ndarray
        Bulk fabric tensor.
    """
    
    l = normalised_center_to_center_vectors
    Nc = l.shape[0]

    # Compute the outer product for each vector, subtract (1/3)*I, and sum them
    A = np.zeros((3, 3))
    I = np.eye(3)

    for i in prange(Nc):
        A += np.outer(l[i], l[i]) - (1.0 / 3.0) * I

    A /= Nc

    return A