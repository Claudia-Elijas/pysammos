import numpy as np
from numba import njit, prange

# ####################################################################################

# BULK VECTOR GRADIENT (with physical units)
def Compute_VectorBulk_Gradient(Velocity_bulk_CG, nodes, spacing):
    """
    Compute the gradient tensor of a vector field on a regular grid for 1D, 2D, or 3D.
    Velocity_bulk_CG: (N, 3) array (flattened grid, 3 components)
    nodes: tuple of grid shape (nx, [ny, [nz]])
    dx, dy, dz: grid spacings (defaults to 1.0)
    Returns:
        GradTen: (N, 3, ndim) array, where ndim is the number of spatial dimensions
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
def Compute_ShearRate_Tensor(VelocityGrad):
    """
    Calculate the shear rate tensor from the velocity gradient tensor.
    Args:
        VelocityGrad: (Npoints, Nphases, 3, 3) or (Npoints, 3, 3) array
    Returns:
        ShearRateTensor: (Npoints, Nphases, 3, 3) or (Npoints, 3, 3) array
    """
    # case for BULK property only
    Npoints, dims, _ = VelocityGrad.shape
    ShearRateTensor = np.zeros((Npoints, dims, dims))
    for g in range(Npoints):
        ShearRateTensor[g, :, :] = 0.5 * (VelocityGrad[g, :, :] + VelocityGrad[g, :, :].T)
        
    return ShearRateTensor

# TENSOR DEVIATOR 
@njit
def Compute_Deviatoric_Tensor(Tensor): 
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
def Compute_Second_Invariant(Tensor, factor=0.5):
    """
    Compute the second invariant for a batch of 2x2 or 3x3 deviatoric stress tensors.
    Args:
        dev_stress_tensor: (Npoints, 3, 3) or (Npoints, 2, 2) array
    Returns:
        second_invariant: (Npoints,) array
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
def Compute_Pressure(StressTensor):
    """
    Calculate the pressure from the stress tensor.
    Args:
        StressTensor: (Npoints, Nphases, 3, 3) or (Npoints, 3, 3) array
    Returns:
        Pressure: (Npoints, Nphases) or (Npoints,) array
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
def Compute_InertialNumber(shear_rate_tens_dev_mag, pressure, particle_density_mix, particle_diameter_mix, density_phases, diameter_phases):
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
def Compute_Granular_Temperature(DensityMixture, KineticTensor):
    """
    Calculate the granular temperature from the density mixture and kinetic tensor.
    Args:
        DensityMixture: (Npoints, Nphases) or (Npoints,) array
        KineticTensor: (Npoints, Nphases, 3, 3) or (Npoints, 3, 3) array
    Returns:
        GranularTemperature: (Npoints, Nphases) or (Npoints,) array
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
def Compute_Bulk_FabricTensor_Sun2015(normalised_center_to_center_vectors):
    l = normalised_center_to_center_vectors
    Nc = l.shape[0]

    # Compute the outer product for each vector, subtract (1/3)*I, and sum them
    A = np.zeros((3, 3))
    I = np.eye(3)

    for i in prange(Nc):
        A += np.outer(l[i], l[i]) - (1.0 / 3.0) * I

    A /= Nc

    return A