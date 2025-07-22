from .scalars import (scalar_polydisperse, scalar_polydisperse_scaled, scalar_monodisperse, scalar_monodisperse_scaled) 
from .vectors import (vector_polydisperse, vector_polydisperse_scaled, vector_monodisperse, vector_monodisperse_scaled)  
from .tensors import (tensor_polydisperse, tensor_polydisperse_scaled, tensor_monodisperse, tensor_monodisperse_scaled, 
                      kinetic_tensor_interpolation_polydisperse, kinetic_tensor_interpolation_monodisperse)


# generic scalar
def scalar(weights, visibility, grid_indices, Data, Data_scale, Phase, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        if Data_scale is None:
            return scalar_monodisperse(weights, visibility, grid_indices, Data)
        else:
            return scalar_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        if Data_scale is None:
            return scalar_polydisperse(weights, visibility, grid_indices, Data, Phase)
        else:
            return scalar_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")  

# generic vector
def vector(weights, visibility, grid_indices, Data, Data_scale, Phase, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        if Data_scale is None:
            return vector_monodisperse(weights, visibility, grid_indices, Data)
        else:
            return vector_monodisperse_scaled(weights, visibility, grid_indices, Data, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        if Data_scale is None:
            return vector_polydisperse(weights, visibility, grid_indices, Data, Phase)
        else:
            return vector_polydisperse_scaled(weights, visibility, grid_indices, Data, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")

# generic tensor
def tensor(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        if Data_scale is None:
            return tensor_monodisperse(weights, visibility, grid_indices, Data1, Data2)
        else:
            return tensor_monodisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale)
    elif cg_calc_mode == 'Polydisperse':
        if Data_scale is None:
            return tensor_polydisperse(weights, visibility, grid_indices, Data1, Data2, Phase)
        else:
            return tensor_polydisperse_scaled(weights, visibility, grid_indices, Data1, Data2, Data_scale, Phase)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")

# kinetic tensor including interpolation
def kinetic_tensor(weights, visibility, grid_indices, Displacement,
                                Particle_Velocity, Particle_Mass, 
                                Velocity_Field, Velocity_Field_Gradient, 
                                phase_array, cg_calc_mode):
    if cg_calc_mode == 'Monodisperse':
        return kinetic_tensor_interpolation_monodisperse(weights, visibility, grid_indices, Displacement,
                                             Particle_Velocity, Particle_Mass, 
                                             Velocity_Field, Velocity_Field_Gradient)
    elif cg_calc_mode == 'Polydisperse':
        return kinetic_tensor_interpolation_polydisperse(weights, visibility, grid_indices, Displacement,
                                Particle_Velocity, Particle_Mass, 
                                Velocity_Field, Velocity_Field_Gradient, 
                                phase_array)
    else:
        raise ValueError(f"Unknown cg_calc_mode: {cg_calc_mode}. Expected 'Monodisperse' or 'Polydisperse'.")
    
