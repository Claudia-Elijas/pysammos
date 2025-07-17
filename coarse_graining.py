import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy
# subpackage imports
from macroscopic_fields.field_dependencies import get_fields_to_compute




class CoarseGraining: 
    def __init__(self, 
                 particle_path, contacts_path, output_path,
                 start_timestep, end_timestep, dt_time_step,
                 DEM_keymap,
                 grid_info,
                 weight_function, 
                 fields_to_export, 
                 ignore_phases):
        
        # data info
        self.particle_path = particle_path
        self.contacts_path = contacts_path
        self.time_steps = np.arange(start_timestep, end_timestep+1, dt_time_step)#np.array([3, 10, 25, 50, 75, 100, 125, 150, 175])#np.arange(start_timestep, end_timestep+1, dt_time_step)

        # DEM key mapping dictionary
        self.DEM_keymap = DEM_keymap

        # CG grid info dictionary
        self.grid_info = grid_info
           
        # CG smoothing function of choice
        self.weight_function = weight_function

        # CG partial fields ignore
        self.ignore_phases = ignore_phases

        # custom outputs and their dependencies
        self.field_to_export = fields_to_export
        self.fields_to_compute = get_fields_to_compute(fields_to_export)

        # create output path folder 
        self.output_path = output_path
        if not os.path.exists(self.Output_path):
            os.makedirs(self.output_path)
            print(f"Output path created: {self.output_path}")
        else:
            print(f"Output path already exists: {self.output_path}")
          
    
    