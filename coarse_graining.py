import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy
# subpackage imports
from macroscopic_fields.field_dependencies import get_fields_to_compute
from data_read.mfix import cell_data, point_data, file_read
from data_read.mfix.utils import get_bounds, get_point_data_variable


# Coarse Graining Class
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


    def data_sampling(self):

        
        # READ TIME STEP 0
        path_t0 = self.particle_path+f"{self.time_steps[0]:04d}.vtp" #:04d 
        self.file_type = file_read.get_file_type(path_t0) # detect the file type
        poly_out_t0 = file_read.reader(self.file_type, path_t0).GetOutput() # read the vtp file
        #polydata_t0 = Reader_vtm(self.Particle_path + f"{self.TimeSteps[0]}.vtm").GetOutput() # read the vtm file

        # BOUNDS      
        self.BoundsData_t0 = get_bounds(poly_out_t0).reshape(3,2) ; print(f"particle data bounds {self.BoundsData_t0}")

        # PARTICLE PROPERTIES
        if self.DEM_keymap["Particle_Diameter"] is not None:
            Diameter_t0 = get_point_data_variable(self.DEM_keymap["Particle_Diameter"], poly_out_t0)
        else: 
            Diameter_t0 = get_point_data_variable(self.DEM_keymap["Particle_Radius"], poly_out_t0) * 2
        Density_t0 = get_point_data_variable(self.DEM_keymap["Particle_Density"], poly_out_t0)
        Mass_t0 = get_point_data_variable(self.DEM_keymap["Particle_Mass"], poly_out_t0)
        GlobalID_t0 = get_point_data_variable(self.DEM_keymap["Particle_Global_ID"], poly_out_t0)
        
        return self.BoundsData_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0
          
    
    