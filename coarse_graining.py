import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy

# subpackage imports ----------------------------------------------
# specify what fields to compute
from macroscopic_fields.field_dependencies import get_fields_to_compute
# data reading 
from data_read.mfix import cell_data, point_data, file_read
from data_read.mfix.utils import get_bounds, get_point_data_variable
# phase finding
from particle_phase.clustering import find_phases, plot_phases
# weights
from spatial_weights import kernels
from spatial_weights.resolution import calc_half_width, calc_cutoff
from spatial_weights.hashtable_search import make_hash_table, hash_table_search



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
    
    def get_particle_size_statistics(self, diameter_t0, mass_t0):
        

        # Sort the diameter and mass arrays based on the diameter
        sorted_indices = np.argsort(diameter_t0)
        diameter_t0_sort = diameter_t0[sorted_indices]
        mass_t0_sort = mass_t0[sorted_indices] 

        # Calculate the diameter statistics
        sizes , counts = np.unique(diameter_t0_sort, return_counts=True) # count unique particle sizes
        self.dmax = np.max(sizes) # calculate maximum particle size
        self.d43 = np.sum(counts*sizes**4)/np.sum(counts*sizes**3) # calculate volume-weighted mean diameter
        self.d32 = np.sum(counts*sizes**3)/np.sum(counts*sizes**2) # calculate surface-weighted mean diameter
        self.drms = np.sqrt(np.sum(counts*sizes**2)/np.sum(counts)) # calculate root mean square diameter
        
        # calculate d50 mass-weighted diameter
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
        self.d50 = d50 # calculate median particle size

        return self.d43, self.d32
    
    def get_particle_phases(self, diameter_t0, density_t0, global_id, n_max_phases = 6, plot=True):
     
        # ignore phases
        if self.ignore_phases:
            self.phases = np.array([[self.d50,self.density_t0.mean()]]) # use mean density and d50 as phase
            print("Ignoring phases. Using mean density and d50 as phase.")
            self.Phase_Array = None
            self.cg_calc_mode = "Monodisperse"
        
        # find different phases 
        else: 
            
            # employ clustering
            self.phases, phase_array = find_phases(diameter_t0, density_t0, n_max_phases) # find the phases and phase array
            print(f"-------- Number of phases found: {len(self.phases)}")
            if plot:
                plot_phases(diameter_t0, density_t0, self.phases, phase_array)

            # polydisperse or monodisperse calcs
            if len(self.phases) == 1: # monodisperse
                print("Input data is Monodisperse ---> using Monodisperse calculation")
                self.cg_calc_mode = "Monodisperse"
            elif len(self.phases) > 1: # polydisperse
                if self.ignore_phases == True: # ignore partial fields
                    print("Input data is Polydisperse. Entered Partial_Fields_Ignore=True ---> using Monodisperse calculation")
                    self.cg_calc_mode = "Monodisperse"
                elif self.ignore_phases == False: # calculate partial fields
                    print("Input data is Polydisperse. Entered Partial_Fields_Ignore=False ---> using Polydisperse calculation")
                    self.cg_calc_mode = "Polydisperse"
                else:
                    raise ValueError("Partial_Fields_Ignore must be True or False")
            
            # particle phase array 
            if self.cg_calc_mode == "Monodisperse":
                self.Phase_Array = None
            elif self.cg_calc_mode == "Polydisperse":
                self.Phase_Array = phase_array[np.argsort(global_id)] # sort the phase array by global ID

    def set_resolution(self, average_diameter, w_mult = 0.75):
        
        self.w = calc_half_width(average_diameter, w_mult) # calculate the half width
        self.c = calc_cutoff(self.w, self.weight_function) # calculate the cutoff distance

    
          
    
    