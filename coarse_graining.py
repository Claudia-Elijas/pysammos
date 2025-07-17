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
# grid generation
from grid_generation import regular_cuboid 


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

    def make_grid(self, smoothing_length):

        """Generate the CG grid based on the provided grid information."""
        print("generating grid")
        # generate the grid
        self.GridPoints, self.Nodes, self.Spacing, self.Ranges = regular_cuboid.Grid_Generation(
                                                                        smoothing_length=smoothing_length, 
                                                                        particle_bounds=self.BoundsData_t0, 
                                                                        grid_dimension=self.grid_info["grid_dimension"], 
                                                                        grid_axes=self.grid_info["grid_axes"],
                                                                        automatic_grid=self.grid_info["automatic_grid"],
                                                                        max_particle_size=self.dmax,
                                                                        custom_grid_range=[self.grid_info["x_min"], self.grid_info["x_max"],
                                                                                        self.grid_info["y_min"], self.grid_info["y_max"],
                                                                                        self.grid_info["z_min"], self.grid_info["z_max"]],
                                                                        custom_grid_transects=[self.grid_info["x_transect"], 
                                                                                               self.grid_info["y_transect"], 
                                                                                               self.grid_info["z_transect"]],
                                                                        bound_period=[self.grid_info["x_axis_periodic"],
                                                                                        self.grid_info["y_axis_periodic"],
                                                                                        self.grid_info["z_axis_periodic"]]).Generate()
    
    def load_data(self, t):
        """Load particle and contact data for a given timestep."""

        # Load particle data ==========================================================
        PD = Reader(self.file_type, self.Particle_path + f"{t:04d}.vtp")
        #PD = Reader_vtm(self.Particle_path + f"{t}.vtm") # read vtm data (needed for Polydisperse and MultiSphere)

        Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Radius, Coordination_Number, bounds_t = ParticleData(
            PD,
            Global_ID_string=self.DEM_keymap["Global_ID"],
            Velocity_string=self.DEM_keymap["Particle_Velocity"],
            Diameter_string=self.DEM_keymap["Particle_Diameter"],
            Density_string=self.DEM_keymap["Particle_Density"],
            Volume_string=self.DEM_keymap["Particle_Volume"],
            Mass_string=self.DEM_keymap["Particle_Mass"],
            Radius_string=self.DEM_keymap["Particle_Radius"],
            Coordination_Number_string=self.DEM_keymap["Coordination_Number"],
        )
        print("Particle data loaded") 
        print(f"  {len(Position)} particles")

        # phase array
        Phase_Array_t = self.Phase_Array # [Global_ID - 1] if self.Phase_Array is not None else None# no need to update as each time step we argsort

        # check particle size data is provided 
        if Diameter is None:
            if Radius is None:
                raise ValueError("Diameter or Radius must be provided")
            else:
                Diameter = 2 * Radius

        # update the model domain bounds for a robust calc of branch vecot
        Bounds_t = np.array(bounds_t).reshape(3,2)
        Ranges_t = Ranges_t = Bounds_t[:, 1] - Bounds_t[:, 0]

        # Load contact data ===========================================================
        CD = Reader("vtp", self.Contacts_path + f"{t:04d}.vtp")  
        Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og = ContactData(CD,                                                               
            Particle_i_string=self.DEM_keymap["Particle_i_ID"],
            Particle_j_string=self.DEM_keymap["Particle_j_ID"],
            Force_ij_string=self.DEM_keymap["Force_ij"],
            Contact_ij_string=self.DEM_keymap["Contact_ij"], )
      
        # Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og = ContactData__JP(
        #     CD,
        #     Part_ids_string=self.DEM_keymap["Particle_i_ID"],
        #     Force_ij_string=self.DEM_keymap["Force_ij"],
        #     Contact_ij_string=self.DEM_keymap["Contact_ij"]            
        # )   

        # Handle contact data
        Particle_i, Particle_j, F_ij, Contact_ij = Check_for_Duplicate_Pairs(Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og)
        Position_i, Force_i, BranchVector_i, CenterToCenterVector_LL_dup, Volume_i, Phase_Array_i_t, d_inContact_mean = Arange_ContactData(
            Global_ID, Position, Diameter, Density, Volume, 
            Particle_i, Particle_j, F_ij, Contact_ij,
            ModelAxesRanges=Ranges_t,
            AxesPeriodicity=self.bound_period,
            Return_Volume=True, 
            Particle_Phase_Array_t=Phase_Array_t ) 

        # coordination number calculation
        if "coordination_number" in self.fields_to_compute:
            if Coordination_Number is None:
                print("Coordination number not provided. Calculating it.")
                Particle_i_dup = np.concatenate((Particle_i.astype(np.int64), Particle_j.astype(np.int64)))
                Coordination_Number, Coordination_Number_corrected = Calc_Coordination_Number(Particle_i_dup,
                                                                Global_ID.astype(np.int64))
                print(f"coordination number max , min = {Coordination_Number.max(), Coordination_Number.min(), len(Coordination_Number)}")
            else:
                print("Coordination number provided. Using it, and making a copy with no rattlers.")
                Coordination_Number = Coordination_Number.astype(np.int64)
                #Coordination_Number_corrected = Coordination_Number[Coordination_Number > 1]
        else: 
            Coordination_Number == None

        # Flush or delete the of contact data
        del Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og
        # ================================================================================
        return {

            # particle data
            "Position": Position,
            "Velocity": Velocity,
            "Diameter": Diameter,
            "Density": Density,
            "Volume": Volume,
            "Mass": Mass,
            "Phase_Array": Phase_Array_t,
            "Coordination_Number": Coordination_Number,

            # contact data
            "Position_i": Position_i,
            "Force_i": Force_i,
            "BranchVector_i": BranchVector_i,
            "CenterToCenterVector_LL": CenterToCenterVector_LL_dup,
            "Volume_i": Volume_i,
            "PhaseArray_i": Phase_Array_i_t,
            "d_inContact_mean": d_inContact_mean
        
            } 
    
    