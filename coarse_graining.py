import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy
import time
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
# handle data
from data_handle.contacts.particle_mapper import map_contact_data
from data_handle.contacts.qualitycheck import duplicates
# coordination number
from data_handle.contacts.complete import coordination_number
# data writing
from data_write.h5.writer import H5XarrayManager
from data_write.vtkhdf.writer import VTKHDFWriter


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
        PD = file_read.reader(self.file_type, self.particle_path + f"{t:04d}.vtp") 
        Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Coordination_Number, bounds_t = point_data.particles(PD,
            Global_ID_string=self.DEM_keymap["Global_ID"],
            Velocity_string=self.DEM_keymap["Particle_Velocity"],
            Diameter_string=self.DEM_keymap["Particle_Diameter"],
            Density_string=self.DEM_keymap["Particle_Density"],
            Volume_string=self.DEM_keymap["Particle_Volume"],
            Mass_string=self.DEM_keymap["Particle_Mass"],
            Radius_string=self.DEM_keymap["Particle_Radius"],
            Coordination_Number_string=self.DEM_keymap["Coordination_Number"])
        print("Particle data loaded") 
        # Load contact data ===========================================================
        CD = file_read.reader("vtp", self.Contacts_path + f"{t:04d}.vtp")  
        Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og = point_data.contacts(CD,                                                               
            Particle_i_string=self.DEM_keymap["Particle_i_ID"],
            Particle_j_string=self.DEM_keymap["Particle_j_ID"],
            Force_ij_string=self.DEM_keymap["Force_ij"],
            Contact_ij_string=self.DEM_keymap["Contact_ij"])
        # Handle contact data
        Bounds_t = np.array(bounds_t).reshape(3,2) ; Ranges_t = Bounds_t[:, 1] - Bounds_t[:, 0] # update the model domain bounds for a robust calc of branch vecot
        Particle_i, Particle_j, F_ij, Contact_ij = duplicates.delete(Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og) # remove duplicates        
        Position_i, Force_i, BranchVector_i, CenterToCenterVector_LL_dup, Volume_i, Phase_Array_i_t, d_inContact_mean = map_contact_data(
            Global_ID, Position, Diameter, Density, Volume, 
            Particle_i, Particle_j, F_ij, Contact_ij,
            ModelAxesRanges=Ranges_t, AxesPeriodicity=self.bound_period,
            Return_Volume=True, Particle_Phase_Array_t=self.Phase_Array ) 
        print("Contact data loaded and mapped")
        # calculate coordination number
        if "coordination_number" in self.fields_to_compute:
            if Coordination_Number is None:
                print("Coordination number not provided. Calculating it.")
                Coordination_Number, _ = coordination_number.count(np.concatenate((Particle_i, Particle_j)),Global_ID)
        # Flush or delete the of contact data
        del Particle_i, Particle_j, F_ij, Contact_ij, Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og 
        # ================================================================================
        return {
            # particle data
            "Position": Position,
            "Velocity": Velocity,
            "Diameter": Diameter,
            "Density": Density,
            "Volume": Volume,
            "Mass": Mass,
            "Phase_Array": self.Phase_Array,
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
    
    def compute_fields(self, c_custom): 
                                                                 
        print("-------------------- Calculating Coarse Grained Fields --------------------")
        
        # cut-off distance of coarse-graining function 
        if c_custom is None:
            if not hasattr(self, 'c'):
                raise AttributeError("self.c is not initialized. Ensure Calc_CG_Grid_Spacing is called before this method.")
            c = self.c
        else:
            c = c_custom
        

        # time loop
        for t in range(len(self.time_steps)):

            real_time = self.time_steps[t]
            print("------------------------------------------------------------")
            time_start = time.time()

            print( "  ");print(f">>> Loading data for time step {real_time}")
            data = self.Data_Loading(real_time) # Load the data for the current time step

            print( "  ");print(f">>> Calculating CG fields for time step {real_time}")
            results_timestep = self._compute_fields_single_timestep(c, data) # Calculate the CG fields for that time step

            print( "  ");print(f">>> Saving CG fields for time step {real_time}") 
            # write .h5 files 
            manager = H5XarrayManager(f"{self.output_path}CG_{self.weight_function}_{self.cg_calc_mode}.h5") 
            manager.add_positions(self.GridPoints)
            if self.cg_calc_mode == "Polydisperse":
                phase_labels = ["Bulk"] + [f"Phase_{p}" for p in self.phases]
                manager.add_phases(phase_labels)
            manager.update_h5py_file(results_timestep, dim_index=t, dim_value=real_time, dim_name="time")
            # write .VTKHDF files 
            writer = VTKHDFWriter(node_dimensions=self.Nodes,  
                         node_spacing=self.Spacing, 
                         origin=self.GridPoints[0,:],
                         path=f"{self.Output_path}CG_{self.weight_function}_{self.cg_calc_mode}_{real_time:04d}")
            if self.cg_calc_mode == "Monodisperse":
                writer.write(data_dict=results_timestep)
            elif self.cg_calc_mode == "Polydisperse": 
                writer.write_polydisperse(data_dict=results_timestep,
                                          n_phases=len(self.phases)+1, 
                                          phase_indepen_field_names=["d32", "d43", 
                                                                     "coordination_number", 
                                                                     "velocity_gradient",
                                                                     "fabric_tensor", 
                                                                     "shear_rate_tensor_xyz", "shear_rate_tensor_xyz_mag",
                                                                     "shear_rate_tensor_xy", "shear_rate_tensor_xy_mag",
                                                                     "shear_rate_tensor_xyz_dev", "shear_rate_tensor_xyz_dev_mag",
                                                                      "shear_rate_tensor_xy_dev","shear_rate_tensor_xy_dev_mag"
                                                                     ])
            time_end = time.time()
            print(f"Timestep {t} took {time_end - time_start} to run...")
              
            pass

    def _compute_fields_single_timestep(self, timestep, data_t):
        
        print(f"Computing CG fields at timestep {timestep}")
        data = self.load_data(timestep) if data_t is None else data_t

        d = self._unpack_data(data)
        g = self._assign_particles_to_grid_nodes(d, self.c)
        W_p, W_c, Wint_c = self._compute_weights(d, g, self.c)
        computed_fields = self._compute_cg_fields(d, g, W_p, W_c, Wint_c)
        results = self._export_fields(computed_fields)
        
        return results
    
    def _unpack_data(self, data):
        return {
            "Position": data["Position"],
            "Velocity": data["Velocity"],
            "Diameter": data["Diameter"],
            "Density": data["Density"],
            "Volume": data["Volume"],
            "Mass": data["Mass"],
            "Coordination_Number": data["Coordination_Number"],
            "Phase_Array": data["Phase_Array"],
            "Position_i": data["Position_i"],
            "Force_i": data["Force_i"],
            "BranchVector_i": data["BranchVector_i"],
            "CenterToCenterVector_LL": data["CenterToCenterVector_LL"],
            "Volume_i": data["Volume_i"],
            "PhaseArray_i": data["PhaseArray_i"],
            "d_inContact_mean": data["d_inContact_mean"]
        }

    def _assign_particles_to_grid_nodes(self, d, c):
        grid_ind_p, part_ind_p = Particle_Node_Correspondance(self.GridPoints, d["Position"], c)
        r_ri, r_ri_dist = Calc_Displacement_and_Distance(self.GridPoints, d["Position"], grid_ind_p, part_ind_p, return_disp=True, return_dist=True)

        grid_ind_c, part_ind_c = Particle_Node_Correspondance(self.GridPoints, d["Position_i"], c)
        r_ri_c, _ = Calc_Displacement_and_Distance(self.GridPoints, d["Position_i"], grid_ind_c, part_ind_c, return_disp=True, return_dist=False)

        return {
            "grid_ind_p": grid_ind_p,
            "part_ind_p": part_ind_p,
            "r_ri": r_ri,
            "r_ri_dist": r_ri_dist,
            "grid_ind_c": grid_ind_c,
            "part_ind_c": part_ind_c,
            "r_ri_c": r_ri_c
        }

    def _compute_weights(self, d, g, c):
        # Select CG kernel
        if self.weight_function == "Gaussian":
            WeightFunc = ComputeGaussianWeight
        elif self.weight_function == "Lucy":
            WeightFunc = ComputeLucyWeight
        elif self.weight_function == "HeavySide":
            WeightFunc = ComputeHeavySideWeight
        else:
            raise ValueError("Invalid CG function")

        # Particle weights
        hash_table_p, stepsize_p = make_hash_table(WeightFunc, c, sensitivity=1000)
        W_p = hash_table_search(g["r_ri_dist"], hash_table_p, stepsize_p)

        # Contact weights
        s = integration_scalar(0, 1, 10)
        dist_along_branch = compute_dist_along_branch_numba(g["r_ri_c"], s, d["BranchVector_i"], g["part_ind_c"])
        hash_table_c, stepsize_c = make_hash_table(WeightFunc, c, sensitivity=1000)
        W_c = hash_table_search(dist_along_branch, hash_table_c, stepsize_c)
        Wint_c = trapezoidal_integration(0, 1, 10, W_c)

        return W_p, W_c, Wint_c

    def _compute_cg_fields(self, d, g, W_p, W_c, Wint_c):
        results = {}

        if "volume_fraction" in self.fields_to_compute:
            results["volume_fraction"] = CG_Scalar(W_p, g["part_ind_p"], d["Volume"])
        
        # Add more fields here as needed

        return results

    def _export_fields(self, computed_fields):
        return {k: computed_fields[k] for k in self.field_to_export if k in computed_fields}

