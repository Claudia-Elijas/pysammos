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
from spatial_weights.utils import integration_scalar, trapezoidal_integration, compute_dist_along_branch
# particle-node correspondence
from neighbour_search.grid_particle_search import particle_node_match, calc_displacement
# grid generation
from grid_generation import regular_cuboid 
# handle data
from data_handle.contacts.particle_mapper import map_contact_data
from data_handle.contacts.qualitycheck import duplicates
from data_handle.particles.particle_stats import d50_calc 
# coordination number
from data_handle.contacts.complete import coordination_number
# computing fields
from macroscopic_fields.gridded import dispatcher 
from macroscopic_fields.gridded import secondary
from macroscopic_fields.gridded import scalars
import macroscopic_fields.sliced.granular_temperature as sliced
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
        if not os.path.exists(self.output_path):
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
        GlobalID_t0 = get_point_data_variable(self.DEM_keymap["Global_ID"], poly_out_t0)
        
        return self.BoundsData_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0
    
    def get_particle_size_statistics(self, diameter_t0, mass_t0):
        
        # Sort the diameter and mass arrays based on the diameter
        sorted_indices = np.argsort(diameter_t0)
        diameter_t0_sort = diameter_t0[sorted_indices]
        mass_t0_sort = mass_t0[sorted_indices] 
        sizes , counts = np.unique(diameter_t0_sort, return_counts=True) # count unique particle sizes

        # Calculate the diameter statistics
        self.dmax = np.max(sizes) # calculate maximum particle size
        self.d43 = np.sum(counts*sizes**4)/np.sum(counts*sizes**3) # calculate volume-weighted mean diameter
        self.d32 = np.sum(counts*sizes**3)/np.sum(counts*sizes**2) # calculate surface-weighted mean diameter
        self.drms = np.sqrt(np.sum(counts*sizes**2)/np.sum(counts)) # calculate root mean square diameter        
        self.d50 = d50_calc(diameter_t0_sort, mass_t0_sort) # calculate median particle size

        return self.d43, self.d32
    
    def get_particle_phases(self, diameter_t0, density_t0, global_id, n_max_phases = 6, plot=True):
     
        # IGNORE phases
        if self.ignore_phases:
            self.phases = np.array([[self.d50,density_t0.mean()]]) # use mean density and d50 as phase
            print("Ignoring phases. Using mean density and d50 as phase.")
            self.Phase_Array = None
            self.cg_calc_mode = "Monodisperse"
        
        # find different phases via clustering
        else: 
            self.phases, phase_array = find_phases(diameter_t0, density_t0, n_max_phases) # find the phases and phase array
            print(f"-------- Number of phases found: {len(self.phases)}")
            if plot:
                plot_phases(diameter_t0, density_t0, self.phases, phase_array)
            # particle phase array 
            if len(self.phases) == 1: # if only one phase is found
                self.cg_calc_mode = "Monodisperse"
                self.Phase_Array = None
            elif len(self.phases) > 1: # if multiple phases are found
                self.cg_calc_mode = "Polydisperse"
                self.Phase_Array = phase_array[np.argsort(global_id)] # sort the phase array by global ID

    def set_resolution(self, average_diameter, w_mult = 0.75):
        
        self.w = calc_half_width(average_diameter, w_mult) # calculate the half width
        self.c = calc_cutoff(self.w, self.weight_function) # calculate the cutoff distance

    def generate_grid(self, smoothing_length):

        """Generate the CG grid based on the provided grid information."""
        print("generating grid")
        # generate the grid
        self.GridPoints, self.Nodes, self.Spacing, self.Ranges = regular_cuboid.Grid_Generation(
                                                                        smoothing_length=smoothing_length, 
                                                                        particle_bounds=self.BoundsData_t0, 
                                                                        grid_dimensions=self.grid_info["grid_dimension"], 
                                                                        grid_axes=self.grid_info["grid_axes"],
                                                                        max_particle_diameter=self.dmax,
                                                                        automatic_range=self.grid_info["automatic_grid"],
                                                                        custom_grid_range=[self.grid_info["x_min"], self.grid_info["x_max"],
                                                                                        self.grid_info["y_min"], self.grid_info["y_max"],
                                                                                        self.grid_info["z_min"], self.grid_info["z_max"]],
                                                                        custom_grid_transects=[self.grid_info["x_transect"], 
                                                                                               self.grid_info["y_transect"], 
                                                                                               self.grid_info["z_transect"]]).Generate()
    
    def fields_in_time(self): 
                                                                 
        print("-------------------- Calculating Coarse Grained Fields --------------------")
        # time loop
        for t in range(len(self.time_steps)):
            time_of_timestep = self.time_steps[t]
            time_start = time.time()
            # ========================================================================
            data = self._load_data(time_of_timestep) # Read the data for the current time step
            results = self._fields_single_time(data) # Calculate the CG fields for that time step
            self._write_results(results, t, time_of_timestep) # Write the results to .h5 and .VTKHDF files
            # ========================================================================
            time_end = time.time()
            print(f"Timestep {t} took {time_end - time_start} to run...")
            pass
    
    def _load_data(self, time_of_timestep):
        """Load particle and contact data for a given timestep."""
        print(f"Loading data for timestep {time_of_timestep}...")
        # Load particle data ==========================================================
        PD = file_read.reader(self.file_type, self.particle_path + f"{time_of_timestep:04d}.vtp") 
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
        CD = file_read.reader("vtp", self.contacts_path + f"{time_of_timestep:04d}.vtp")  
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
            ModelAxesRanges=Ranges_t, AxesPeriodicity=[self.grid_info["x_axis_periodic"], 
                                                      self.grid_info["y_axis_periodic"], 
                                                      self.grid_info["z_axis_periodic"]],
            Return_Volume=True, Particle_Phase_Array_t=self.Phase_Array ) 
        print("Contact data loaded and mapped")
        # calculate coordination number
        if "coordination_number" in self.fields_to_compute:
            if Coordination_Number is None:
                print("Coordination number not provided. Calculating it.")
                Coordination_Number, _ = coordination_number.count(np.concatenate((Particle_i.astype(np.int64), Particle_j.astype(np.int64))),Global_ID.astype(np.int64))
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

    def _fields_single_time(self, data):
        
        # 1. assign particles to grid nodes
        particle_to_grid_map = self._assign_particles_to_grid_nodes(data) ; print("... particles assigned to grid nodes")
        # 2. compute weights for particles and contacts
        weights_particle, weights_integral = self._compute_weights(particle_to_grid_map, data); print("... weights computed")
        # 3. compute coarse-grained fields needed for export
        cg_fields = self._compute_fields(data, particle_to_grid_map, weights_particle, weights_integral); print("... fields computed")
        
        return cg_fields
    
    def _write_results(self, results_timestep, time_step, time_of_timestep):
        print(f"Writing results for timestep {time_of_timestep}...")
        # write .h5 files 
        manager = H5XarrayManager(f"{self.output_path}CG_{self.weight_function}_{self.cg_calc_mode}.h5") 
        manager.add_positions(self.GridPoints)
        if self.cg_calc_mode == "Polydisperse":
            phase_labels = ["Bulk"] + [f"Phase_{p}" for p in self.phases]
            manager.add_phases(phase_labels)
        manager.update_h5py_file(results_timestep, dim_index=time_step, dim_value=time_of_timestep, dim_name="time")
        # write .VTKHDF files 
        writer = VTKHDFWriter(node_dimensions=self.Nodes,  
                        node_spacing=self.Spacing, 
                        origin=self.GridPoints[0,:],
                        path=f"{self.output_path}CG_{self.weight_function}_{self.cg_calc_mode}_{time_of_timestep:04d}")
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

    def _assign_particles_to_grid_nodes(self, data):

        # particle data .........................................
        grid_ind_p, part_ind_p = particle_node_match(self.GridPoints, data["Position"], self.c) # kd-tree function
        r_ri, r_ri_dist = calc_displacement(self.GridPoints, data["Position"], 
                                            grid_ind_p, part_ind_p, 
                                            return_disp=True, return_dist=True) # calculate the displacement and distance
        # contact data ...........................................
        grid_ind_c, part_ind_c = particle_node_match(self.GridPoints, data["Position_i"], self.c) # kd-tree function
        r_ri_c, _ = calc_displacement(self.GridPoints, data["Position_i"],
                                    grid_ind_c, part_ind_c, 
                                    return_disp=True, return_dist=False)
        return {
            "grid_ind_p": grid_ind_p,
            "part_ind_p": part_ind_p,
            "r_ri": r_ri,
            "r_ri_dist": r_ri_dist,
            "grid_ind_c": grid_ind_c,
            "part_ind_c": part_ind_c,
            "r_ri_c": r_ri_c
        }

    def _compute_weights(self, particle_map, data):
        # Select CG kernel
        if self.weight_function == "Gaussian":
            WeightFunc = kernels.gaussian
        elif self.weight_function == "Lucy":
            WeightFunc = kernels.lucy
        elif self.weight_function == "HeavySide":
            WeightFunc = kernels.heavySide
        else:
            raise ValueError("Invalid CG function")

        # Particle weights
        hash_table_p, stepsize_p = make_hash_table(WeightFunc, self.c, sensitivity=1000)
        W_p = hash_table_search(particle_map["r_ri_dist"], hash_table_p, stepsize_p)

        # Contact weights
        s = integration_scalar(0, 1, 10)
        dist_along_branch = compute_dist_along_branch(particle_map["r_ri_c"], s, data["BranchVector_i"], particle_map["part_ind_c"])
        hash_table_c, stepsize_c = make_hash_table(WeightFunc, self.c, sensitivity=1000)
        W_c = hash_table_search(dist_along_branch, hash_table_c, stepsize_c)
        Wint_c = trapezoidal_integration(0, 1, 10, W_c)

        return W_p, Wint_c

    def _compute_fields(self, data, g, W_p, Wint_c):
        # 1. Unpack data
        Position = data["Position"]
        Velocity = data["Velocity"]
        Diameter = data["Diameter"]
        Density = data["Density"]
        Mass = data["Mass"]
        Volume = data["Volume"]
        Coordination_Number = data["Coordination_Number"]
        Phase_Array_p = data["Phase_Array"]
        Position_i = data["Position_i"]
        Force_i = data["Force_i"]
        BranchVector_i = data["BranchVector_i"]
        CenterToCenterVector_LL = data["CenterToCenterVector_LL"]
        Volume_i = data["Volume_i"]
        PhaseArray_i = data["PhaseArray_i"]
        d_inContact_mean = data["d_inContact_mean"]
        r_ri = g["r_ri"]
        grid_ind_p = g["grid_ind_p"]
        part_ind_p = g["part_ind_p"]
        grid_ind_c = g["grid_ind_c"]
        part_ind_c = g["part_ind_c"]   

        # check data types
        # print(f"Data types")
        # print(f"Position: {Position.dtype}, Velocity: {Velocity.dtype}, BranchVector_i: {BranchVector_i.dtype}, Force_i: {Force_i.dtype}") 
        # print(f"Phase_Array_p: {Phase_Array_p.dtype}, PhaseArray_i: {PhaseArray_i.dtype}")
        # print(f"Grid indices: {grid_ind_p.dtype}, Particle indices: {part_ind_p.dtype}, distance r_ri: {r_ri.dtype}")
        # print(f"Weights: {W_p.dtype}, Wint_c: {Wint_c.dtype}")


        # 2. Compute fields based on the fields to compute ..................................................................
        # volume fraction
        if "volume_fraction" in self.fields_to_compute:
            VolumeFraction_CG = dispatcher.scalar(W_p, part_ind_p, grid_ind_p, Volume, None, Phase_Array_p, self.cg_calc_mode)
            print(f"volum fraction dtype {VolumeFraction_CG.dtype}")
            print('volume fraction done')

        # density
        if "density_mixture" in self.fields_to_compute:
            DensityMixture_CG = dispatcher.scalar(W_p, part_ind_p, grid_ind_p, Mass, None, Phase_Array_p, self.cg_calc_mode)
            print('mixture density done')
    
        # velocity
        if "momentum_density" in self.fields_to_compute:
            MomentumDens_CG = dispatcher.vector(W_p, part_ind_p, grid_ind_p, Velocity, Mass,  Phase_Array_p, self.cg_calc_mode)
            print('momentum density done')

        # velocity and kinetic tensor    
        # Check if we have phase information
        if self.cg_calc_mode == 'Monodisperse':  # Monodisperse case
            if "velocity" in self.fields_to_compute:
                Velocity_CG = MomentumDens_CG / DensityMixture_CG[:, np.newaxis] # Velocity CG
            if "velocity_gradient" in self.fields_to_compute:
                GradV_CG = secondary.compute_vector_bulk_gradient(Velocity_CG, self.Nodes, self.Spacing) # Velocity Gradient
            if "kinetic_tensor" in self.fields_to_compute:
                KineticTensor_CG = dispatcher.kinetic_tensor(W_p, part_ind_p, grid_ind_p, r_ri, Velocity, Mass, Velocity_CG, GradV_CG, Phase_Array_p, self.cg_calc_mode) # Kinetic tensor
        else:  # Polydisperse case
            if "velocity" in self.fields_to_compute:
                Velocity_CG = MomentumDens_CG / DensityMixture_CG[..., np.newaxis] # Velocity CG
                print('velocity done')
            if "velocity_gradient" in self.fields_to_compute:
                GradV_CG = secondary.compute_vector_bulk_gradient(Velocity_CG[:,0,:], self.Nodes, self.Spacing) # Velocity Gradient
                print('velocity gradient done')
            if "kinetic_tensor" in self.fields_to_compute:
                KineticTensor_CG = dispatcher.kinetic_tensor(W_p, part_ind_p, grid_ind_p, r_ri, Velocity, Mass, Velocity_CG[:, 0, :], GradV_CG, Phase_Array_p, self.cg_calc_mode) # Kinetic tensor
                print('kinetic tensor done')
        # contact tensor
        if "contact_tensor" in self.fields_to_compute:
            ContactTensor_CG = dispatcher.tensor(Wint_c, part_ind_c, grid_ind_c, Force_i, BranchVector_i, None, PhaseArray_i, self.cg_calc_mode)
            print('contact tensor done')
    
        # Density of particle cg
        if "density_particle" in self.fields_to_compute:
            DensityParticle_CG = DensityMixture_CG / VolumeFraction_CG
            print('particle density done')

        # Total stress tensor
        if "total_stress_tensor" in self.fields_to_compute:
            TotalStressTensor_CG = KineticTensor_CG + ContactTensor_CG
            TotalStressTensor_CG_xyz_mag = secondary.compute_second_invariant(TotalStressTensor_CG, factor=0.5) # calculate the second invariant of the total stress tensor
            # deviatoric stress
            TotalStressDeviator_xyz = secondary.compute_deviatoric_tensor(TotalStressTensor_CG[...,:, :]) # 3D
            TotalStressDeviator_xy = secondary.compute_deviatoric_tensor(TotalStressTensor_CG[...,:2,:2]) # 2D
            TotalStressDeviator_xyz_mag = secondary.compute_second_invariant(TotalStressDeviator_xyz, factor=0.5) # 3D mag
            TotalStressDeviator_xy_mag = secondary.compute_second_invariant(TotalStressDeviator_xy, factor=0.5) # 2D mag
            print('total stress done')

        # Volume-weighted mean diameter, d43
        if "d43" in self.fields_to_compute:
            d43_CG = scalars.mean_grainsize(W_p, part_ind_p, grid_ind_p, Diameter, n_flag=3)
            print('d43 done')
        if "d32" in self.fields_to_compute:
            d32_CG = scalars.mean_grainsize(W_p, part_ind_p, grid_ind_p, Diameter, n_flag=2)
            print('d32 done')
  
        # l_CG = None
        # Coordination number
        if "coordination_number" in self.fields_to_compute:
            CoordinationNumber_rattlers = scalars.scalar_x_volume(W_p, part_ind_p, grid_ind_p, Coordination_Number)
            print('Z done')
        # Pressure
        if "pressure" in self.fields_to_compute:
            Pressure_xyz = secondary.compute_pressure(TotalStressTensor_CG)
            Pressure_xy = secondary.compute_pressure(TotalStressTensor_CG[...,0:2, 0:2])
            Pressure_x = TotalStressTensor_CG[...,0, 0] 
            Pressure_y = TotalStressTensor_CG[...,1, 1] 
            Pressure_z = TotalStressTensor_CG[...,2, 2] 
            print('pressure done')

        # granular temperature
        if "granular_temperature" in self.fields_to_compute:
            GranularTemperature_xyz = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG) 
            GranularTemperature_x = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG[...,0,0]) 
            GranularTemperature_y = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG[...,1,1]) 
            GranularTemperature_z = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG[...,2,2]) 
            print('granular temp done')
        if "granular_temperature_alternatives" in self.fields_to_compute:
            GranularTemperature_KimKamrin20, GranularTemperature_LAMMPS = sliced.granular_temperature(dy=self.Spacing[1], 
                                                                                                    y0=self.GridPoints[:,1].min(),
                                                                                                    y1=self.GridPoints[:,1].max(),
                                                                                                    velocity_all=Velocity, 
                                                                                                    diam_all=Diameter, 
                                                                                                    density_all=Density, 
                                                                                                    mass_all=Mass, 
                                                                                                    particle_positions_all=Position)
    
        # shear rate tensor 
        if "shear_rate_tensor" in self.fields_to_compute:
            # 2d - grid
            ShearRateTensor_xy = secondary.compute_shear_rate_tensor(GradV_CG[...,0:2, 0:2])
            ShearRateTensor_xy_mag = secondary.compute_second_invariant(ShearRateTensor_xy, factor=2) # 2D mag
            # shear rate deviator 
            ShearRateDeviator_xy = secondary.compute_deviatoric_tensor(ShearRateTensor_xy)
            ShearRateDeviator_xy_mag = secondary.compute_second_invariant(ShearRateDeviator_xy, factor=2) # 2D mag
            # 3d - grid
            if self.grid_info["grid_dimension"] == 3:
                ShearRateTensor_xyz = secondary.compute_shear_rate_tensor(GradV_CG)
                ShearRateTensor_xyz_mag = secondary.compute_second_invariant(ShearRateTensor_xyz, factor=2) # 3D mag
                # shear rate deviator 
                ShearRateDeviator_xyz = secondary.compute_deviatoric_tensor(ShearRateTensor_xyz)
                ShearRateDeviator_xyz_mag = secondary.compute_second_invariant(ShearRateDeviator_xyz, factor=2) # 3D mag
            print('shear rate done')
 
        # inertial number
        if "inertial_number" in self.fields_to_compute:
            InertialNumber_xy_Pxyz_d43 = secondary.compute_inertial_number(ShearRateTensor_xy_mag, Pressure_xyz, DensityParticle_CG, d43_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Pxy_d43 = secondary.compute_inertial_number(ShearRateTensor_xy_mag, Pressure_xy, DensityParticle_CG, d43_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Py_d43 = secondary.compute_inertial_number(ShearRateTensor_xy_mag, Pressure_y, DensityParticle_CG, d43_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Pxyz_d32 = secondary.compute_inertial_number(ShearRateTensor_xy_mag, Pressure_xyz, DensityParticle_CG, d32_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Pxy_d32 = secondary.compute_inertial_number(ShearRateTensor_xy_mag, Pressure_xy, DensityParticle_CG, d32_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Py_d32 = secondary.compute_inertial_number(ShearRateTensor_xy_mag, Pressure_y, DensityParticle_CG, d32_CG, self.phases[:,1], self.phases[:,0])
            # InertialNumber_xy_Pxyz_l = None
            # InertialNumbe_xy_Pxy_l = None
            # Inertial_Number_xy_Py_l = None
            print('inertial number done')

        # frictional coefficient
        if "frictional_coefficient" in self.fields_to_compute:
            FrictionalCoefficient_Dxy_Pxyz = TotalStressDeviator_xy_mag / Pressure_xyz
            FrictionalCoefficient_Dxy_Pxy = TotalStressDeviator_xy_mag / Pressure_xy
            FrictionalCoefficient_Dxy_Py = TotalStressDeviator_xy_mag / Pressure_y
            FrictionalCoefficient_Dxyz_Pxyz = TotalStressDeviator_xyz_mag / Pressure_xyz
            FrictionalCoefficient_Dxyz_Pxy = TotalStressDeviator_xyz_mag / Pressure_xy
            FrictionalCoefficient_Dxyz_Py = TotalStressDeviator_xyz_mag / Pressure_y
            FrictionalCoefficient_01_Pxyz = TotalStressTensor_CG[...,0,1] / Pressure_xyz
            FrictionalCoefficient_01_Pxy = TotalStressTensor_CG[...,0,1] / Pressure_xy
            FrictionalCoefficient_01_Py = TotalStressTensor_CG[...,0,1] / Pressure_y
            print('mu done')
    
        # fabric tensor
        if "fabric_tensor" in self.fields_to_compute:
            l_i_mag = np.linalg.norm(CenterToCenterVector_LL, axis=-1) # calculate the magnitude of the branch vector
            l_i_normalised = CenterToCenterVector_LL / l_i_mag[:, np.newaxis]
            l_inContact_mean = np.mean(l_i_mag)
            FabricTensor_Monodisperse_CG = dispatcher.tensor(Wint_c, part_ind_c, grid_ind_c, l_i_normalised, l_i_normalised, Volume_i, None, "Monodisperse")
            #FabricTensor_Sun = Compute_Bulk_FabricTensor_Sun2015(l_i_normalised)
            print('frabric tensor done')

        # 3. Prepare results for export based on fields to export ................................................
        
        # Prepare results dictionary
        results = {}

        # volume fraction
        if self.field_to_export.get("volume_fraction"): results["volume_fraction"] = VolumeFraction_CG

        # density of mixture
        if self.field_to_export.get("density_mixture"): results["density_mixture"] = DensityMixture_CG

        # particle density
        if self.field_to_export.get("density_particle"): results["density_particle"] = DensityParticle_CG

        # particle size
        if self.field_to_export.get("d32"): results["d32"] = d32_CG
        if self.field_to_export.get("d43"): results["d43"] = d43_CG

        # particle contacts
        if self.field_to_export.get("coordination_number"): results["coordination_number"] = CoordinationNumber_rattlers
        if self.field_to_export.get("coordination_number_without_rattlers"): results["coordination_number_without_rattlers"] = None

        # velocity
        if self.field_to_export.get("momentum_density"): results["momentum_density"] = MomentumDens_CG
        if self.field_to_export.get("velocity"): results["velocity"] = Velocity_CG
        if self.field_to_export.get("velocity_gradient"): results["velocity_gradient"] = GradV_CG

        # stress tensor
        if self.field_to_export.get("kinetic_tensor"): results["kinetic_tensor"] = KineticTensor_CG
        if self.field_to_export.get("contact_tensor"): results["contact_tensor"] = ContactTensor_CG
        if self.field_to_export.get("total_stress_tensor"): 
            results["total_stress_tensor_xy_dev"] = TotalStressDeviator_xy
            results["total_stress_tensor_xy_dev_mag"] = TotalStressDeviator_xy_mag
            results["total_stress_tensor_xyz"] = TotalStressTensor_CG
            results["total_stress_tensor_xyz_mag"] = TotalStressTensor_CG_xyz_mag
            results["total_stress_tensor_xyz_dev"] = TotalStressDeviator_xyz
            results["total_stress_tensor_xyz_dev_mag"] = TotalStressDeviator_xyz_mag

        # fabric tensor
        if self.field_to_export.get("fabric_tensor"): results["fabric_tensor"] = FabricTensor_Monodisperse_CG
        #if self.field_to_export.get("fabric_tensor_Sun"): results["fabric_tensor_Sun"] = FabricTensor_Sun

        # shear rate tensor
        if self.field_to_export.get("shear_rate_tensor"): 
            results["shear_rate_tensor_xy"] = ShearRateTensor_xy
            results["shear_rate_tensor_xy_mag"] = ShearRateTensor_xy_mag
            results["shear_rate_tensor_xy_dev"] = ShearRateDeviator_xy
            results["shear_rate_tensor_xy_dev_mag"] = ShearRateDeviator_xy_mag
            if self.grid_info["grid_dimension"] == 3:
                results["shear_rate_tensor_xyz"] = ShearRateTensor_xyz
                results["shear_rate_tensor_xyz_mag"] = ShearRateTensor_xyz_mag
                results["shear_rate_tensor_xyz_dev"] = ShearRateDeviator_xyz
                results["shear_rate_tensor_xyz_dev_mag"] = ShearRateDeviator_xyz_mag

        # pressure
        if self.field_to_export.get("pressure"): 
            results["pressure_xyz"] = Pressure_xyz
            results["pressure_xy"] = Pressure_xy
            results["pressure_x"] = Pressure_x
            results["pressure_y"] = Pressure_y
            results["pressure_z"] = Pressure_z

        # granular temperature
        if self.field_to_export.get("granular_temperature"): 
            results["granular_temperature_xyz"] = GranularTemperature_xyz
            results["granular_temperature_x"] = GranularTemperature_x
            results["granular_temperature_y"] = GranularTemperature_y
            results["granular_temperature_z"] = GranularTemperature_z

        # inertial number
        if self.field_to_export.get("inertial_number"): 
            results["inertial_number_Sxy_Pxyz_d43"] = InertialNumber_xy_Pxyz_d43
            results["inertial_number_Sxy_Pxy_d43"] = InertialNumber_xy_Pxy_d43
            results["inertial_number_Sxy_Py_d43"] = InertialNumber_xy_Py_d43
            results["inertial_number_Sxy_Pxyz_d32"] = InertialNumber_xy_Pxyz_d32
            results["inertial_number_Sxy_Pxy_d32"] = InertialNumber_xy_Pxy_d32
            results["inertial_number_Sxy_Py_d32"] = InertialNumber_xy_Py_d32

        # frictional coefficient
        if self.field_to_export.get("frictional_coefficient"): 
            results["frictional_coefficient_Dxy_Pxyz"] = FrictionalCoefficient_Dxy_Pxyz
            results["frictional_coefficient_Dxy_Pxy"] = FrictionalCoefficient_Dxy_Pxy
            results["frictional_coefficient_Dxy_Py"] = FrictionalCoefficient_Dxy_Py
            results["frictional_coefficient_Dxyz_Pxyz"] = FrictionalCoefficient_Dxyz_Pxyz
            results["frictional_coefficient_Dxyz_Pxy"] = FrictionalCoefficient_Dxyz_Pxy
            results["frictional_coefficient_Dxyz_Py"] = FrictionalCoefficient_Dxyz_Py
            results["frictional_coefficient_01_Pxyz"] = FrictionalCoefficient_01_Pxyz
            results["frictional_coefficient_01_Pxy"] = FrictionalCoefficient_01_Pxy
            results["frictional_coefficient_01_Py"] = FrictionalCoefficient_01_Py
       
                    
        print(" results assigned ")
        return results

    
