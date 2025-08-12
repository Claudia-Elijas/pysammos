"""
Coarse Graining Module
======================

This module provides functionality for coarse graining particle data from discrete element method (DEM) simulations.
It includes methods for loading particle data, calculating particle size statistics, generating coarse-grained grids, 
computing macroscopic fields, handling particle phases, and writing output data.

Main Class
----------
CoarseGraining
    Encapsulates the coarse graining process, including initialization, configuration, data sampling, phase identification, grid generation, and field computation.

Key Methods
-----------
- data_sampling()
    Loads particle data for the first time step.
- get_particle_size_statistics()
    Calculates particle size statistics such as d43, d32, dmax, drms, and d50.
- get_particle_phases()
    Identifies particle phases based on diameter and density.
- generate_grid()
    Generates a coarse-grained grid based on particle bounds and specified resolution.
- fields_in_time()
    Computes and writes macroscopic fields over specified time steps.
- _load_data()
    Loads particle and contact data for a given time step.
- _fields_single_time()
    Computes coarse-grained fields for a single time step.
- _write_results()
    Writes computed results to .h5 and .VTKHDF files.
- _assign_particles_to_grid_nodes()
    Assigns particles and contacts to grid nodes for coarse graining.
- _compute_weights()
    Computes spatial weights for particles and contacts based on the specified weight function.

Notes
-----
- Designed for extensibility and integration with DEM simulation workflows.
- Output formats include HDF5 and VTKHDF for compatibility with scientific visualization tools.

"""


# import standard libraries ----------------------------------------------
import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy
import time
from typing import Tuple
# subpackage imports ----------------------------------------------
# specify what fields to compute
from .macroscopic_fields.field_dependencies import get_fields_to_compute
# data reading 
from .data_read.mfix import cell_data, point_data, file_read
from .data_read.mfix.utils import get_bounds, get_point_data_variable
# phase finding
from .particle_phase.clustering import find_phases, plot_phases
# weights
from .spatial_weights import kernels
from .spatial_weights.resolution import calc_half_width, calc_cutoff
from .spatial_weights.hashtable_search import make_hash_table, hash_table_search
from .spatial_weights.utils import integration_scalar, trapezoidal_integration, compute_dist_along_branch
# particle-node correspondence
from .neighbour_search.grid_particle_search import particle_node_match, calc_displacement
# grid generation
from .grid_generation import regular_cuboid 
# handle data
from .data_handle.contacts.particle_mapper import map_contact_data
from .data_handle.contacts.qualitycheck import duplicates
from .data_handle.particles.particle_stats import d50_calc 
# coordination number
from .data_handle.contacts.complete import coordination_number
# computing fields
from .macroscopic_fields.gridded import dispatcher 
from .macroscopic_fields.gridded import secondary
from .macroscopic_fields.gridded import scalars
from .macroscopic_fields.sliced import granular_temperature as sliced
# data writing
from .data_write.h5.writer import H5XarrayManager
from .data_write.vtkhdf.writer import VTKHDFWriter



# Coarse Graining Class
class CoarseGraining: 
    def __init__(self, 
                 particle_path:str, contacts_path:str, output_path:str,
                 start_timestep:int, end_timestep:int, dt_time_step:int,
                 DEM_keymap:dict,
                 grid_info:dict,
                 weight_function:str, 
                 fields_to_export:dict, 
                 ignore_phases:bool):
        """
        Initialize the CoarseGraining class with necessary parameters.

        Parameters
        ----------
        particle_path : str
            Path to the particle data files.
        contacts_path : str
            Path to the contact data files.
        output_path : str
            Path where the output files will be saved.
        start_timestep : int
            The starting timestep for the simulation.
        end_timestep : int
            The ending timestep for the simulation.
        dt_time_step : int          
            The time step interval for the simulation.
        DEM_keymap : dict
            Mapping DEM data keys to their variable names.
        grid_info : dict
            Grid information such as dimensions, axes, and ranges.
        weight_function : str
            Type of weight function to use for coarse graining.
        fields_to_export : dict
            Fields to be exported.
        ignore_phases : bool
            Whether to ignore particle phases.  

        Attributes
        ----------
        particle_path : str
            Path to the particle data files.
        contacts_path : str 
            Path to the contact data files.
        time_steps : np.ndarray 
            Array of time steps for the simulation.
        DEM_keymap : dict
            Mapping DEM data keys to their variable names.
        grid_info : dict
            Grid information such as dimensions, axes, and ranges.
        weight_function : str
            Type of weight function to use for coarse graining.
        ignore_phases : bool
            Whether to ignore particle phases.
        field_to_export : dict
            Fields to be exported.
        fields_to_compute : dict
            Fields that need to be computed based on dependencies.
        output_path : str
            Path where the output files will be saved.

        """
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

    def data_sampling(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            Load the particle data for the first time step to obtain particle data.

            Returns
            -------
            BoundsData_t0 : np.ndarray, shape (3, 2)
                The bounds of the particle data for the first time step.
            Diameter_t0 : np.ndarray, shape (N,)
                The diameters of the particles for the first time step.
            Density_t0 : np.ndarray, shape (N,)
                The densities of the particles for the first time step.
            Mass_t0 : np.ndarray, shape (N,)
                The masses of the particles for the first time step.
            GlobalID_t0 : np.ndarray, shape (N,)
                The global IDs of the particles for the first time step.
        """
                
        # READ TIME STEP 0
        path_t0 = self.particle_path+f"{self.time_steps[0]:04d}.vtp" #:04d 
        self.file_type = file_read.get_file_type(path_t0) # detect the file type
        poly_out_t0 = file_read.reader(self.file_type, path_t0).GetOutput() # read the vtp file
        #polydata_t0 = Reader_vtm(self.Particle_path + f"{self.TimeSteps[0]}.vtm").GetOutput() # read the vtm file

        # BOUNDS      
        self.BoundsData_t0 = get_bounds(poly_out_t0).reshape(3,2)

        # PARTICLE PROPERTIES
        if self.DEM_keymap["Particle_Diameter"] is not None:
            Diameter_t0 = get_point_data_variable(self.DEM_keymap["Particle_Diameter"], poly_out_t0)
        else: 
            Diameter_t0 = get_point_data_variable(self.DEM_keymap["Particle_Radius"], poly_out_t0) * 2
        Density_t0 = get_point_data_variable(self.DEM_keymap["Particle_Density"], poly_out_t0)
        Mass_t0 = get_point_data_variable(self.DEM_keymap["Particle_Mass"], poly_out_t0)
        GlobalID_t0 = get_point_data_variable(self.DEM_keymap["Global_ID"], poly_out_t0)
        
        return self.BoundsData_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0
    
    def get_particle_size_statistics(self, diameter_t0:np.ndarray, mass_t0:np.ndarray) -> Tuple[float, float]:

        """
        Calculate particle size statistics based on particle diameter and mass.

        Parameters
        ----------
        diameter_t0 : np.ndarray, shape (N,)
            The particle diameters.
        mass_t0 : np.ndarray, shape (N,)
            The particle masses.
        Returns
        -------
        d43 : float
            The volume-weighted mean diameter.
        d32 : float
            The surface-weighted mean diameter.
    
        Attributes
        ----------
        d43 : float
            The volume-weighted mean diameter.
        d32 : float
            The surface-weighted mean diameter.
        dmax : float
            The maximum particle diameter.
        drms : float
            The root mean square diameter.
        d50 : float
            The median particle size.

        
        """
        
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
    
    def get_particle_phases(self, diameter_t0:np.ndarray, density_t0:np.ndarray, 
                            global_id:np.ndarray, n_max_phases = 6, plot=True):
        """
        Obtain particle phases based on diameter and density of a given time step (e.g., 0).

        Parameters
        ----------
        diameter_t0 : np.ndarray, shape (N,)
            The particle diameters. Sorted by global ID.
        density_t0 : np.ndarray, shape (N,)
            The particle densities. Sorted.
        global_id : np.ndarray, shape (N,)
            The global IDs of the particles. Sorted by global ID.
        n_max_phases : int, optional
            The maximum number of phases to find. Default is 6.
        plot : bool, optional
            Whether to plot the phases. Default is True.
        Attributes
        ----------
        phases : np.ndarray, shape (M, 2)
            The identified phases, each represented by a tuple of (diameter, density).
        Phase_Array : np.ndarray, shape (N,)
            An array indicating the phase of each particle (e.g., [0, 1, 2, ...]).
        cg_calc_mode : str
            The coarse graining calculation mode, either "Monodisperse" or "Polydisperse".
        
        """
        # IGNORE phases
        if self.ignore_phases:
            self.phases = np.array([[self.d50,density_t0.mean()]]) # use mean density and d50 as phase
            print("Ignoring phases. Using mean density and d50 as phase.")
            self.Phase_Array = None
            self.cg_calc_mode = "Monodisperse"
        
        # find different phases via clustering
        else: 
            self.phases, phase_array = find_phases(diameter_t0, density_t0, n_max_phases) # find the phases and phase array
            if plot:
                plot_phases(diameter_t0, density_t0, self.phases, phase_array)
            # particle phase array 
            if len(self.phases) == 1: # if only one phase is found
                self.cg_calc_mode = "Monodisperse"
                self.Phase_Array = None
            elif len(self.phases) > 1: # if multiple phases are found
                self.cg_calc_mode = "Polydisperse"
                self.Phase_Array = phase_array[np.argsort(global_id)] # sort the phase array by global ID

    def set_resolution(self, average_diameter:float, w_mult = 0.75):
        """
        Set the resolution for coarse graining based on the average particle diameter.
        
        Parameters
        ----------
        average_diameter : float
            The average particle diameter. Can be any representative diameter: d50, d32, d43, or drms.
        w_mult : float, optional
            The multiplier for the half-width of the smoothing kernel. Default is 0.75.
        
        Attributes
        ----------
        w : float
            The half-width of the smoothing kernel.
        c : float
            The cutoff distance for the smoothing kernel.

        """
        self.w = calc_half_width(average_diameter, w_mult) # calculate the half width
        self.c = calc_cutoff(self.w, self.weight_function) # calculate the cutoff distance

    def generate_grid(self, smoothing_length:float):

        """
        Generate the CG grid based on the provided grid information.

        Parameters
        ----------
        smoothing_length : float
            The smoothing length for the grid generation. Can be set to the cutoff distance calculated 
            in `set_resolution` or to a custom value.
        
        Attributes
        ----------
        GridPoints : np.ndarray, shape (N, 3)
            The grid points coordinates for the coarse graining.
        Nodes : np.ndarray, shape (D1, D2, D3)
            The grid nodes, where D1, D2, and D3 are the number of nodes in each dimension of the grid.
        Spacing : np.ndarray, shape (3,)
            The spacing between grid points in each dimension.
        Ranges : np.ndarray, shape (3, 2)
            The ranges of the grid in each dimension, calculated from the bounds of the particle data.
    
        """
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
        """

        Calculate coarse-grained fields over time steps. This method performs the following steps for each time step:
        1. Load particle and contact data for the current time step, using the `_load_data` method.
        2. Compute coarse-grained fields based on the loaded data, using the `_fields_single_time` method.
        3. Write the computed results to .h5 and .VTKHDF files, using the `_write_results` method.
        
        """
                                                                 
        print(" "); print("-------------------- Calculating Coarse Grained Fields --------------------"); print(" ")
        # time loop
        for t in range(len(self.time_steps)):
            print(f"---> Time step {t}: time {self.time_steps[t]:04d} ................................................")
            time_of_timestep = self.time_steps[t]
            time_start = time.time()
            # ========================================================================
            data = self._load_data(time_of_timestep); print("... data loaded") # Read the data for the current time step
            results = self._fields_single_time(data) # Calculate the CG fields for that time step
            self._write_results(results, t, time_of_timestep); print("... results written") # Write the results to .h5 and .VTKHDF files
            # ========================================================================
            time_end = time.time()
            print(f">> time step {t} took {time_end - time_start} to run."); print("  ")
            pass
    
    def _load_data(self, time_of_timestep:int) -> dict:
        """
        Load particle and contact data for a given timestep.

        Parameters
        ----------
        time_of_timestep : int
            The timestep for which to load the data.
        Returns
        -------
        data : dict
            A dictionary containing the loaded particle and contact data containing the following fields:
        Position : np.ndarray
            The positions of the particles.
        Velocity : np.ndarray
            The velocities of the particles.
        Diameter : np.ndarray
            The diameters of the particles.
        Density : np.ndarray
            The densities of the particles.
        Volume : np.ndarray
            The volumes of the particles.
        Mass : np.ndarray   
            The masses of the particles.
        Coordination_Number : np.ndarray
            The coordination numbers of the particles.
        Position_i : np.ndarray
            The positions of the particles involved in contacts.
        Force_i : np.ndarray
            The forces acting on the particles involved in contacts.
        BranchVector_i : np.ndarray
            The branch vectors of the contacts.
        CenterToCenterVector_LL : np.ndarray
            The center-to-center vectors of the contacts.
        Volume_i : np.ndarray
            The volumes of the particles involved in contacts.
        PhaseArray_i : np.ndarray
            The phase array of the particles involved in contacts.
        d_inContact_mean : float
            The mean distance of particles in contact.

        """
        print("Loading data ... ")
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
        print("  Particle data loaded") 
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
            ModelAxesRanges=Ranges_t, AxesPeriodicity=np.array([self.grid_info["x_axis_periodic"], 
                                                      self.grid_info["y_axis_periodic"], 
                                                      self.grid_info["z_axis_periodic"]]),
            Return_Volume=True, Particle_Phase_Array_t=self.Phase_Array ) 
        print("  Contact data loaded and mapped")
        # calculate coordination number
        if "coordination_number" in self.fields_to_compute:
            if Coordination_Number is None:
                print("  Coordination number not provided. Calculating it.")
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

    def _fields_single_time(self, data:dict) -> dict:
        """

        Calculate coarse-grained fields for a single time step. This method performs the following steps:
        1. Assign particles to grid nodes using the `_assign_particles_to_grid_nodes` method.
        2. Compute weights for particles and contacts using the `_compute_weights` method.
        3. Compute coarse-grained fields based on the dependencies defined in `field_dependencies.py` 
           using the `_compute_fields` method.

        Parameters
        ----------
        data : dict
            A dictionary containing the particle and contact data for the current 
            time step returned by `_load_data`.
        Returns
        -------
        cg_fields : dict
            A dictionary containing the coarse-grained fields, specified in `fields_to_export`, 
            computed for the current time step
            
        """
        
        # 1. assign particles to grid nodes
        particle_to_grid_map = self._assign_particles_to_grid_nodes(data) ; print("... particles assigned to grid nodes")
        # 2. compute weights for particles and contacts
        weights_particle, weights_integral = self._compute_weights(particle_to_grid_map, data); print("... weights computed")
        # 3. compute coarse-grained fields needed for export
        cg_fields = self._compute_fields(data, particle_to_grid_map, weights_particle, weights_integral); print("... fields computed")
        
        return cg_fields
    
    def _write_results(self, results_timestep:dict, time_step:int, time_of_timestep:int):
        """
        Write the computed coarse-grained fields to .h5 and .VTKHDF files 
        for a given time step  in the specified output path.

        Parameters
        ----------
        results_timestep : dict
            A dictionary containing the coarse-grained fields computed for the current time step, ret
            urned by `_fields_single_time`.
        time_step : int
            The index of the current time step in the simulation.
        time_of_timestep : int
            The actual time value corresponding to the current time step as given in `time_steps`.
    
        Notes
        -----
            The output files are written in both .h5 and .VTKHDF formats,
            allowing for easy access and visualization of the coarse-grained data.
            The .h5 file contains the coarse-grained positions and phases (if applicable),
            while the .VTKHDF file contains the coarse-grained fields in a format suitable for visualization
            using VTK-compatible software. The output files are named according to the weight function and coarse-graining calculation mode.
            For example, if the weight function is "Gaussian" and the calculation mode is "Monodisperse", 
            the output files will be named "CG_Gaussian_Monodisperse.h5" and "CG_Gaussian_Monodisperse_0001.vtkhdf".
            If the calculation mode is "Polydisperse", the output files will include phase information
            in their names, such as "CG_Gaussian_Polydisperse_Phase_1.vtkhdf".
         

        """


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

    def _assign_particles_to_grid_nodes(self, data:dict) -> dict:
        """
        Assign particles to grid nodes and calculate displacements and distances.
        This method performs the following steps:
        1. Match particles to grid points using a kd-tree function.
        2. Calculate the displacement and distance between particles and grid points.
        Parameters
        ----------
        data : dict
            A dictionary containing the particle and contact data for the current time step, 
            returned by `_load_data`.
        Returns
        -------
        particle_map : dict
        A dictionary containing the following keys:
        - grid_ind_p: np.ndarray, shape (N,), Indices of the grid points corresponding to particles.
        - part_ind_p: np.ndarray, shape (N,), Indices of the particles corresponding to grid points.
        - r_ri: np.ndarray, shape (N, 3), Displacement vectors from grid points to particles.
        - r_ri_dist: np.ndarray, shape (N,), Distances from grid points to particles.
        - grid_ind_c: np.ndarray, shape (M,), Indices of the grid points corresponding to contacts.
        - part_ind_c: np.ndarray, shape (M,), Indices of the contacts corresponding to grid points.
        - r_ri_c: np.ndarray, shape (M, 3)
            Displacement vectors from grid points to contacts.
        Notes
        -----
        This method uses the `particle_node_match` function to find the nearest grid points for each particle and contact.
        It also calculates the displacement vectors and distances between the grid points and the particles or contacts.
        The results are stored in a dictionary for further processing.
        The grid points are generated based on the bounds of the particle data and the specified grid dimensions and axes.
        The grid points are used to assign particles and contacts to the grid nodes, allowing for coarse graining of the data.
        The method prints progress messages to indicate the status of the matching and calculation processes.
        The method assumes that the grid points have already been generated and stored in `self.GridPoints`.    
        The method is designed to handle both particle and contact data, allowing for a comprehensive coarse graining process.  


        """

        print(f"Matching particles to grid points ...")
        # particle data .........................................
        grid_ind_p, part_ind_p = particle_node_match(self.GridPoints, data["Position"], self.c) # kd-tree function
        r_ri, r_ri_dist = calc_displacement(self.GridPoints, data["Position"], grid_ind_p, part_ind_p) # calculate the displacement and distance
        # contact data ...........................................
        grid_ind_c, part_ind_c = particle_node_match(self.GridPoints, data["Position_i"], self.c) # kd-tree function
        r_ri_c, _ = calc_displacement(self.GridPoints, data["Position_i"], grid_ind_c, part_ind_c)
                            

        return {
            "grid_ind_p": grid_ind_p,
            "part_ind_p": part_ind_p,
            "r_ri": r_ri,
            "r_ri_dist": r_ri_dist,
            "grid_ind_c": grid_ind_c,
            "part_ind_c": part_ind_c,
            "r_ri_c": r_ri_c
        }

    def _compute_weights(self, particle_map:dict, data:dict) -> Tuple[np.ndarray, np.ndarray]:

        """
        Compute the weights for particles and contacts based on the specified weight function.
        
        Parameters
        ----------
        particle_map : dict
            A dictionary containing the particle and contact data for the current time step,
            returned by `_assign_particles_to_grid_nodes`.
        data : dict
            A dictionary containing the particle and contact data for the current time step,
            returned by `_load_data`.
        
        Returns
        -------
        W_p : np.ndarray, shape (N,)
            The weights for particles, where N is the number of particles.
        Wint_c : np.ndarray, shape (M,)
            The integral of the weights for contacts, computed using trapezoidal integration, 
            where M is the number of contacts.
        
        Notes
        -----
        This method uses the specified weight function (Gaussian, Lucy, or HeavySide) to compute
        the weights for particles and contacts. The weights are computed based on the distances
        between the grid points and the particles or contacts.
        The method first selects the appropriate weight function based on the `weight_function` attribute.
        It then creates a hash table for the weight function and uses it to compute the weights for
        particles and contacts. The weights for particles are computed using the distances from the grid points
        to the particles, while the weights for contacts are computed using the distances along the branch vectors
        of the contacts.
        The method also performs trapezoidal integration for the contact weights to obtain a single integral value
        for the contacts.   

        """

        print(f"Computing weights ...")
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

    def _compute_fields(self, data:dict, g:dict, W_p:np.ndarray, Wint_c:np.ndarray) -> dict:

        """
        Compute coarse-grained fields for a given time step that are specified in `fields_to_compute`.
        
        Parameters
        ----------
        data : dict
            A dictionary containing the particle and contact data for the current time step,
            returned by `_load_data`.
        g : dict
            A dictionary containing the particle mapping information, returned by `_assign_particles_to_grid_nodes`.
        W_p : np.ndarray, shape (N,)
            The weights for particles, where N is the number of particles, computed in `_compute_weights`.
        Wint_c : np.ndarray, shape (M,)
            The integral of the weights for contacts, where M is the number of contacts, computed in `_compute_weights`.    
        
        Returns
        -------
        cg_fields : dict
            A dictionary containing the coarse-grained fields computed for the current time step,
            based on the specified fields to export in `fields_to_export`. Note that the fields to
            compute are defined in `fields_to_compute`, which is derived from `fields_to_export`.    
        

        """
        
        print(f"Computing Coarse Graining fields...")

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


        # 2. Compute fields based on the fields to compute ..................................................................
        # volume fraction
        if "volume_fraction" in self.fields_to_compute:
            VolumeFraction_CG = dispatcher.scalar(W_p, part_ind_p, grid_ind_p, Volume, None, Phase_Array_p, self.cg_calc_mode)
            print('  volume fraction done')

        # density
        if "density_mixture" in self.fields_to_compute:
            DensityMixture_CG = dispatcher.scalar(W_p, part_ind_p, grid_ind_p, Mass, None, Phase_Array_p, self.cg_calc_mode)
            print('  mixture density done')
    
        # velocity
        if "momentum_density" in self.fields_to_compute:
            MomentumDens_CG = dispatcher.vector(W_p, part_ind_p, grid_ind_p, Velocity, Mass,  Phase_Array_p, self.cg_calc_mode)
            print('  momentum density done')

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
                print('  velocity done')
            if "velocity_gradient" in self.fields_to_compute:
                GradV_CG = secondary.compute_vector_bulk_gradient(Velocity_CG[:,0,:], self.Nodes, self.Spacing) # Velocity Gradient
                print('  velocity gradient done')
            if "kinetic_tensor" in self.fields_to_compute:
                KineticTensor_CG = dispatcher.kinetic_tensor(W_p, part_ind_p, grid_ind_p, r_ri, Velocity, Mass, Velocity_CG[:, 0, :], GradV_CG, Phase_Array_p, self.cg_calc_mode) # Kinetic tensor
                print('  kinetic tensor done')

        # contact tensor
        if "contact_tensor" in self.fields_to_compute:
            ContactTensor_CG = dispatcher.tensor(Wint_c, part_ind_c, grid_ind_c, Force_i, BranchVector_i, None, PhaseArray_i, self.cg_calc_mode)
            print('  contact tensor done')
    
        # Density of particle cg
        if "density_particle" in self.fields_to_compute:
            DensityParticle_CG = DensityMixture_CG / VolumeFraction_CG
            print('  particle density done')

        # Total stress tensor
        if "total_stress_tensor" in self.fields_to_compute:
            TotalStressTensor_CG = KineticTensor_CG + ContactTensor_CG
            TotalStressTensor_CG_xyz_mag = secondary.compute_second_invariant(TotalStressTensor_CG, factor=0.5) # calculate the second invariant of the total stress tensor
            # deviatoric stress
            TotalStressDeviator_xyz = secondary.compute_deviatoric_tensor(TotalStressTensor_CG[...,:, :]) # 3D
            TotalStressDeviator_xy = secondary.compute_deviatoric_tensor(TotalStressTensor_CG[...,:2,:2]) # 2D
            TotalStressDeviator_xyz_mag = secondary.compute_second_invariant(TotalStressDeviator_xyz, factor=0.5) # 3D mag
            TotalStressDeviator_xy_mag = secondary.compute_second_invariant(TotalStressDeviator_xy, factor=0.5) # 2D mag
            print('  total stress done')

        # Volume-weighted mean diameter, d43
        if "d43" in self.fields_to_compute:
            d43_CG = scalars.mean_grainsize(W_p, part_ind_p, grid_ind_p, Diameter, n_flag=3)
            print('  d43 done')
        if "d32" in self.fields_to_compute:
            d32_CG = scalars.mean_grainsize(W_p, part_ind_p, grid_ind_p, Diameter, n_flag=2)
            print('  d32 done')
  
        # l_CG = None
        # Coordination number
        if "coordination_number" in self.fields_to_compute:
            CoordinationNumber_rattlers = scalars.scalar_x_volume(W_p, part_ind_p, grid_ind_p, Coordination_Number)
            print('  Z done')
        # Pressure
        if "pressure" in self.fields_to_compute:
            Pressure_xyz = secondary.compute_pressure(TotalStressTensor_CG)
            Pressure_xy = secondary.compute_pressure(TotalStressTensor_CG[...,0:2, 0:2])
            Pressure_x = TotalStressTensor_CG[...,0, 0] 
            Pressure_y = TotalStressTensor_CG[...,1, 1] 
            Pressure_z = TotalStressTensor_CG[...,2, 2] 
            print('  pressure done')

        # granular temperature
        if "granular_temperature" in self.fields_to_compute:
            GranularTemperature_xyz = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG) 
            GranularTemperature_x = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG[...,0,0]) 
            GranularTemperature_y = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG[...,1,1]) 
            GranularTemperature_z = secondary.compute_granular_temperature(DensityMixture_CG, KineticTensor_CG[...,2,2]) 
            print('  granular temp done')
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
            print('  shear rate done')
 
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
            print('  inertial number done')

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
            print('  mu done')
    
        # fabric tensor
        if "fabric_tensor" in self.fields_to_compute:
            l_i_mag = np.linalg.norm(CenterToCenterVector_LL, axis=-1) # calculate the magnitude of the branch vector
            l_i_normalised = CenterToCenterVector_LL / l_i_mag[:, np.newaxis]
            l_inContact_mean = np.mean(l_i_mag)
            FabricTensor_Monodisperse_CG = dispatcher.tensor(Wint_c, part_ind_c, grid_ind_c, l_i_normalised, l_i_normalised, Volume_i, None, "Monodisperse")
            #FabricTensor_Sun = Compute_Bulk_FabricTensor_Sun2015(l_i_normalised)
            print('  frabric tensor done')

        
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
       
                    
        
        return results

    
