
from Grid_Generation_Class import Grid_Generation # class for generating the CG grid
from Load_Data_Class import * # class for loading the data

from Calc_Fields_Numba_Functions import * # module for CG fields calculations
from CG_Smoothing_Functions import * # module for CG smoothing functions
from Handle_Contact_Data_Functions import * # module to handle contact data
from Particle_Search_Functions import * # module to search for particles in the grid nodes
from CG_Weights_Functions import * # module for CG weights calculations
from Calc_Secondary_Variables import * # module for calculating secondary variables
from Calc_Granular_Temperature_Alternatives import * # module for calculating granular temperature alternatives
from Phase_Clustering import Find_Phases # module for finding phases
from Write_Data_Class import * 

from vtk.util.numpy_support import vtk_to_numpy
import xarray as xr
import pandas as pd
import os 
import time


# NOT NEEDED 
import scipy.io as sio 

class Coarse_Graining: 
    def __init__(self, LoadingClass, WriteClass, GridClass, 
                 Particle_path, Contacts_path, Output_path,
                 start_timestep, end_timestep, dt_time_step,
                 DEM_keymap,
                 Grid_info,
                 CG_Function, 
                 fields_to_export, 
                 partialFields_Ignore):
        
        # imported classes
        self.Load = LoadingClass
        self.Write = WriteClass
        self.Grid = GridClass

        # data info
        self.Particle_path = Particle_path
        self.Contacts_path = Contacts_path
        self.TimeSteps = np.arange(start_timestep, end_timestep+1, dt_time_step)#np.array([3, 10, 25, 50, 75, 100, 125, 150, 175])#np.arange(start_timestep, end_timestep+1, dt_time_step)

        # DEM key mapping dictionary
        self.DEM_keymap = DEM_keymap

        # CG grid info dictionary
        self.grid_info = Grid_info
           
        # CG smoothing function of choice
        self.CG_Function = CG_Function

        # CG partial fields ignore
        self.PartialFields_Ignore = partialFields_Ignore

        # custom outputs and their dependencies
        self.field_to_export = fields_to_export
        self.fields_to_compute = self.Get_Fields_to_Compute(fields_to_export)

        # create output path folder 
        self.Output_path = Output_path
        if not os.path.exists(self.Output_path):
            os.makedirs(self.Output_path)
            print(f"Output path created: {self.Output_path}")
        else:
            print(f"Output path already exists: {self.Output_path}")
          
    
    def Get_Fields_to_Compute(self, fields_to_export):

        field_dependencies = {
                "density_particle": ["volume_fraction","density_mixture"],
                "velocity": ["momentum_density", "density_mixture"],
                "velocity_gradient" : ["velocity"],
                "shear_rate_tensor": ["velocity_gradient"],
                "kinetic_tensor": ["velocity", "velocity_gradient"],
                "total_stress_tensor": ["kinetic_tensor", "contact_tensor"],
                "pressure": ["total_stress_tensor"],
                "inertial_number": ["shear_rate_tensor", "pressure", "density_particle", "d43", "d32"],
                "frictional_coefficient": ["total_stress_tensor", "pressure"],
                "granular_temperature": ["kinetic_tensor", "density_mixture"],
                            }
    
        fields_to_compute = set()
        # recursive functions for dependencies
        def add_field(field):
            if field not in fields_to_compute:
                fields_to_compute.add(field)
                for dep in field_dependencies.get(field, []):
                    add_field(dep)

        # Add all user-selected outputs and their dependencies
        for field, include in fields_to_export.items():
            if include:
                add_field(field)

        return fields_to_compute

    def Get_ModelData_t0(self):
        print("Getting general model data")
        # import the particle data for 1st time step
        path_t0 = self.Particle_path+f"{self.TimeSteps[0]:04d}.vtp" #:04d 
        self.file_type = Get_File_Type(path_t0) # detect the file type
        polydata_t0 = Reader(self.file_type, path_t0).GetOutput() # read the vtp file
        #polydata_t0 = Reader_vtm(self.Particle_path + f"{self.TimeSteps[0]}.vtm").GetOutput() # read the vtm file
                
        bounds = np.array(polydata_t0.GetPoints().GetBounds()) 
        self.BoundsData_t0 = bounds.reshape(3,2)
        print(f"particle data bounds {self.BoundsData_t0}")

        if self.DEM_keymap["Particle_Diameter"] is not None:
            Diameter_t0 = vtk_to_numpy(polydata_t0.GetPointData().GetArray(self.DEM_keymap["Particle_Diameter"])) 
        else: 
            Diameter_t0 = vtk_to_numpy(polydata_t0.GetPointData().GetArray(self.DEM_keymap["Particle_Radius"])) * 2

        Density_t0 = vtk_to_numpy(polydata_t0.GetPointData().GetArray(self.DEM_keymap["Particle_Density"])) 
        Mass_t0 = vtk_to_numpy(polydata_t0.GetPointData().GetArray(self.DEM_keymap["Particle_Mass"]))
        GlobalID_t0 = vtk_to_numpy(polydata_t0.GetPointData().GetArray(self.DEM_keymap["Global_ID"]))
        
        return self.BoundsData_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0

    def Get_Phases(self, diameter_t0, density_t0, global_id, n_max_phases = 6):
     
        # find cluster of particle phases

        if self.PartialFields_Ignore:
            self.phases = np.array([[0,0]])
            self.Phase_Array = None
            self.cg_calc_mode = "Monodisperse"
        else: 

            print("Finding phases")
            self.phases, phase_array = Find_Phases(diameter_t0, density_t0, n_max_phases) # find the phases and phase array
            print(f"-------- Number of phases found: {len(self.phases)}")
            # polydisperse or monodisperse calcs
            if len(self.phases) == 1: # monodisperse
                print("Input data is Monodisperse ---> using Monodisperse calculation")
                self.cg_calc_mode = "Monodisperse"
            elif len(self.phases) > 1: # polydisperse
                if self.PartialFields_Ignore == True: # ignore partial fields
                    print("Input data is Polydisperse. Entered Partial_Fields_Ignore=True ---> using Monodisperse calculation")
                    self.cg_calc_mode = "Monodisperse"
                elif self.PartialFields_Ignore == False: # calculate partial fields
                    print("Input data is Polydisperse. Entered Partial_Fields_Ignore=False ---> using Polydisperse calculation")
                    self.cg_calc_mode = "Polydisperse"
                else:
                    raise ValueError("Partial_Fields_Ignore must be True or False")
            
            # particle phase array 
            if self.cg_calc_mode == "Monodisperse":
                self.Phase_Array = None
            elif self.cg_calc_mode == "Polydisperse":
                self.Phase_Array = phase_array[np.argsort(global_id)] # sort the phase array by global ID

    def Get_ParticleSize_Data(self,diameter_t0, mass_t0):
        
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
    
    def Calc_CG_Grid_Spacing(self, average_diameter, w_mult = 0.75):
        
        self.w = calc_half_width(average_diameter, w_mult) # calculate the half width
        self.c = calc_cutoff_distance(self.w, self.CG_Function) # calculate the cutoff distance
    
    def Generate_CG_Grid(self, smoothing_length):

        """Generate the CG grid based on the provided grid information."""
        print("generating grid")
        # unpack grid info
        self.grid_dim = self.grid_info["grid_dimension"]
        self.grid_axes = self.grid_info["grid_axes"]
        self.automatic_grid = self.grid_info["automatic_grid"]
        self.custom_grid_range = [self.grid_info["x_min"], self.grid_info["x_max"],
                                  self.grid_info["y_min"], self.grid_info["y_max"],
                                  self.grid_info["z_min"], self.grid_info["z_max"]]
        self.custom_grid_transects = [self.grid_info["x_transect"], self.grid_info["y_transect"], self.grid_info["z_transect"]]
        self.bound_period = [self.grid_info["x_axis_periodic"],
                             self.grid_info["y_axis_periodic"],
                             self.grid_info["z_axis_periodic"]]
        
        # generate the grid
        self.GridPoints, self.Nodes, self.Spacing, self.Ranges = self.Grid(smoothing_length, 
                                                                           self.BoundsData_t0, 
                                                                           self.grid_dim, 
                                                                           self.grid_axes,
                                                                           self.dmax, 
                                                                           self.automatic_grid, 
                                                                           self.custom_grid_range, 
                                                                           self.custom_grid_transects).Generate()
        
        # load matlab grid

    def Data_Loading(self, t):
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
        
    def Calcs_CG(self, c, data):

        # Unpack data
        Position = data["Position"]
        Velocity = data["Velocity"]
        Diameter = data["Diameter"]
        Density = data["Density"]
        Volume = data["Volume"]
        Mass = data["Mass"]
        Coordination_Number = data["Coordination_Number"]
        #Radius = data["Radius"]
        Phase_Array_p = data["Phase_Array"]
        Position_i = data["Position_i"] 
        Force_i = data["Force_i"]
        BranchVector_i = data["BranchVector_i"]
        CenterToCenterVector_LL = data["CenterToCenterVector_LL"]
        Volume_i = data["Volume_i"]
        PhaseArray_i = data["PhaseArray_i"]
        d_inContact_mean = data["d_inContact_mean"]

        # ASSIGN PARTICLES TO GRID NODES ==========================
        print( "  "); print(">>> Assign particles to grid nodes")
        # particle data ...........................................
        grid_ind_p, part_ind_p = Particle_Node_Correspondance(self.GridPoints, Position, c) # kd-tree function
        r_ri, r_ri_dist = Calc_Displacement_and_Distance(self.GridPoints, Position, 
                                                         grid_ind_p, part_ind_p, 
                                                         return_disp=True, return_dist=True) # calculate the displacement and distance


        # contact data ...........................................
        grid_ind_c, part_ind_c = Particle_Node_Correspondance(self.GridPoints, Position_i, c) # kd-tree function
        r_ri_c, _ = Calc_Displacement_and_Distance(self.GridPoints, Position_i,
                                                grid_ind_c, part_ind_c, 
                                                return_disp=True, return_dist=False)

        # CALCULATE THE CG WEITHTS ===========================
        print( "  "); print(">>> Calculate the CG weights")
        if self.CG_Function == "Gaussian":
            print("Using Gaussian weight function")
            WeightFunc = ComputeGaussianWeight
        elif self.CG_Function == "Lucy":
            print("Using Lucy weight function")
            WeightFunc = ComputeLucyWeight
        elif self.CG_Function == "HeavySide":
            print("Using HeavySide weight function")
            WeightFunc = ComputeHeavySideWeight
        else:
            raise ValueError("Invalid CG function. Choose 'Gaussian', 'Lucy', or 'HeavySide'.")

        # particle data ...........................
        hash_table_p, stepsize_p = make_hash_table(WeightFunc, c, sensitivity=1000)
        W_p = hash_table_search(r_ri_dist, hash_table_p, stepsize_p) # Compute the weights using the hash table algorithm (10000 is good for benchmarking)
        print("weights of particles calculated")
        
        # contact data ............................
        s = integration_scalar(0, 1, 10)#[:,np.newaxis,np.newaxis] # Integration vector s
        #print("integration scalar calculated") #dist_along_branch = np.linalg.norm(r_ri_c[np.newaxis,:,:] + s * BranchVector_i[part_ind_c][np.newaxis,:,:], axis=2) # Calculate distances projected along the branch vector
        dist_along_branch = compute_dist_along_branch_numba(r_ri_c, s, BranchVector_i, part_ind_c)
        #print(f"distance along branch vector calculated, shape: {dist_along_branch.shape}")
        hash_table_c, stepsize_c = make_hash_table(WeightFunc, c, sensitivity=1000)
        #print(f"hash made")
        W_c = hash_table_search(dist_along_branch, hash_table_c, stepsize_c) # Weights (10000 is good for benchmarking)
        #print("weights calculated")
        Wint_c = trapezoidal_integration(0, 1, 10, W_c) # Calculate the integral of the weights
        #print("integral of weights calculated")
        
        # ============================ CALCULATE THE CG FIELDS ============================
        print( "  ");print(">>> Calculate CG")
        # volume fraction
        if "volume_fraction" in self.fields_to_compute:
            VolumeFraction_CG = CG_Scalar(W_p, part_ind_p, grid_ind_p, Volume, None, Phase_Array_p, self.cg_calc_mode)
            print('volume fraction done')

        # density
        if "density_mixture" in self.fields_to_compute:
            DensityMixture_CG = CG_Scalar(W_p, part_ind_p, grid_ind_p, Mass, None, Phase_Array_p, self.cg_calc_mode)
            print('mixture density done')
    
        # velocity
        if "momentum_density" in self.fields_to_compute:
            MomentumDens_CG = CG_Vector(W_p, part_ind_p, grid_ind_p, Velocity, Mass,  Phase_Array_p, self.cg_calc_mode)
            print('momentum density done')

        # velocity and kinetic tensor    
        # Check if we have phase information
        if self.cg_calc_mode == 'Monodisperse':  # Monodisperse case
            if "velocity" in self.fields_to_compute:
                Velocity_CG = MomentumDens_CG / DensityMixture_CG[:, np.newaxis] # Velocity CG
            if "velocity_gradient" in self.fields_to_compute:
                GradV_CG = Compute_VectorBulk_Gradient(Velocity_CG, self.Nodes, self.Spacing) # Velocity Gradient
            if "kinetic_tensor" in self.fields_to_compute:
                KineticTensor_CG = CG_KineticTensor(W_p, part_ind_p, grid_ind_p, r_ri, Velocity, Mass, Velocity_CG, GradV_CG, Phase_Array_p, self.cg_calc_mode) # Kinetic tensor
        else:  # Polydisperse case
            if "velocity" in self.fields_to_compute:
                Velocity_CG = MomentumDens_CG / DensityMixture_CG[..., np.newaxis] # Velocity CG
                print('velocity done')
            if "velocity_gradient" in self.fields_to_compute:
                GradV_CG = Compute_VectorBulk_Gradient(Velocity_CG[:,0,:], self.Nodes, self.Spacing) # Velocity Gradient
                print('velocity gradient done')
            if "kinetic_tensor" in self.fields_to_compute:
                KineticTensor_CG = CG_KineticTensor(W_p, part_ind_p, grid_ind_p, r_ri, Velocity, Mass, Velocity_CG[:, 0, :], GradV_CG, Phase_Array_p, self.cg_calc_mode) # Kinetic tensor
                print('kinetic tensor done')
        # contact tensor
        if "contact_tensor" in self.fields_to_compute:
            ContactTensor_CG = CG_Tensor(Wint_c, part_ind_c, grid_ind_c, Force_i, BranchVector_i, None, PhaseArray_i, self.cg_calc_mode)
            print('contact tensor done')
        # SECONDARY VARIABLES ==========================================
        # Density of particle cg
        if "density_particle" in self.fields_to_compute:
            DensityParticle_CG = DensityMixture_CG / VolumeFraction_CG
            print('particle density done')

        # Total stress tensor
        if "total_stress_tensor" in self.fields_to_compute:
            TotalStressTensor_CG = KineticTensor_CG + ContactTensor_CG
            TotalStressTensor_CG_xyz_mag = Compute_Second_Invariant(TotalStressTensor_CG, factor=0.5) # calculate the second invariant of the total stress tensor
            # deviatoric stress
            TotalStressDeviator_xyz = Compute_Deviatoric_Tensor(TotalStressTensor_CG[...,:, :]) # 3D
            TotalStressDeviator_xy = Compute_Deviatoric_Tensor(TotalStressTensor_CG[...,:2,:2]) # 2D
            TotalStressDeviator_xyz_mag = Compute_Second_Invariant(TotalStressDeviator_xyz, factor=0.5) # 3D mag
            TotalStressDeviator_xy_mag = Compute_Second_Invariant(TotalStressDeviator_xy, factor=0.5) # 2D mag
            print('total stress done')

        # Volume-weighted mean diameter, d43
        if "d43" in self.fields_to_compute:
            d43_CG = CG_Weighted_Mean_Grainsize(W_p, part_ind_p, grid_ind_p, Diameter, n_flag=3)
            print('d43 done')
        if "d32" in self.fields_to_compute:
            d32_CG = CG_Weighted_Mean_Grainsize(W_p, part_ind_p, grid_ind_p, Diameter, n_flag=2)
            print('d32 done')
  
        # l_CG = None
        # Coordination number
        if "coordination_number" in self.fields_to_compute:
            CoordinationNumber_rattlers = CG_Scalar_1D_from_same_scalar(W_p, part_ind_p, grid_ind_p, Coordination_Number)
            print('Z done')
        # Pressure
        if "pressure" in self.fields_to_compute:
            Pressure_xyz = Compute_Pressure(TotalStressTensor_CG)
            Pressure_xy = Compute_Pressure(TotalStressTensor_CG[...,0:2, 0:2])
            Pressure_x = TotalStressTensor_CG[...,0, 0] 
            Pressure_y = TotalStressTensor_CG[...,1, 1] 
            Pressure_z = TotalStressTensor_CG[...,2, 2] 
            print('pressure done')

        # granular temperature
        if "granular_temperature" in self.fields_to_compute:
            GranularTemperature_xyz = Compute_Granular_Temperature(DensityMixture_CG, KineticTensor_CG) 
            GranularTemperature_x = Compute_Granular_Temperature(DensityMixture_CG, KineticTensor_CG[...,0,0]) 
            GranularTemperature_y = Compute_Granular_Temperature(DensityMixture_CG, KineticTensor_CG[...,1,1]) 
            GranularTemperature_z = Compute_Granular_Temperature(DensityMixture_CG, KineticTensor_CG[...,2,2]) 
            print('granular temp done')
        if "granular_temperature_alternatives" in self.fields_to_compute:
            GranularTemperature_KimKamrin20, GranularTemperature_LAMMPS = GranularTemperature_inSlices(dy=self.Spacing[1], 
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
            ShearRateTensor_xy = Compute_ShearRate_Tensor(GradV_CG[...,0:2, 0:2])
            ShearRateTensor_xy_mag = Compute_Second_Invariant(ShearRateTensor_xy, factor=2) # 2D mag
            # shear rate deviator 
            ShearRateDeviator_xy = Compute_Deviatoric_Tensor(ShearRateTensor_xy)
            ShearRateDeviator_xy_mag = Compute_Second_Invariant(ShearRateDeviator_xy, factor=2) # 2D mag
            # 3d - grid
            if self.grid_dim == 3:
                ShearRateTensor_xyz = Compute_ShearRate_Tensor(GradV_CG)
                ShearRateTensor_xyz_mag = Compute_Second_Invariant(ShearRateTensor_xyz, factor=2) # 3D mag
                # shear rate deviator 
                ShearRateDeviator_xyz = Compute_Deviatoric_Tensor(ShearRateTensor_xyz)
                ShearRateDeviator_xyz_mag = Compute_Second_Invariant(ShearRateDeviator_xyz, factor=2) # 3D mag
            print('shear rate done')
 
        # inertial number
        if "inertial_number" in self.fields_to_compute:
            InertialNumber_xy_Pxyz_d43 = Compute_InertialNumber(ShearRateTensor_xy_mag, Pressure_xyz, DensityParticle_CG, d43_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Pxy_d43 = Compute_InertialNumber(ShearRateTensor_xy_mag, Pressure_xy, DensityParticle_CG, d43_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Py_d43 = Compute_InertialNumber(ShearRateTensor_xy_mag, Pressure_y, DensityParticle_CG, d43_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Pxyz_d32 = Compute_InertialNumber(ShearRateTensor_xy_mag, Pressure_xyz, DensityParticle_CG, d32_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Pxy_d32 = Compute_InertialNumber(ShearRateTensor_xy_mag, Pressure_xy, DensityParticle_CG, d32_CG, self.phases[:,1], self.phases[:,0])
            InertialNumber_xy_Py_d32 = Compute_InertialNumber(ShearRateTensor_xy_mag, Pressure_y, DensityParticle_CG, d32_CG, self.phases[:,1], self.phases[:,0])
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
            FabricTensor_Monodisperse_CG = CG_Tensor(Wint_c, part_ind_c, grid_ind_c, l_i_normalised, l_i_normalised, Volume_i, None, "Monodisperse")
            FabricTensor_Sun = Compute_Bulk_FabricTensor_Sun2015(l_i_normalised)
            print('frabric tensor done')

        print(f" cg calculation over ")
        # ================================================================================================== # 
        # collect result to write it into file
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
        if self.field_to_export.get("fabric_tensor_Sun"): results["fabric_tensor_Sun"] = FabricTensor_Sun

        # shear rate tensor
        if self.field_to_export.get("shear_rate_tensor"): 
            results["shear_rate_tensor_xy"] = ShearRateTensor_xy
            results["shear_rate_tensor_xy_mag"] = ShearRateTensor_xy_mag
            results["shear_rate_tensor_xy_dev"] = ShearRateDeviator_xy
            results["shear_rate_tensor_xy_dev_mag"] = ShearRateDeviator_xy_mag
            if self.grid_dim == 3:
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

    def Calculate_CoarseGrained_Fields_Time(self, c_custom): 
                                                                 

        print("-------------------- Calculating Coarse Grained Fields --------------------")
        # default or custom c 
        if c_custom is None:
            if not hasattr(self, 'c'):
                raise AttributeError("self.c is not initialized. Ensure Calc_CG_Grid_Spacing is called before this method.")
            c = self.c
        else:
            c = c_custom
        # initialize an empty dataset to store the results

        # LOOP OVER TIME STEPS ==========================================
        for t in range(len(self.TimeSteps)):# len(self.TimeSteps)

            real_time = self.TimeSteps[t]

            print("------------------------------------------------------------")
            time_start = time.time()

            print( "  ");print(f">>> Loading data for time step {real_time}")
            data = self.Data_Loading(real_time) # Load the data for the current time step

            print( "  ");print(f">>> Calculating CG fields for time step {real_time}")
            results_timestep = self.Calcs_CG(c, data) # Calculate the CG fields for that time step

            print( "  ");print(f">>> Saving CG fields for time step {real_time}") #
            
            
            # write .h5 files 
            manager = H5XarrayManager(f"{self.Output_path}CG_{self.CG_Function}_{self.cg_calc_mode}.h5")
            manager.add_positions(self.GridPoints)

            if self.cg_calc_mode == "Polydisperse":
                phase_labels = ["Bulk"] + [f"Phase_{p}" for p in self.phases]
                manager.add_phases(phase_labels)

            manager.update_h5py_file(results_timestep, dim_index=t, dim_value=real_time, dim_name="time")

            # write .VTKHDF files for ParaView visualisation
            writer = VTKHDFWriter(node_dimensions=self.Nodes,  
                         node_spacing=self.Spacing, 
                         origin=self.GridPoints[0,:],
                         path=f"{self.Output_path}CG_{self.CG_Function}_{self.cg_calc_mode}_{real_time:04d}")
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

    def Test_CG_Stability(self, w_values_sweep):
        start = time.time()

        # Test the stability of the CG fields by sweeping over a range of w values
        print(f">>>> Testing CG stability with w/d values: {w_values_sweep}")

        # calculating the corresponding c values
        c_cust_vals = calc_cutoff_distance(w_values_sweep, self.CG_Function) # calculate the cutoff distance

        print(f">>> Loading data for TIME STEP 0 of your data set")
        # load the data
        data = self.Data_Loading(self.TimeSteps[0])

        # loop through the c values
        for i in range(len(w_values_sweep)):# len(w_values_sweep)

            w = w_values_sweep[i] ; print(f"W value = {w} -----------------------------------")
            
            # calculate the CG fields for that c value
            c_cust = c_cust_vals[i]
            results_w = self.Calcs_CG(c=c_cust, data=data)

            print(f">>> Saving CG fields for w value {w}")
            # write .h5 files 
            manager = H5XarrayManager(f"{self.Output_path}CG_{self.CG_Function}_{self.cg_calc_mode}_w_sweep.h5")
            manager.add_positions(self.GridPoints)
            if self.cg_calc_mode == "Polydisperse":
                phase_labels = ["Bulk"] + [f"Phase_{p}" for p in self.phases]
                manager.add_phases(phase_labels)
            manager.update_h5py_file(results_w, dim_index=i, dim_value=w, dim_name="w")

            # write .VTKHDF files for ParaView visualisation
            writer = VTKHDFWriter(node_dimensions=self.Nodes,  
                         node_spacing=self.Spacing, 
                         origin=self.GridPoints[0,:],
                         path=f"{self.Output_path}CG_{self.CG_Function}_{self.cg_calc_mode}_w_sweep")
            if self.cg_calc_mode == "Monodisperse":
                writer.write(data_dict=results_w)
            elif self.cg_calc_mode == "Polydisperse": 
                writer.write_polydisperse(data_dict=results_w,
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
              


         
            

        
        end = time.time()
        print(f"Time taken for CG stability test: {end - start} seconds")
        


if __name__ == "__main__":


    # ------------------------------ INPUT CONFIGURATION ---------------------------------
    # BEDLOAD EXAMPLE 
    particles_path = './bedload_example/VTU/DES_FB1_'
    contacts_path = './bedload_example/VTU/ENTIRE_DOMAIN_'

    key_mapping = { # tell it the exact name of your variables in the MFIX vtp files
    "Global_ID": "Particle_ID",
    "Particle_Velocity": "Velocity",
    "Particle_Diameter": "Diameter",    
    "Particle_Density": "Density",
    "Particle_Volume": "Volume",
    "Particle_Mass": "Mass",
    "Particle_Radius": None,
    "Coordination_Number": None, 
    "Particle_i_ID": "Particle_ID_1",
    "Particle_j_ID": "Particle_ID_2",
    "Force_ij": "FORCE_CHAIN_FC",
    "Contact_ij": "FORCE_CHAIN_CONTACT_POINT"
                    }

    grid_info = {
    "grid_dimension": 3, # 3D
    "grid_axes": 'xyz', # orientation (e.g., if 2D, it can be xy, xz, yz)
    "automatic_grid": False, 
    "x_min": 0.00105, # can be None if automatic_grid = True
    "x_max": 0.5,
    "y_min": 0.001, 
    "y_max": 0.24,
    "z_min": 0.0, 
    "z_max": 0.02,
    "x_transect": None, # if your grid is 2D or 1D you can select at what x,z or y it is located. Note: our vtkhdf writer doesnt like 2D grids
    "y_transect": None,
    "z_transect": None,
    "x_axis_periodic": True, # let it know if axis is periodic in case you want to CG up to the boundary and you don't have the contact points
    "y_axis_periodic": False,
    "z_axis_periodic": False
            }

    cg_custom_output = { # dependencies are taken care of inside the code, so only set true those that you want written out
                
        "volume_fraction": True, 
        "density_particle": True, 
        "density_mixture": True,
        "momentum_density": False,
        "velocity": True,
        "velocity_gradient": True,
        "kinetic_tensor": False,
        "contact_tensor": False,
        "total_stress_tensor": True,  
        "pressure": True,
        "granular_temperature": True,
        "granular_temperature_alternatives": False, # leave this as False, as I've not written exports for this yet
        "fabric_tensor": True,
        "inertial_number": True,
        "coordination_number": True,
        "d43": True,
        "d32": True,
        "frictional_coefficient": True,
        "shear_rate_tensor": True,

                        }
    out_path = "./bedload_example/PysammosCG/" # directory where you save CG output
    t0 = 150 ; tf = 400 # first and last time steps (assumes 1 unit increment of time steps). It pads it with zeros. 
    partialignore = True # forces monodisperse CG if True

    # ------------------------------- CALLING COARSE-GRAINING CLASS -----------------------------
    CG = Coarse_Graining(LoadingClass=None, WriteClass=None, GridClass=Grid_Generation, 
                         Particle_path=particles_path,
                         Contacts_path=contacts_path,
                         Output_path=out_path,
                         start_timestep=t0, end_timestep=tf, dt_time_step=1,
                         DEM_keymap=key_mapping,
                         Grid_info = grid_info, 
                         CG_Function="Lucy",
                         fields_to_export=cg_custom_output, 
                         partialFields_Ignore=partialignore)
                          

    # Load the size-relevant particle data for the first time step
    Bounds_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0 = CG.Get_ModelData_t0()
    
    # Get the phases
    CG.Get_Phases(Diameter_t0, Density_t0, GlobalID_t0, 8)
    print(">> Phases: ", CG.phases)
    print("       Diameter: ", CG.phases[:,0])
    print("       Density: ", CG.phases[:,1])
    
    # Calculate the particle size range
    d43, d32 = CG.Get_ParticleSize_Data(Diameter_t0, Mass_t0)
    print(">> d43: ", d43)
    print(">> dmax: ", CG.dmax)
    print(">> d50: ", CG.d50)
    print(">> d32: ", d32)
    print(">> drms: ", CG.drms)

    # Calculate the CG grid spacing
    CG.Calc_CG_Grid_Spacing(d43) # here you can input different number, to make w and c bigger or smaller 
    print("c :", CG.c)
    print("w :", CG.w)

    # Generate the CG grid
    CG.Generate_CG_Grid(smoothing_length=CG.c)
    print("Grid Points: ", CG.GridPoints.shape, "First Point: ", CG.GridPoints[0])
    print("Nodes: ", CG.Nodes)
    print("Spacing: ", CG.Spacing)

    # Calculate the CG fields
    CG.Calculate_CoarseGrained_Fields_Time(c_custom = None)

    print("------------------------------------------------------------")
    print(">>> Coarse Graining completed successfully")
    print("------------------------------------------------------------")
        
    
   
    
