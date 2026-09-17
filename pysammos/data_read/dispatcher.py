"""
This module contains the dispatch_read function, which serves as a unified interface for reading particle and contact data from different DEM software outputs (e.g., MFIX and LAMMPS). 
"""
import numpy as np
# import the mfix reader
from pysammos.data_read.mfix import file_read as mfix_file_read
from pysammos.data_read.mfix import point_data as mfix_point_data
from pysammos.data_read.mfix import utils as mfix_utils
# import the lammps reader
from pysammos.data_read.lammps import file_read as lammps_file_read
from pysammos.data_read.lammps import point_data as lammps_point_data


def dispatch_read(dem_software, DEM_keymap, particle_path, contacts_path, time_of_timestep, t_pad):
    '''

    This function serves as a dispatcher to read particle and contact data from different DEM software outputs (e.g., MFIX and LAMMPS).
    Inputs
    ------
    dem_software: string, name of the DEM software used to generate the data (e.g., "mfix" or "lammps")
    DEM_keymap: dictionary, mapping of standardized field names to the corresponding column names in the input data (e.g., {"Global_ID": "id", "Particle_Position": ["x", "y", "z"], ...})
    particle_path: string, path to the particle data files (e.g., "path/to/particles/")
    contacts_path: string, path to the contact data files (e.g., "path/to/contacts/")
    time_of_timestep: int, the time step for which to read the data (e.g., 0, 1, 2, ...)
    Outputs
    -------
    particle_data: list of numpy arrays containing particle properties (Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Coordination_Number, bounds_t)
    contact_data: list of numpy arrays containing contact information (Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og)

    '''

    if dem_software == "mfix":

        PD = mfix_file_read.reader("vtp", particle_path + f"{time_of_timestep:0{t_pad}d}.vtp") 
        Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Coordination_Number, bounds_t = mfix_point_data.particles(PD,
            Global_ID_string=DEM_keymap["Global_ID"],
            Velocity_string=DEM_keymap["Particle_Velocity"],
            Diameter_string=DEM_keymap["Particle_Diameter"],
            Density_string=DEM_keymap["Particle_Density"],
            Volume_string=DEM_keymap["Particle_Volume"],
            Mass_string=DEM_keymap["Particle_Mass"],
            Radius_string=DEM_keymap["Particle_Radius"],
            Coordination_Number_string=DEM_keymap["Coordination_Number"])
        
        print("  Particle data loaded") 
        # Load contact data ===========================================================
        CD = mfix_file_read.reader("vtp", contacts_path + f"{time_of_timestep:0{t_pad}d}.vtp")  
        Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og = mfix_point_data.contacts(CD,                                                               
            Particle_i_string=DEM_keymap["Particle_i_ID"],
            Particle_j_string=DEM_keymap["Particle_j_ID"],
            Force_ij_string=DEM_keymap["Force_ij"],
            Contact_ij_string=DEM_keymap["Contact_ij"])

    elif dem_software == "lammps":

        bounds_t, df = lammps_file_read.read_dump(particle_path + f"{time_of_timestep:0{t_pad}d}.dump", type="particles")
        Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Coordination_Number = lammps_point_data.particles(df,
            Position_string=DEM_keymap["Particle_Position"],
            Global_ID_string=DEM_keymap["Global_ID"],
            Velocity_string=DEM_keymap["Particle_Velocity"],
            Diameter_string=DEM_keymap["Particle_Diameter"],
            Density_string=DEM_keymap["Particle_Density"],
            Volume_string=DEM_keymap["Particle_Volume"],
            Mass_string=DEM_keymap["Particle_Mass"],
            Radius_string=DEM_keymap["Particle_Radius"],
            Coordination_Number_string=DEM_keymap["Coordination_Number"])
        
        print("  Particle data loaded") 
        # Load contact data ===========================================================
        bounds_c, df_c = lammps_file_read.read_dump(contacts_path + f"{time_of_timestep:0{t_pad}d}.dump", type="contacts")
        Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og = lammps_point_data.contacts(df_c,
            id1_string=DEM_keymap["Particle_i_ID"],
            id2_string=DEM_keymap["Particle_j_ID"],
            force_string=DEM_keymap["Force_ij"],
            contact_point_string=DEM_keymap["Contact_ij"])

    else:
        raise ValueError(f"DEM software {dem_software} not recognized. Please choose either 'mfix' or 'lammps'.")

    return [Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Coordination_Number, bounds_t],[Particle_i_og, Particle_j_og, F_ij_og, Contact_ij_og]

def dispatch_read_t0(DEM_software, DEM_keymap, particle_path, time_steps, t_pad):
    
    if DEM_software == "mfix":
        # READ TIME STEP 0
        path_t0 = particle_path+f"{time_steps[0]:0{t_pad}d}.vtp" #:0{t_pad}d
        file_type = mfix_file_read.get_file_type(path_t0) # detect the file type
        poly_out_t0 = mfix_file_read.reader(file_type, path_t0).GetOutput() # read the vtp file
        #polydata_t0 = mfix_utils.Reader_vtm(particle_path + f"{time_steps[0]}.vtm").GetOutput() # read the vtm file

        # BOUNDS      
        BoundsData_t0 = mfix_utils.get_bounds(poly_out_t0).reshape(3,2)

        # PARTICLE PROPERTIES
        if DEM_keymap["Particle_Diameter"] is not None:
            Diameter_t0 = mfix_utils.get_point_data_variable(DEM_keymap["Particle_Diameter"], poly_out_t0)
        else: 
            Diameter_t0 = mfix_utils.get_point_data_variable(DEM_keymap["Particle_Radius"], poly_out_t0) * 2
        Density_t0 = mfix_utils.get_point_data_variable(DEM_keymap["Particle_Density"], poly_out_t0)
        Mass_t0 = mfix_utils.get_point_data_variable(DEM_keymap["Particle_Mass"], poly_out_t0)
        GlobalID_t0 = mfix_utils.get_point_data_variable(DEM_keymap["Global_ID"], poly_out_t0)

    elif DEM_software == "lammps":
        # READ TIME STEP 0
        path_t0 = particle_path+f"{time_steps[0]:0{t_pad}d}.dump" #:0{t_pad}d
        bounds_t, df = lammps_file_read.read_dump(path_t0, type="particles") # read the dump file

        # BOUNDS
        BoundsData_t0 = np.array(bounds_t).reshape(3,2)

        # PARTICLE PROPERTIES
        # check if thethe diameter density mass and global id given in the keymap are in the df
        if DEM_keymap["Particle_Diameter"] not in df.columns:
            if DEM_keymap["Particle_Radius"] not in df.columns:
                raise ValueError(f"Neither Particle_Diameter nor Particle_Radius found in the data. Please check the keymap and the input data.")
        elif DEM_keymap["Particle_Density"] not in df.columns:
            raise ValueError(f"Particle_Density not found in the data. Please check the keymap and the input data.")
        elif DEM_keymap["Particle_Mass"] not in df.columns:
            raise ValueError(f"Particle_Mass not found in the data. Please check the keymap and the input data.")
        elif DEM_keymap["Global_ID"] not in df.columns:
            raise ValueError(f"Global_ID not found in the data. Please check the keymap and the input data.")

        if DEM_keymap["Particle_Diameter"] is not None:
            Diameter_t0 = df[DEM_keymap["Particle_Diameter"]].values
        else:
            Diameter_t0 = df[DEM_keymap["Particle_Radius"]].values * 2
        Density_t0 = df[DEM_keymap["Particle_Density"]].values
        Mass_t0 = df[DEM_keymap["Particle_Mass"]].values
        GlobalID_t0 = df[DEM_keymap["Global_ID"]].values

    else:
        raise ValueError(f"DEM software {DEM_software} not recognized. Please choose either 'mfix' or 'lammps'.")
            
    return BoundsData_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0