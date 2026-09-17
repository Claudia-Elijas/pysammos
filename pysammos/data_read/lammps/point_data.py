"""
This module contains functions for reading and processing point data from LAMMPS simulations, including particle properties and contact information.
The main functions provided in this module are:
    1. :func:`particles`: Extracts particle properties such as global ID, position, velocity, diameter, density, volume, mass, and coordination number from the input data.
    2. :func:`contacts`: Extracts contact information such as particle IDs involved in contacts, contact forces, and contact points from the input data.
"""
import numpy as np
from .utils import _check_fields



def particles(data, Position_string, Global_ID_string, Velocity_string, Diameter_string, Density_string, Volume_string, Mass_string, Radius_string, Coordination_Number_string):
    
    '''
    Extracts particle properties from the data.

    Inputs
    ------
    data: pandas DataFrame containing the particle data.
    Global_ID_string: string, name of the column containing the global ID of each particle
    Position_string: list of strings, names of the columns containing the position components (e.g. [x, y, z])
    Velocity_string: list of strings, names of the columns containing the velocity components (e.g. [vx, vy, vz])
    Diameter_string: string, name of the column containing the diameter of each particle (optional if Radius_string is provided)
    Density_string: string, name of the column containing the density of each particle (optional)       
    Volume_string: string, name of the column containing the volume of each particle (optional)
    Mass_string: string, name of the column containing the mass of each particle (optional)
    Radius_string: string, name of the column containing the radius of each particle (optional if Diameter_string is provided)
    Coordination_Number_string: string, name of the column containing the coordination number of each particle (optional)
    
    Outputs
    ------- 
    Global_ID: numpy array of shape (n_particles,), containing the global ID of each particle
    Position: numpy array of shape (n_particles, 3), containing the position components of each particle
    Velocity: numpy array of shape (n_particles, 3), containing the velocity components of each particle
    Diameter: numpy array of shape (n_particles,), containing the diameter of each particle
    Density: numpy array of shape (n_particles,), containing the density of each particle (or None if not provided)
    Volume: numpy array of shape (n_particles,), containing the volume of each particle (or None if not provided)
    Mass: numpy array of shape (n_particles,), containing the mass of each particle (or None if not provided)
    Coordination_Number: numpy array of shape (n_particles,), containing the coordination number of each particle (or None if not provided)
    
    
    '''

    # check that required/optional columns are present
    # ---------------------------------------------------------------------------------------------------------------------------------------- #
    _check_fields(data, 
    required_fields = {'Global_ID': Global_ID_string,
                        'Position':  Position_string,
                        'Velocity':  Velocity_string,
                        'Diameter':  Diameter_string,
                        'Density':   Density_string,
                        'Mass': Mass_string,
                        'Volume': Volume_string,},
    optional_fields = {'Radius': Radius_string,  
                        'Coordination_Number': Coordination_Number_string,})
    #--------------------------------------------------------------------------------------------------------------------------------------- #
    
    # sort by global id
    data = data.sort_values(by=Global_ID_string)

    # global id
    if Global_ID_string is not None:
        Global_ID = data[Global_ID_string].values.astype(np.int32)
    else:
        raise ValueError("Global_ID_string is required for sorting and indexing.")

    # position
    if Position_string is not None and len(Position_string) == 3:
        Position = np.array([data[Position_string[0]].values, data[Position_string[1]].values, data[Position_string[2]].values]).T.astype(np.float32) 
    else:
        raise ValueError("Position_string must be a list of three column names.")

    # velocity
    if Velocity_string is not None and len(Velocity_string) == 3:
        Velocity = np.array([data[Velocity_string[0]].values, data[Velocity_string[1]].values, data[Velocity_string[2]].values]).T.astype(np.float32) 
    else:
        raise ValueError("Velocity_string must be a list of three column names.")

    # diameter
    if Diameter_string is not None:
        Diameter = data[Diameter_string].values.astype(np.float32)
    else:
        if Radius_string is not None:
            Diameter = 2 * data[Radius_string].values.astype(np.float32)
        else:
            raise ValueError("Either Diameter_string or Radius_string must be provided to compute Diameter.")
   
    # density
    if Density_string is not None:
        Density = data[Density_string].values.astype(np.float32)
    else:
        Density = None

    # volume
    if Volume_string is not None:
        Volume = data[Volume_string].values.astype(np.float32)
    else:
        Volume = None 

    # mass
    if Mass_string is not None:
        Mass = data[Mass_string].values.astype(np.float32)
    else:
        Mass = None

    # coordination number 
    if Coordination_Number_string is not None:
        Coordination_Number = data[Coordination_Number_string].values.astype(np.float32)
    else:
        Coordination_Number = None

    return Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Coordination_Number



def contacts(data, id1_string, id2_string, force_string, contact_point_string):

    '''
    Extracts contact information from the data.

    Inputs
    ------  
    data: pandas DataFrame containing the contact data.
    id1_string: string, name of the column containing the first particle ID in the contact
    id2_string: string, name of the column containing the second particle ID in the contact 
    force_string: list of strings, names of the columns containing the contact force components (e.g. [fx, fy, fz])
    contact_point_string: list of strings, names of the columns containing the contact point components (e.g. [cx, cy, cz])

    Outputs
    -------
    Particle_i_ID: numpy array of shape (n_contacts,), containing the IDs of the first particle in each contact
    Particle_j_ID: numpy array of shape (n_contacts,), containing the IDs of the second particle in each contact
    Force_ij: numpy array of shape (n_contacts, 3), containing the contact force components for each contact
    Contact_ij: numpy array of shape (n_contacts, 3), containing the contact point components for each contact, or None if contact_point_string is not provided
    
    '''

    # check that required/optional columns are present
    # ----------------------------------------------------- #
    _check_fields(data, 
    required_fields = {'Particle_i_ID': id1_string,
                        'Particle_j_ID': id2_string,
                        'Force_ij':       force_string},
    optional_fields = {'Contact_ij':     contact_point_string})
    # ----------------------------------------------------- #

    # particle i and j ids
    if id1_string is not None:
        Particle_i_ID = data[id1_string].values.astype(np.int32)
    else:
        Particle_i_ID = None
    if id2_string is not None:
        Particle_j_ID = data[id2_string].values.astype(np.int32)
    else:
        Particle_j_ID = None

    # contact force
    if force_string is not None:
        Force_ij = np.array([data[col].values for col in force_string]).T.astype(np.float32)
    else:
        Force_ij = None

    # contact point
    if contact_point_string is not None:
        Contact_ij = np.array([data[col].values for col in contact_point_string]).T.astype(np.float32)
    else:
        Contact_ij = None

    return Particle_i_ID, Particle_j_ID, Force_ij, Contact_ij