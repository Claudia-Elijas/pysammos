"""
Point data reading functions for MFIX simulations.
==================================================

This module provides functions to read and process point data from MFIX simulations.
It extracts particle information such as position, global ID, velocity, diameter, density,
volume, mass, and coordination number from the point data, allowing for further analysis and manipulation.
It is designed to work with VTK data structures, converting them into NumPy arrays for easier handling.

Functions
---------
- `particles`: Extracts particle data from the input connection, including position, global ID,
    velocity, diameter, density, volume, mass, and coordination number.
- `contacts`: Extracts contact data from the input connection, including particle IDs, total forces
    and contact points.
- `Reader_vtm`: Reads VTM files and extracts all PolyData blocks, merging them into a single PolyData object.
"""

# import necessary libraries
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from .utils import get_point_data_variable, get_bounds
from typing import Tuple, Optional


def particles(InputConnection:vtk.vtkAlgorithmOutput, 
                Global_ID_string:Optional[str]="Particle_ID", 
                Velocity_string:Optional[str]="Velocity", 
                Diameter_string:Optional[str]="Diameter",
                Density_string:Optional[str]="Density",   
                Volume_string:Optional[str]="Volume",
                Mass_string:Optional[str]="Mass", 
                Radius_string:Optional[str]="Radius", 
                Coordination_Number_string="Coordination_Number")-> Tuple[np.ndarray, np.ndarray, 
                    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], 
                    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Tuple]:
    """

    Extracts particle data from the input connection, including position, global ID,
    velocity, diameter, density, volume, mass, and coordination number.

    Parameters:
    ----------
    InputConnection : vtk.vtkAlgorithmOutput
        The input connection containing the VTK data.
    Global_ID_string : str, optional
        The name of the global ID variable in the point data (default is "Particle_ID").
    Velocity_string : str, optional
        The name of the velocity variable in the point data (default is "Velocity").
    Diameter_string : str, optional
        The name of the diameter variable in the point data (default is "Diameter").
    Density_string : str, optional
        The name of the density variable in the point data (default is "Density").
    Volume_string : str, optional
        The name of the volume variable in the point data (default is "Volume").
    Mass_string : str, optional
        The name of the mass variable in the point data (default is "Mass").
    Radius_string : str, optional       
        The name of the radius variable in the point data (default is "Radius").
    Coordination_Number_string : str, optional
        The name of the coordination number variable in the point data (default is "Coordination_Number").
    
    Returns:
    --------
    Position_sorted : np.ndarray, shape (N, 3).
        The sorted positions of the particles.
    Global_ID_sorted : np.ndarray, shape (N,).
        The sorted global IDs of the particles.
    Velocity_sorted : np.ndarray, shape (N, 3), optional
        The sorted velocities of the particles, if available.
    Diameter_sorted : np.ndarray, shape (N,), optional
        The sorted diameters of the particles, if available.
    Density_sorted : np.ndarray, shape (N,), optional
        The sorted densities of the particles, if available.
    Volume_sorted : np.ndarray, shape (N,), optional
        The sorted volumes of the particles, if available.
    Mass_sorted : np.ndarray, shape (N,), optional
        The sorted masses of the particles, if available.
    Coordination_Number_sorted : np.ndarray, shape (N,), optional
        The sorted coordination numbers of the particles, if available.
    Bounds_t : tuple
        The bounds of the point data, represented as a tuple of (xmin, xmax, ymin, ymax, zmin, zmax).
    
    Raises:
    -------
    ValueError
        If any of the required strings (Global_ID_string, Density_string, etc.) are None or not provided.
    
    Notes:
    -----
    - The function retrieves the point data from the input connection and extracts the specified variables.
    - The global IDs are sorted, and the corresponding positions, velocities, diameters, densities
      volumes, masses, and coordination numbers are also sorted based on the global IDs.
    - If any optional variables (Velocity_string, Diameter_string, etc.) are not provided,
      the corresponding output will be None.
    - The function returns the sorted arrays and the bounds of the point data.  


    """

    poly_output = InputConnection.GetOutput()

    if Global_ID_string is not None:
        Global_ID = get_point_data_variable(Global_ID_string, poly_output)
        sorted_idx = np.argsort(Global_ID)
        Global_ID_sorted = Global_ID[sorted_idx].astype(np.int32)

        Position = vtk_to_numpy(poly_output.GetPoints().GetData())
        Position_sorted = Position[sorted_idx]
    else:
        raise ValueError("Global_ID_string is None. Please provide a valid string.")
    
    if Velocity_string is not None:
        Velocity = get_point_data_variable(Velocity_string, poly_output)
        Velocity_sorted = Velocity[sorted_idx]
    else:
        Velocity_sorted = None

    if Diameter_string is not None:
        Diameter = get_point_data_variable(Diameter_string, poly_output)
        Diameter_sorted = Diameter[sorted_idx]
    else:

        if Radius_string is not None:
            Radius = get_point_data_variable(Radius_string, poly_output)
            Radius_sorted = Radius[sorted_idx]
            Diameter_sorted = 2 * Radius_sorted
        if Radius_string is None: 
            raise ValueError("Diameter_string is None. Please provide a valid string.")

    if Density_string is not None:
        Density = get_point_data_variable(Density_string, poly_output)
        Density_sorted = Density[sorted_idx]
    else:
        raise ValueError("Density_string is None. Please provide a valid string.")

    if Volume_string is not None:
        Volume = get_point_data_variable(Volume_string, poly_output)
        Volume_sorted = Volume[sorted_idx]
    else:
        Volume_sorted = None

    if Mass_string is not None:
        Mass = get_point_data_variable(Mass_string, poly_output)
        Mass_sorted = Mass[sorted_idx]
    else:
        Mass_sorted = None

    if Coordination_Number_string is not None:
        Coordination_Number = get_point_data_variable(Coordination_Number_string, poly_output)
        Coordination_Number_sorted = Coordination_Number[sorted_idx]
    else:
        Coordination_Number_sorted = None
        Coordination_Number = None
    Bounds_t = get_bounds(poly_output)
    return Position_sorted, Global_ID_sorted, Velocity_sorted, Diameter_sorted, Density_sorted, Volume_sorted, Mass_sorted, Coordination_Number_sorted, Bounds_t
    #return Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Radius

def contacts(InputConnection:vtk.vtkAlgorithmOutput,    
                Particle_i_string:Optional[str]="Particle_ID_1", 
                Particle_j_string:Optional[str]="Particle_ID_2", 
                Force_ij_string:Optional[str]="FORCE_CHAIN_FC", 
                Contact_ij_string:Optional[str]=None)-> Tuple[np.ndarray, np.ndarray, 
                                                            np.ndarray, Optional[np.ndarray]]:
    
    """
    Extracts contact data from the input connection, including particle IDs, total forces
    and contact points.

    Parameters:
    ----------
    InputConnection : vtk.vtkAlgorithmOutput
        The input connection containing the VTK data.
    Particle_i_string : str, optional
        The name of the first particle ID variable in the point data (default is "Particle_ID_1").
    Particle_j_string : str, optional
        The name of the second particle ID variable in the point data (default is "Particle_ID_2").
    Force_ij_string : str, optional
        The name of the total force variable in the point data (default is "FORCE_CHAIN_FC").
    Contact_ij_string : str, optional   
        The name of the contact points variable in the point data (default is None).

    Returns:
    --------
    Particle_i : np.ndarray, shape (N,)
        The particle IDs of the first particles involved in the contacts.
    Particle_j : np.ndarray, shape (N,)
        The particle IDs of the second particles involved in the contacts.
    F_ij : np.ndarray, shape (N, 3)
        The total forces acting on the contacts, if available.
    Contact_ij : np.ndarray, shape (N, 3), optional
        The contact points between the particles, if available.


    """

    poly_output = InputConnection.GetOutput()
    F_ij = get_point_data_variable(Force_ij_string, poly_output) if Force_ij_string else None
    Particle_i = get_point_data_variable(Particle_i_string, poly_output) if Particle_i_string else None
    Particle_j = get_point_data_variable(Particle_j_string, poly_output) if Particle_j_string else None
    Contact_ij = get_point_data_variable(Contact_ij_string, poly_output) if Contact_ij_string else None

    return Particle_i, Particle_j, F_ij, Contact_ij



# ============================================= #
# for DEM data containing multiple PolyData blocks
# ============================================= #
# Read vtp data
def extract_all_polydata(dataset, level=0):
    """Recursively extract all vtkPolyData blocks from any nested vtkMultiBlockDataSet or vtkMultiPieceDataSet."""
    polydata_list = []
    if isinstance(dataset, vtk.vtkPolyData):
        polydata_list.append(dataset)
    elif isinstance(dataset, vtk.vtkMultiBlockDataSet):
        for i in range(dataset.GetNumberOfBlocks()):
            block = dataset.GetBlock(i)
            if block is not None:
                polydata_list.extend(extract_all_polydata(block, level+1))
    elif isinstance(dataset, vtk.vtkMultiPieceDataSet):
        for i in range(dataset.GetNumberOfPieces()):
            piece = dataset.GetPiece(i)
            if piece is not None:
                polydata_list.extend(extract_all_polydata(piece, level+1))
    return polydata_list
def Reader_vtm(path):
    # Load the VTM file
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(path)
    reader.Update()

    top_dataset = reader.GetOutput()

    # Recursively extract all PolyData blocks
    all_polydata_blocks = extract_all_polydata(top_dataset)

    # Merge them using vtkAppendPolyData
    append_filter = vtk.vtkAppendPolyData()
    for poly in all_polydata_blocks:
        append_filter.AddInputData(poly)
    append_filter.Update()

    return append_filter



