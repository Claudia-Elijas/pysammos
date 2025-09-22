"""
This module provides functions to read and process cell data from MFIX simulations.
So far, it only includes functionality for reading contact data.
It extracts particle IDs, total forces, and contact points from the cell data, 
allowing for further analysis and manipulation.
It is designed to work with VTK data structures, converting them into NumPy arrays for easier handling.

The main function provided in this module is:
    1. :func:`contacts`: Extracts contact data from the input connection, including particle IDs, total forces
    and contact points.
"""


# import necessary libraries
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from typing import Tuple


# if contacts is in cell data...
def contacts(InputConnection:vtk.vtkAlgorithmOutput, 
    Part_ids_string="contact_ids", 
    Force_ij_string="total_force", 
    Contact_ij_string="contact_points")-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 

    r"""
    Extract contact data from the input connection, including particle IDs, total forces, and contact points.
    Inputs
    ------
    InputConnection : vtk.vtkAlgorithmOutput
        The input connection from which to extract cell data.
    Part_ids_string : str, optional
        The name of the array containing particle IDs in the cell data. Default is "contact_ids".
    Force_ij_string : str, optional
        The name of the array containing total forces in the cell data. Default is "total_force".
    Contact_ij_string : str, optional
        The name of the array containing contact points in the cell data. Default is "contact_points".
    Outputs 
    -------
    particle_i : ndarray, shape (M,)
        Array of particle IDs for the first particle in each contact.
    particle_j : ndarray, shape (M,)
        Array of particle IDs for the second particle in each contact.
    total_force : ndarray, shape (M, 3)
        Array of total forces for each contact.
    contact_points : ndarray, shape (M, 3)
        Array of contact points for each contact.
    """


    cell_data_ct = InputConnection.GetOutput().GetCellData(); print("Contact Data loaded as Cell Data")
    total_force = vtk_to_numpy(cell_data_ct.GetArray(Force_ij_string))
    contact_ids = vtk_to_numpy(cell_data_ct.GetArray(Part_ids_string))
    particle_i = contact_ids[:, 0]
    particle_j = contact_ids[:, 1]
    contact_points = vtk_to_numpy(cell_data_ct.GetArray(Contact_ij_string))

    return particle_i, particle_j, total_force, contact_points