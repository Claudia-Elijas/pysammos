"""
Cell data handling for MFIX simulations.
========================================

This module provides functions to read and process cell data from MFIX simulations,
specifically focusing on contact data. It extracts particle IDs, total forces,
and contact points from the cell data, allowing for further analysis and manipulation.
It is designed to work with VTK data structures, converting them into NumPy arrays for easier handling.

Functions
---------
- `contacts`: Extracts contact data from the input connection, including particle IDs, total forces
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

    cell_data_ct = InputConnection.GetOutput().GetCellData(); print("Contact Data loaded as Cell Data")
    total_force = vtk_to_numpy(cell_data_ct.GetArray(Force_ij_string))
    contact_ids = vtk_to_numpy(cell_data_ct.GetArray(Part_ids_string))
    particle_i = contact_ids[:, 0]
    particle_j = contact_ids[:, 1]
    contact_points = vtk_to_numpy(cell_data_ct.GetArray(Contact_ij_string))

    return particle_i, particle_j, total_force, contact_points