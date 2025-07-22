import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


# if contacts is in cell data...
def contacts(InputConnection, 
    Part_ids_string="contact_ids", 
    Force_ij_string="total_force", 
    Contact_ij_string="contact_points"): 

    cell_data_ct = InputConnection.GetOutput().GetCellData(); print("Contact Data loaded as Cell Data")
    total_force = vtk_to_numpy(cell_data_ct.GetArray(Force_ij_string))
    contact_ids = vtk_to_numpy(cell_data_ct.GetArray(Part_ids_string))
    particle_i = contact_ids[:, 0]
    particle_j = contact_ids[:, 1]
    contact_points = vtk_to_numpy(cell_data_ct.GetArray(Contact_ij_string))

    return particle_i, particle_j, total_force, contact_points