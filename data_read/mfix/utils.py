import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

def get_bounds(polydata_t0):
    """
    Helper function to get bounds from polydata.
    """
    return np.array(polydata_t0.GetPoints().GetBounds()).reshape(3, 2)

def get_point_data_variable(var_name, polydata):
    """
    Helper function to get point data variable from polydata.
    """
    return vtk_to_numpy(polydata.GetPointData().GetArray(var_name))