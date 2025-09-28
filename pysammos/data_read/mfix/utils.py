import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

def get_bounds(polydata_t0):
    """
    Helper function to get bounds from polydata object.

    Inputs
    ------
    polydata_t0 : vtk.vtkPolyData
        The polydata object from which to extract bounds.
    
    Outputs
    -------
    np.ndarray
        An array containing the bounds of the polydata in the format [xmin, xmax, ymin, ymax, zmin, zmax].
    """
    return np.array(polydata_t0.GetPoints().GetBounds()).reshape(3, 2)

def get_point_data_variable(var_name, polydata):
    """
    Helper function to get point data variable from polydata.
    
    Parameters
    ----------
    var_name : str
        The name of the variable to extract from the point data.
    polydata : vtk.vtkPolyData
        The polydata object from which to extract the variable. 
    
    Returns
    -------
    np.ndarray
        The variable data as a NumPy array.
    """
    try:
        array = polydata.GetPointData().GetArray(var_name)
        return vtk_to_numpy(array)
        
    except:
        # Get available variable names
        point_data = polydata.GetPointData()
        available_vars = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
        
        raise ValueError("get_point_data_variable: Unable to read point data variable.\n"
            f"Variable '{var_name}' not found in point data.\n"
            f"Available variables: {available_vars}"
        )
