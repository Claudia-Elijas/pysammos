import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from data_read.mfix import cell_data, point_data, file_read


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


# read one of the data files in bedloadd_example VTU
path_t0 = "/exports/csce/datastore/geos/users/s1857688/Coarse_Graining/Pysammos_Reckoner/bedload_example/VTU/DES_FB1_0150.vtp"
file_type = file_read.get_file_type(path_t0) # detect the file type
polydata_t0 = file_read.reader(file_type, path_t0).GetOutput() # read the vtp file


bounds = get_bounds(polydata_t0).reshape(3, 2)
print(bounds)

Diameter_t0 = get_point_data_variable("Diameter", polydata_t0)
print("Diameter:", Diameter_t0)

        