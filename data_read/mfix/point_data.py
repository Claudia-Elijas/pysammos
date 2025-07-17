import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from .utils import get_point_data_variable, get_bounds


def particles(InputConnection, 
                Global_ID_string="Particle_ID", 
                Velocity_string="Velocity", 
                Diameter_string="Diameter",
                Density_string="Density",   
                Volume_string="Volume",
                Mass_string="Mass", 
                Radius_string="Radius", 
                Coordination_Number_string="Coordination_Number",): 
    
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
            Diameter_sorted = None
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

    if Radius_string is not None:
        Radius = get_point_data_variable(Radius_string, poly_output)
        Radius_sorted = Radius[sorted_idx]
    else:
        Radius_sorted = None
        Radius = None
    if Coordination_Number_string is not None:
        Coordination_Number = get_point_data_variable(Coordination_Number_string, poly_output)
        Coordination_Number_sorted = Coordination_Number[sorted_idx]
    else:
        Coordination_Number_sorted = None
        Coordination_Number = None
    Bounds_t = get_bounds(poly_output)
    return Position_sorted, Global_ID_sorted, Velocity_sorted, Diameter_sorted, Density_sorted, Volume_sorted, Mass_sorted, Radius_sorted, Coordination_Number_sorted, Bounds_t
    #return Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Radius

def contacts(InputConnection,    
                Particle_i_string="Particle_ID_1", 
                Particle_j_string="Particle_ID_2", 
                Force_ij_string="FORCE_CHAIN_FC", 
                Contact_ij_string=None):

    poly_output = InputConnection.GetOutput()
    F_ij = get_point_data_variable(Force_ij_string, poly_output) if Force_ij_string else None
    Particle_i = get_point_data_variable(Particle_i_string, poly_output) if Particle_i_string else None
    Particle_j = get_point_data_variable(Particle_j_string, poly_output) if Particle_j_string else None
    Contact_ij = get_point_data_variable(Contact_ij_string, poly_output) if Contact_ij_string else None

    return Particle_i, Particle_j, F_ij, Contact_ij



# ============================================= #
# for benchmarking JP's models
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



