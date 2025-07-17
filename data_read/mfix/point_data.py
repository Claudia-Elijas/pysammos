import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


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
    Point_Data = poly_output.GetPointData()

    if Global_ID_string is not None:
        Global_ID = vtk_to_numpy(Point_Data.GetArray(Global_ID_string)) 
        sorted_idx = np.argsort(Global_ID)
        Global_ID_sorted = Global_ID[sorted_idx].astype(np.int32)

        Position = vtk_to_numpy(poly_output.GetPoints().GetData())
        Position_sorted = Position[sorted_idx]
    else:
        raise ValueError("Global_ID_string is None. Please provide a valid string.")
    
    if Velocity_string is not None:
        Velocity = vtk_to_numpy(Point_Data.GetArray(Velocity_string))
        Velocity_sorted = Velocity[sorted_idx]
    else:
        Velocity_sorted = None

    if Diameter_string is not None:
        Diameter = vtk_to_numpy(Point_Data.GetArray(Diameter_string))
        Diameter_sorted = Diameter[sorted_idx]
    else:

        if Radius_string is not None:
            Diameter_sorted = None
        if Radius_string is None: 
            raise ValueError("Diameter_string is None. Please provide a valid string.")

    if Density_string is not None:
        Density = vtk_to_numpy(Point_Data.GetArray(Density_string))
        Density_sorted = Density[sorted_idx]
    else:
        raise ValueError("Density_string is None. Please provide a valid string.")

    if Volume_string is not None:
        Volume = vtk_to_numpy(Point_Data.GetArray(Volume_string))
        Volume_sorted = Volume[sorted_idx]
    else:
        Volume_sorted = None

    if Mass_string is not None:
        Mass = vtk_to_numpy(Point_Data.GetArray(Mass_string))
        Mass_sorted = Mass[sorted_idx]
    else:
        Mass_sorted = None

    if Radius_string is not None:
        Radius = vtk_to_numpy(Point_Data.GetArray(Radius_string))
        Radius_sorted = Radius[sorted_idx]
    else:
        Radius_sorted = None
        Radius = None
    if Coordination_Number_string is not None:
        Coordination_Number = vtk_to_numpy(Point_Data.GetArray(Coordination_Number_string))
        Coordination_Number_sorted = Coordination_Number[sorted_idx]
    else:
        Coordination_Number_sorted = None
        Coordination_Number = None
    Bounds_t = poly_output.GetPoints().GetBounds()
    return Position_sorted, Global_ID_sorted, Velocity_sorted, Diameter_sorted, Density_sorted, Volume_sorted, Mass_sorted, Radius_sorted, Coordination_Number_sorted, Bounds_t
    #return Position, Global_ID, Velocity, Diameter, Density, Volume, Mass, Radius

def contacts(InputConnection,    
                Particle_i_string="Particle_ID_1", 
                Particle_j_string="Particle_ID_2", 
                Force_ij_string="FORCE_CHAIN_FC", 
                Contact_ij_string=None):

    F_ij = vtk_to_numpy(InputConnection.GetOutput().GetPointData().GetArray(Force_ij_string)) if Force_ij_string else None
    Particle_i = vtk_to_numpy(InputConnection.GetOutput().GetPointData().GetArray(Particle_i_string)) if Particle_i_string else None
    Particle_j = vtk_to_numpy(InputConnection.GetOutput().GetPointData().GetArray(Particle_j_string)) if Particle_j_string else None
    Contact_ij = vtk_to_numpy(InputConnection.GetOutput().GetPointData().GetArray(Contact_ij_string)) if Contact_ij_string else None

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



