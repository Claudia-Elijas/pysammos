import vtk
import numpy as np


def get_file_type(path):
    """
    Detects the type of VTK file (PolyData or UnstructuredGrid) by inspecting the file content.

    Returns:
    --------
    str
        The file type: "vtp" for PolyData or "vtk" for UnstructuredGrid.
    """
    with open(path, 'rb') as file:  # Open in binary mode
        first_bytes = file.read(100)  # Read the first 100 bytes
        if b"<?xml" in first_bytes:
            print("XML-based PolyData detected.")
            return "vtp"  # XML-based PolyData
        elif b"# vtk" in first_bytes:
            print("Legacy UnstructuredGrid detected.")
            return "vtk"  # Legacy UnstructuredGrid
        else:
            raise ValueError("Unsupported or unknown file format.")

def reader(file_type, path):

    # Use the appropriate reader
    if file_type == "vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_type == "vtk":
        reader = vtk.vtkUnstructuredGridReader()
    else:
        raise ValueError("Unsupported file format.")

    # Read the file
    reader.SetFileName(path)
    reader.Update()

    # Return the output data
    return reader
