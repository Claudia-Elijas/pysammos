"""
Read VTK files and determine their type (PolyData or UnstructuredGrid) for MFIX simulations.
================================================================

This module provides functionality to read VTK files and determine their type based on the file content.
It supports both XML-based PolyData files (with .vtp extension) and legacy UnstructuredGrid files (with .vtk extension).
The appropriate VTK reader is selected based on the detected file type, allowing for further processing of the data.

Functions
---------
- `get_file_type`: Detects the type of VTK file by inspecting the file content.
- `reader`: Reads the VTK file using the appropriate reader based on the detected file type.

"""


# import necessary libraries
import vtk
import numpy as np


def get_file_type(path:str) -> str:
    """
    Detects the type of VTK file (PolyData or UnstructuredGrid) by inspecting the file content.

    Parameters:
    ----------
    path : str
        The path to the VTK file.

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

def reader(file_type:str, path:str) -> vtk.vtkAlgorithmOutput:
    """
    Reads the VTK file using the appropriate reader based on the detected file type.
    
    Parameters:
    ----------
    file_type : str
        The type of VTK file: "vtp" for PolyData or "vtk" for UnstructuredGrid.
    path : str
        The path to the VTK file.   
    
    Returns:
    --------
    vtk.vtkAlgorithmOutput
        The output data from the VTK reader.
    """

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
