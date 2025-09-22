"""
This module provides functionality to read VTK files and determine their type based on the file content.
It supports both XML-based PolyData files (with .vtp extension) and legacy UnstructuredGrid files (with .vtk extension).
The appropriate VTK reader is selected based on the detected file type, allowing for further processing of the data.

The main functions provided in this module are:
    1. :func:`get_file_type`: Detects the type of VTK file by inspecting the file content.
    2. :func:`reader`: Reads the VTK file using the appropriate reader based on the detected file type.
"""


# import necessary libraries
import vtk
import numpy as np


def get_file_type(path:str) -> str:
    """
    Detects the type of VTK file (PolyData or UnstructuredGrid) by inspecting the file content.

    Inputs
    ------
    path :  str
        The path to the VTK file.

    Outputs
    --------
    str
        The file type: "vtp" for PolyData or "vtk" for UnstructuredGrid.
    Raises
    ------
    ValueError
        If the file format is unsupported or unknown.

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

    Inputs
    ------
    file_type : str
        The type of VTK file: "vtp" for PolyData or "vtk" for UnstructuredGrid.
    path : str
        The path to the VTK file.

    Outputs
    -------
    vtk.vtkAlgorithmOutput
        The output data from the VTK reader.

    Raises
    ------
    ValueError
        If the file format is unsupported.  
    
    Examples
    --------
    >>> file_type = get_file_type("example.vtp")
    XML-based PolyData detected.
    >>> reader_output = reader(file_type, "example.vtp")
    >>> print(type(reader_output))
    <class 'vtkmodules.vtkCommonExecutionModel.vtkAlgorithmOutput'>
    This indicates that the file has been read successfully and the output is a VTK algorithm output object.
    
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
