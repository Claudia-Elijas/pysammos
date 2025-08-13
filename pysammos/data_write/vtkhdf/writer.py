
"""
Module for writing VTKHDF files.
================================

This module provides the `VTKHDFWriter` class, which is used to write data to VTKHDF files.
It supports writing scalar, vector, and tensor data, and can handle both single-phase and polydisperse data.
The writer uses the `pyvista` library for handling VTK data structures and `h5py` for writing HDF5 files.
It also includes functionality to set the origin and dimensions of the data grid.
It is particularly useful for visualizing simulation data in a format compatible with VTK and HDF5 standards.

Class: VTKHDFWriter
-------------------

This class manages the writing of VTKHDF files with specified node dimensions, spacing, and origin.
It provides methods to write scalar, vector, and tensor data, and can handle both single-phase and polydisperse data.
It uses `pyvista` for creating the VTK data structure and `h5py` for writing the data to HDF5 files.
It is designed to ensure that the data is stored in a format that can be easily read and visualized using VTK-compatible tools. 

Methods:
- `__init__`: Initializes the writer with node dimensions, spacing, origin, and file path.
- `write`: Writes the provided data dictionary to a VTKHDF file, handling different data shapes (scalar, vector, tensor).
- `write_polydisperse`: Writes polydisperse data to a VTKHDF file, handling both phase-independent and phase-dependent fields.
"""

# importing necessary modules and libraries
from . import core as v5i
import h5py
import pyvista as pv
from typing import Tuple


class VTKHDFWriter:
    def __init__(self, node_dimensions:Tuple, node_spacing:Tuple, origin:Tuple, path:str):
        """
        Initializes the VTKHDFWriter with node dimensions, spacing, origin, and file path.
        
        Parameters
        ----------
        node_dimensions : tuple
            Dimensions of the grid in number of nodes (e.g., (nx, ny, nz)).
        node_spacing : tuple
            Spacing between nodes in each dimension (e.g., (dx, dy, dz)).
        origin : tuple
            Origin of the grid in the coordinate system (e.g., (ox, oy, oz)).
        path : str
            Path where the VTKHDF file will be saved.   
        
        Attributes
        ----------
        node_dimensions : tuple
            Dimensions of the grid in number of nodes.
        node_spacing : tuple
            Spacing between nodes in each dimension.
        path : str
            Path where the VTKHDF file will be saved.
        origin : tuple
            Origin of the grid in the coordinate system.

        """
        self.node_dimensions = tuple(node_dimensions)
        self.node_spacing = tuple(node_spacing)
        self.path = path
        self.origin = origin #v5i.origin_of_centered_image(self.node_dimensions, self.node_spacing, 2)
        
    def write(self, data_dict:dict):

        """
        Writes the provided data dictionary to a VTKHDF file, handling various data shapes (scalar, vector, tensor).
        
        Parameters
        ----------
        data_dict : dict
            Dictionary containing data to be written, where keys are variable names and values are numpy arrays.    
        The data arrays should be structured as follows:
        - Scalar data: 1D array (shape: (n_nodes,))
        - Vector data: 2D array (shape: (n_nodes, 3))
        - Tensor data: 3D array (shape: (n_nodes, 3, 3))
        Each key in the dictionary corresponds to a variable name, and the values are the data arrays
        to be written to the VTKHDF file.
        If the data is not in the expected shape, it will be reshaped to fit the node dimensions.

        """

        # Create a pyvista ImageData object with the specified dimensions and spacing
        box = pv.ImageData(dimensions=self.node_dimensions, spacing=self.node_spacing, origin=self.origin)

        for key, value in data_dict.items(): # Iterate over each key-value pair in the data dictionary
            if value is not None: # Check if the value is not None
                dims = value.ndim
                if dims == 1: # Scalar data
                    data = value.reshape(*self.node_dimensions, order='C') # Reshape to match node dimensions
                    v5i.set_point_scalar(box, data, key) # Set scalar data in the ImageData object
                elif dims == 2: # Vector data
                    data = value.reshape((*self.node_dimensions, 3), order='C')
                    v5i.set_point_vector(box, data, key) # Set vector data in the ImageData object
                elif dims == 3: # Tensor data
                    ncomp = len(value[0,...].flatten(order='C')) # Number of components in the tensor
                    data = value.reshape((*self.node_dimensions, ncomp), order='C')
                    v5i.set_point_tensor(box, data, key, components=ncomp) # Set tensor data in the ImageData object
                else:
                    print(f"{key}: unknown type, shape = {value.shape}")

        filename = self.path + ".vtkhdf"
        with h5py.File(filename, "w") as f:
            v5i.write_vtkhdf(f, box, compression='gzip', compression_opts=4) # Write the ImageData object to the HDF5 file
        print(f"  File successfully written to {filename}")
        
    def write_polydisperse(self, data_dict, n_phases, phase_indepen_field_names):

        """
        Writes polydisperse data to a VTKHDF file, handling both phase-independent and phase-dependent fields.  
        
        Parameters
        ----------
        data_dict : dict  
            Dictionary containing data to be written, where keys are variable names and values are numpy arrays.
            The data arrays should be structured as follows:
            - Phase-independent fields: 1D, 2D, or 3D arrays (shape: (n_nodes,) or (n_nodes, 3) or (n_nodes, 3, 3))
            - Phase-dependent fields: 2D or 3D arrays with shape (n_nodes, n_phases, ...) for each phase (including
               the bulk!)
        n_phases : int
            Number of phases in the polydisperse system.
        phase_indepen_field_names : list
            List of field names that are independent of the phase. These fields will be written once for all phases.    
        
        Notes
        -----
        This method creates a pyvista ImageData object with the specified dimensions and spacing,
        iterates over the data dictionary, and writes each field to the ImageData object.
        It handles both phase-independent fields (written once) and phase-dependent fields (written for each phase).
        The resulting data is then written to a VTKHDF file with gzip compression.

        """

        box = pv.ImageData(dimensions=self.node_dimensions, spacing=self.node_spacing, origin=self.origin)
        suffixes = ['_bulk'] + [f"_phase{i}" for i in range(1, n_phases)] # nphases should include the bulk
        for key, value in data_dict.items(): 
            if key in phase_indepen_field_names: # phase-independent fields
                if value is not None:
                    dims = value.ndim
                    if dims == 1:
                        data = value.reshape(*self.node_dimensions, order='C')
                        v5i.set_point_scalar(box, data, key)
                    elif dims == 2:
                        data = value.reshape((*self.node_dimensions, 3), order='C')
                        v5i.set_point_vector(box, data, key)
                    elif dims == 3:
                        ncomp = len(value[0,...].flatten(order='C'))
                        data = value.reshape((*self.node_dimensions, ncomp), order='C')
                        v5i.set_point_tensor(box, data, key, components=ncomp)
                    else:
                        print(f"{key}: unknown type, shape = {value.shape}")
            else: # phase-dependent fields
                if value is not None:
                    for p in range(n_phases):
                        value_phase = value[:, p, ...]
                        dims = value_phase.ndim
                        key_suffix = key + suffixes[p]
                        if dims == 1:
                            data = value_phase.reshape(*self.node_dimensions, order='C')
                            v5i.set_point_scalar(box, data, key_suffix)
                        elif dims == 2:
                            data = value_phase.reshape((*self.node_dimensions, 3), order='C')
                            v5i.set_point_vector(box, data, key_suffix)
                        elif dims == 3:
                            ncomp = len(value_phase[0,...].flatten(order='F'))
                            data = value_phase.reshape((*self.node_dimensions, ncomp), order='C')
                            v5i.set_point_tensor(box, data, key_suffix, components=ncomp)
                        else:
                            print(f"{key}: unknown type, shape = {value_phase.shape}")

        filename = self.path + ".vtkhdf"
        with h5py.File(filename, "w") as f:
            v5i.write_vtkhdf(f, box, compression='gzip', compression_opts=4)

        print(f"File successfully written to {filename}")

