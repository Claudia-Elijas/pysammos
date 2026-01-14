
"""
This module provides the :class:`VTKHDFWriter` class, which is used to write data to VTKHDF files.
It also includes functionality to set the origin and dimensions of the data grid.
It is particularly useful for visualizing simulation data in a format compatible with VTK and HDF5 standards.

This module contains the following class:
    1. :class:`VTKHDFWriter`
    This class manages the writing of VTKHDF files with specified node dimensions, spacing, and origin.
    It provides methods to write scalar, vector, and tensor data, and can handle both single-phase and polydisperse data.
    It uses `pyvista` for creating the VTK data structure and `h5py` for writing the data to HDF5 files.
    It is designed to ensure that the data is stored in a format that can be easily read and visualized using VTK-compatible tools. 

    Methods:
        - :meth:`write`: Writes the provided data dictionary to a VTKHDF file, handling different data shapes (scalar, vector, tensor).
        - :meth:`write_polydisperse`: Writes polydisperse data to a VTKHDF file, handling both phase-independent and phase-dependent fields.

"""

# importing necessary modules and libraries
from . import core as v5i
import h5py
import pyvista as pv
from typing import Tuple
import numpy as np


class VTKHDFWriter:

    """
        This class manages the writing of VTKHDF files with specified node dimensions, spacing, and origin.
        It provides methods to write scalar, vector, and tensor data, and can handle both single-phase and polydisperse data.
        
        Inputs
        ------
        node_dimensions : tuple
            Dimensions of the grid in number of nodes (e.g., (nx, ny, nz)).
        node_spacing : tuple
            Spacing between nodes in each dimension (e.g., (dx, dy, dz)).
        origin : tuple
            Origin of the grid in the coordinate system (e.g., (ox, oy, oz)).
        path : str
            Path where the VTKHDF file will be saved.   
        
        Outputs
        -------
        node_dimensions : tuple
            Dimensions of the grid in number of nodes.
        node_spacing : tuple
            Spacing between nodes in each dimension.
        path : str
            Path where the VTKHDF file will be saved.
        origin : tuple
            Origin of the grid in the coordinate system.

        Examples
        --------
        >>> writer = VTKHDFWriter((100, 100, 100), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "output/data")
        >>> writer.write(data_dict)
        >>> writer.write_polydisperse(data_dict, n_phases=3, phase_indepen_field_names=['velocity', 'pressure'])
        This initializes a VTKHDFWriter with a 100x100x100 grid, 1.0 spacing, origin at (0,0,0),
        and saves the file to "output/data.vtkhdf". It then writes data to the file.
        

        """
    def __init__(self, node_dimensions:Tuple, node_spacing:Tuple, origin:Tuple, path:str):
        
        # --- Store attributes ---
        self.node_dimensions = tuple(node_dimensions)
        self.node_spacing = tuple(node_spacing)
        self.path = path
        self.origin = origin #v5i.origin_of_centered_image(self.node_dimensions, self.node_spacing, 2)
        
        # --- Detect and handle 2D ImageData ---
        self.x_is_2d = self.node_dimensions[0] == 1 # Detect 2D ImageData (nx == 1)
        self.y_is_2d = self.node_dimensions[1] == 1 # Detect 2D ImageData (ny == 1)
        self.z_is_2d = self.node_dimensions[2] == 1 # Detect 2D ImageData (nz == 1)
        if self.x_is_2d or self.y_is_2d or self.z_is_2d:
            self.is_2d = True # Flag indicating 2D data
            self._promote_2d_to_thin_3d() # Promote 2D to thin 3D for VTK compatibility with ParaView

    def _promote_2d_to_thin_3d(self):

        """
        Promote 2D ImageData to thin 3D by adding an extra slice in the collapsed dimension.
        
        """

        # Add one extra slice in direction of 2D and ensure non-zero spacing in that direction
        self.node_dimensions = list(self.node_dimensions)
        self.node_spacing = list(self.node_spacing)
        
        ds = 1e-8  # small spacing value to avoid zero spacing

        if self.x_is_2d:
            print("  Warning: VTKHDFWriter promoting 2D ImageData (nx=1) to thin 3D (nx=2) for VTK compatibility.")
            self.node_dimensions[0] = 2
            self.node_spacing = [ds] + self.node_spacing # add a small spacing to the node spacing in X
        if self.y_is_2d:
            print("  Warning: VTKHDFWriter promoting 2D ImageData (ny=1) to thin 3D (ny=2) for VTK compatibility.")
            self.node_dimensions[1] = 2
            self.node_spacing = self.node_spacing[:1] + [ds] + self.node_spacing[1:] # add a small spacing to the node spacing in Y
        if self.z_is_2d:
            print("  Warning: VTKHDFWriter promoting 2D ImageData (nz=1) to thin 3D (nz=2) for VTK compatibility.")
            self.node_dimensions[2] = 2
            self.node_spacing = self.node_spacing[:2] + [ds] # add a small spacing to the node spacing in Z

        # convert back to tuples
        self.node_dimensions = tuple(self.node_dimensions)
        self.node_spacing = tuple(self.node_spacing)   
    
    def _promote_2d_point_data(self, value):
        """
        Promote flattened 2D point data (scalar, vector, or tensor)
        to thin 3D by duplicating the collapsed spatial dimension.

        Inputs
        ------
        value : np.ndarray
            Numpy array containing the point data to be promoted.
        Outputs
        -------
        np.ndarray
            Numpy array with the promoted point data.
        Notes
        ------
        This method handles the promotion of 2D point data to thin 3D by duplicating the collapsed spatial dimension.
        It supports scalar, vector, and tensor data formats. 
        The input data is expected to be in a flattened format, and the output will have the appropriate shape for thin 3D representation.
        2D data is identified based on the node dimensions of the VTKHDFWriter instance.
        The output array will have the same number of components as the input array, but with the spatial dimensions adjusted for thin 3D.
        2D data is identified based on the node dimensions of the VTKHDFWriter instance.

        """

        # promoted (3D) dimensions
        nx, ny, nz = self.node_dimensions  

        # original (pre-promotion) dimensions
        ox = 1 if self.x_is_2d else nx
        oy = 1 if self.y_is_2d else ny
        oz = 1 if self.z_is_2d else nz

        # ---- separate spatial and component dimensions ----
        if value.ndim == 1:
            # scalar
            ncomp = None
            spatial_shape = (ox, oy, oz)
            arr_resh = value.reshape(spatial_shape, order="C")

        elif value.ndim == 2:
            # vector or flattened tensor
            ncomp = value.shape[1]
            spatial_shape = (ox, oy, oz, ncomp)
            arr_resh = value.reshape(spatial_shape, order="C")

        elif value.ndim == 3:
            # true tensor (e.g. 3x3)
            ncomp = value.shape[1] * value.shape[2]
            spatial_shape = (ox, oy, oz, ncomp)
            arr_resh = value.reshape(spatial_shape, order="C") # reshape to (ox, oy, oz, ncomp)

        else:
            raise ValueError(f"Unsupported value shape: {value.shape}")

        # ---- duplicate along collapsed axis ----
        if self.x_is_2d:
            arr_rep = np.repeat(arr_resh, 2, axis=0)
        elif self.y_is_2d:
            arr_rep = np.repeat(arr_resh, 2, axis=1)
        elif self.z_is_2d:
            arr_rep = np.repeat(arr_resh, 2, axis=2)

        # ---- flatten spatial dimensions only ----
        if value.ndim == 1:
            return arr_rep.reshape(-1, order="C")
        elif value.ndim == 2:
            return arr_rep.reshape(-1, arr_resh.shape[-1], order="C")
        elif value.ndim == 3:
            return arr_rep.reshape(-1, arr_resh.shape[-2], arr_resh.shape[-1], order="C")



    def write(self, data_dict:dict):

        """
        Writes the provided data dictionary to a VTKHDF file, handling various data shapes (scalar, vector, tensor).
        
        Inputs
        ------
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

                # Handle 2D data promotion
                if self.is_2d:
                    value = self._promote_2d_point_data(value)

                dims = value.ndim # Get the number of dimensions of the data array
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
        
        Inputs
        ------
        data_dict : dict  
            Dictionary containing data to be written, where keys are variable names and values are numpy arrays.
            The data arrays should be structured as follows:

                - Phase-independent fields: 1D, 2D, or 3D arrays (shape: (n_nodes,) or (n_nodes, 3) or (n_nodes, 3, 3))
                - Phase-dependent fields: 2D or 3D arrays with shape (n_nodes, n_phases, ...) for each phase (including the bulk!)

        n_phases : int
            Number of phases in the polydisperse system.
        phase_indepen_field_names : list
            List of field names that are independent of the phase. These fields will be written once for all phases.    
        
        Notes
        ------
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

                    # Handle 2D data promotion
                    if self.is_2d:
                        value = self._promote_2d_point_data(value)
                    
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

                        # Handle 2D data promotion
                        if self.is_2d:
                            value_phase = self._promote_2d_point_data(value_phase)

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

