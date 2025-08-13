"""
Writer for HDF5 files using h5py and xarray
===========================================

This module provides a manager class for reading and writing HDF5 files containing array data,
and converting them to xarray Datasets. It supports adding grid point positions, phase labels,
and saving data at specific indices with custom dimension values.

It also includes functionality to load the HDF5 file as an xarray Dataset with appropriate
coordinates and dimension names.

It is designed to handle various data shapes, including scalar, vector, and tensor data,
and to ensure unique dimension names for each variable.

It is particularly useful for managing simulation data in a structured format,
allowing for efficient storage and retrieval of large datasets, while providing a convenient
interface for data analysis using xarray.

Class: H5XarrayManager
------------------
This class manages HDF5 files and provides methods to add positions, phases, and update datasets
with new data. It also includes methods to convert the HDF5 file into an xarray Dataset,
handling various data shapes and ensuring appropriate dimension names.
H5XarrayManager(filename: str)
    Initializes the manager with the specified HDF5 file.   
Methods:
- add_positions(positions: array-like): Adds grid point positions to the HDF5 file
- add_phases(phase_labels: list): Adds phase labels to the HDF5 file
- update_h5py_file(data_dict: dict, dim_index: int, dim_value: float, dim_name: str = "time"):
    Saves a single step of data to the HDF5 file at a specific index with a custom dimension value.
- h5_to_xarray(dim_name: str = "time") -> xarray.Dataset:
    Loads the HDF5 file as an xarray Dataset, handling various data shapes and ensuring appropriate dimension names.
"""


# import necessary libraries
import h5py
import numpy as np
import xarray as xr


class H5XarrayManager:
    """
    Manager for reading and writing HDF5 files with array data and converting to xarray Datasets.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to read/write.
    """

    def __init__(self, filename:str):
        """
        Initialize the H5XarrayManager.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to manage.
        """
        self.filename = filename

    def add_positions(self, positions: np.ndarray):
        """
        Add grid point positions to the HDF5 file if not already present.

        Parameters
        ----------
        positions : array-like, shape (n_points, 3)
            Array of grid point positions.
        """
        with h5py.File(self.filename, "a") as f:
            if "positions" not in f:
                f.create_dataset("positions", data=positions)

    def add_phases(self, phase_labels: list):
        """
        Add phase labels to the HDF5 file if not already present.

        Parameters
        ----------
        phase_labels : list or array-like of str
            List of phase label names.
        """
        with h5py.File(self.filename, "a") as f:
            if "phases" not in f:
                f.create_dataset("phases", data=np.array(phase_labels).astype("S"))

    def update_h5py_file(self, data_dict:dict, dim_index:int, dim_value:int, dim_name:str="time"):
        """
        Save a single step of data to an HDF5 file at a specific index, with a custom dimension value.

        Parameters
        ----------
        data_dict : dict
            Dictionary where keys are variable names and values are arrays to store.
        dim_index : int
            The index in the main dimension (e.g., time) to write to.
        dim_value : float or int or str
            The value for the main dimension at this index (e.g., the time value).
        dim_name : str, optional
            The name of the main dimension (default is "time").
        """
        with h5py.File(self.filename, 'a') as f:
            for key, array in data_dict.items():
                array = np.asarray(array)
                # Expand to include dimension at axis 0
                if array.ndim == 0:
                    array = array[np.newaxis]
                else:
                    array = np.expand_dims(array, axis=0)
                if key in f:
                    dset = f[key]
                    if dset.shape[0] <= dim_index:
                        new_shape = (dim_index + 1,) + dset.shape[1:]
                        dset.resize(new_shape)
                    dset[dim_index] = array[0]
                else:
                    maxshape = (None,) + array.shape[1:]
                    shape = (dim_index + 1,) + array.shape[1:]
                    dset = f.create_dataset(
                        key,
                        shape=shape,
                        maxshape=maxshape,
                        chunks=True,
                        compression="gzip"
                    )
                    dset[dim_index] = array[0]

            # Handle the dimension dataset
            if dim_name in f:
                dset = f[dim_name]
                if dset.shape[0] <= dim_index:
                    dset.resize((dim_index + 1,))
                dset[dim_index] = dim_value
            else:
                shape = (dim_index + 1,)
                dset = f.create_dataset(dim_name, shape=shape, maxshape=(None,), chunks=True, dtype=type(dim_value))
                dset[dim_index] = dim_value
        print(f"  File successfully updated to {self.filename}")

    def h5_to_xarray(self, dim_name:str="time"):
        """
        Load the HDF5 file as an xarray Dataset.

        Parameters
        ----------
        dim_name : str, optional
            Name of the main dimension (default is "time").

        Returns
        -------
        ds : xarray.Dataset
            The loaded dataset with appropriate coordinates and dimension names.
            - For 2x2 tensors, trailing dimensions are named 'dim1_2D', 'dim2_2D'.
            - For 3x3 tensors, trailing dimensions are named 'dim1_3D', 'dim2_3D'.
            - For other shapes, unique dimension names are generated per variable.
        """
        with h5py.File(self.filename, "r") as f:
            data_vars = {}
            coords = {}
            dim_values = f[dim_name][:]
            coords[dim_name] = dim_values

            # Add positions as a coordinate for the 'point' dimension
            if "positions" in f:
                positions = f["positions"][:]
                Npoints = positions.shape[0]
                coords["point"] = np.arange(Npoints)
                coords["positions"] = (("point", "xyz"), positions)
            else:
                coords["point"] = None

            # Add phases as a coordinate if present
            if "phases" in f:
                phases = f["phases"][:].astype(str)
                coords["phase"] = phases
            else:
                phases = None

            for key in f:
                print(f"Processing key: {key}")
                if key in (dim_name, "positions", "phases"):
                    continue
                data = f[key][:]
                # POLYDISPERSE: Handle (time, point, phase, ...) shape
                if data.ndim >= 3 and phases is not None and data.shape[2] == len(phases):
                    if data.ndim == 3:
                        # Scalar: (time, point, phase)
                        dims = (dim_name, "point", "phase")
                    elif data.ndim == 4 and data.shape[-1] == 3:
                        # Vector: (time, point, phase, 3)
                        dims = (dim_name, "point", "phase", "dim1_3D")
                    elif data.ndim == 5 and data.shape[-2:] == (2, 2):
                        # 2x2 tensor: (time, point, phase, 2, 2)
                        dims = (dim_name, "point", "phase", "dim1_2D", "dim2_2D")
                    elif data.ndim == 5 and data.shape[-2:] == (3, 3):
                        # 3x3 tensor: (time, point, phase, 3, 3)
                        dims = (dim_name, "point", "phase", "dim1_3D", "dim2_3D")
                    else:
                        # Other: add unique trailing dims
                        dims = (dim_name, "point", "phase") + tuple(f"dim_{i}_{key}" for i in range(3, data.ndim))
                # MONODISPERSE: Handle (time, point, ...) shape
                elif data.ndim == 2 and "point" in coords and data.shape[1] == len(coords["point"]): # scalar
                    dims = (dim_name, "point")
                elif data.ndim >= 4 and data.shape[-2:] == (2, 2): # 2x2 tensor
                    dims = (
                        (dim_name, "point")
                        + tuple(f"dim_{i}_{key}" for i in range(2, data.ndim-2))
                        + ("dim1_2D", "dim2_2D")
                    )
                elif data.ndim >= 4 and data.shape[-2:] == (3, 3): # 3x3 tensor
                    dims = (
                        (dim_name, "point")
                        + tuple(f"dim_{i}_{key}" for i in range(2, data.ndim-2))
                        + ("dim1_3D", "dim2_3D")
                    )
                elif data.ndim >= 3 and data.shape[-1] == 3:
                    # For vectors (last dimension is 3)
                    dims = (dim_name, "point") + tuple(f"dim_{i}_{key}" for i in range(2, data.ndim-1)) + ("dim1_3D",)
                else:
                    dims = (dim_name,) + tuple(f"dim_{i}_{key}" for i in range(1, data.ndim))
                
                # Ensure unique dimension names
                data_vars[key] = (dims, data)

        ds = xr.Dataset(data_vars, coords=coords)
        return ds