import vtkhdf_enhanced3_flexible as v5i
import h5py
import pyvista as pv
import numpy as np
import xarray as xr
import os


class VTKHDFWriter:
    def __init__(self, node_dimensions, node_spacing, origin, path):
        self.node_dimensions = tuple(node_dimensions)
        self.node_spacing = tuple(node_spacing)
        self.path = path
        self.origin = origin #v5i.origin_of_centered_image(self.node_dimensions, self.node_spacing, 2)
        
    def write(self, data_dict):
        box = pv.ImageData(dimensions=self.node_dimensions, spacing=self.node_spacing, origin=self.origin)

        for key, value in data_dict.items():
            #print(f"saving {key}")
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

        filename = self.path + ".vtkhdf"
        with h5py.File(filename, "w") as f:
            v5i.write_vtkhdf(f, box, compression='gzip', compression_opts=4)
        print(f"File successfully written to {filename}")
        
    def write_polydisperse(self, data_dict, n_phases, phase_indepen_field_names):
        box = pv.ImageData(dimensions=self.node_dimensions, spacing=self.node_spacing, origin=self.origin)
        suffixes = ['_bulk'] + [f"_phase{i}" for i in range(1, n_phases)] # nphases should include hte bulk
        for key, value in data_dict.items():
            if key in phase_indepen_field_names:
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
            else:
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
                            #box.point_data[key_suffix] = value_phase.reshape(-1, 9, order='F')
                            ncomp = len(value_phase[0,...].flatten(order='F'))
                            data = value_phase.reshape((*self.node_dimensions, ncomp), order='C')
                            v5i.set_point_tensor(box, data, key_suffix, components=ncomp)
                        else:
                            print(f"{key}: unknown type, shape = {value_phase.shape}")

        filename = self.path + ".vtkhdf"
        with h5py.File(filename, "w") as f:
            v5i.write_vtkhdf(f, box, compression='gzip', compression_opts=4)

        print(f"File successfully written to {filename}")

class H5XarrayManager:
    """
    Manager for reading and writing HDF5 files with array data and converting to xarray Datasets.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to read/write.
    """

    def __init__(self, filename):
        """
        Initialize the H5XarrayManager.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to manage.
        """
        self.filename = filename

    def add_positions(self, positions):
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

    def add_phases(self, phase_labels):
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

    def update_h5py_file(self, data_dict, dim_index, dim_value, dim_name="time"):
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
        print(f"File successfully updated to {self.filename}")

    def h5_to_xarray(self, dim_name="time"):
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