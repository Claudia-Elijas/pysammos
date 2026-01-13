"""
vtkhdf_enhanced3_flexible.py - ECP Breard
Enhanced VTK HDF ImageData format library with full vector and flexible tensor support.
Supports tensors of arbitrary dimensions (6, 9, 27, or any component count).
"""

import h5py
import numpy as np
from numpy.typing import ArrayLike
import pyvista
import vtk
from typing import Union, Tuple, Optional, Dict

# Constants from vtkhdf.image
VTKHDF = "VTKHDF"
IMAGEDATA = "ImageData"
POINTDATA = "PointData"
CELLDATA = "CellData"
FIELDDATA = "FieldData"
VERSION = "Version"
TYPE = "Type"
EXTENT = "WholeExtent"
ORIGIN = "Origin"
SPACING = "Spacing"
DIRECTION = "Direction"
SCALARS = "Scalars"
VECTORS = "Vectors"
TENSORS = "Tensors"  # For standard 9-component tensors
DATA_ORDER = "DataOrder"  # Track orientation

# Common tensor sizes
TENSOR_SYMMETRIC = 6  # Symmetric 3x3: xx, yy, zz, xy, xz, yz
TENSOR_FULL = 9       # Full 3x3: xx, xy, xz, yx, yy, yz, zx, zy, zz
TENSOR_4D = 27        # 3x3x3 tensor

# Utility functions from vtkhdf.image
def c2f_reshape(array: np.ndarray) -> np.ndarray:
    """Convert C-order array to Fortran-order."""
    return np.asfortranarray(array.T)

def f2c_reshape(array: np.ndarray) -> np.ndarray:
    """Convert Fortran-order array to C-order."""
    return np.ascontiguousarray(array.T)

def extent2dimensions(extent: tuple) -> tuple:
    """Convert extent to dimensions."""
    return (extent[1] - extent[0] + 1,
            extent[3] - extent[2] + 1,
            extent[5] - extent[4] + 1)

def dimensions2extent(dimensions: tuple) -> tuple:
    """Convert dimensions to extent."""
    return (0, dimensions[0] - 1,
            0, dimensions[1] - 1,
            0, dimensions[2] - 1)

def point2cell_extent(extent: tuple) -> tuple:
    """Convert point extent to cell extent."""
    return (extent[0], max(extent[0], extent[1] - 1),
            extent[2], max(extent[2], extent[3] - 1),
            extent[4], max(extent[4], extent[5] - 1))

def point2cell_dimensions(dimensions: tuple) -> tuple:
    """Convert point dimensions to cell dimensions."""
    return tuple(max(1, d - 1) for d in dimensions)

def create_field_dataset(h5_file: h5py.File, var: str, data: np.ndarray, **kwargs):
    """Create field dataset."""
    group = h5_file[VTKHDF][FIELDDATA]
    return group.create_dataset(var, data=data, **kwargs)

def get_field_array(image_data: pyvista.ImageData, var: str) -> np.ndarray:
    """Get field array from ImageData."""
    return image_data.field_data[var]

# Enhanced functions
def initialize(file: h5py.File, extent: tuple, origin: tuple = (0, 0, 0),
               spacing: tuple = (1, 1, 1),
               direction: tuple = (1, 0, 0, 0, 1, 0, 0, 0, 1),
               data_order: str = "FortranOrder"):
    """Initialize VTK HDF file structure with orientation tracking."""
    group = file.create_group(VTKHDF)
    group.attrs.create(VERSION, [1, 0])
    group.attrs.create(TYPE, np.bytes_(IMAGEDATA))
    group.attrs.create(EXTENT, extent)
    group.attrs.create(ORIGIN, origin)
    group.attrs.create(SPACING, spacing)
    group.attrs.create(DIRECTION, direction)
    group.attrs.create(DATA_ORDER, np.bytes_(data_order))
    group.create_group(POINTDATA)
    group.create_group(CELLDATA)
    group.create_group(FIELDDATA)


def get_point_data_shape(h5_file: h5py.File) -> tuple:
    """Get point data shape in C order."""
    return extent2dimensions(h5_file[VTKHDF].attrs[EXTENT])[::-1]


def get_cell_data_shape(h5_file: h5py.File) -> tuple:
    """Get cell data shape in C order."""
    return point2cell_dimensions(get_point_data_shape(h5_file))


def is_vector_data(array: np.ndarray, expected_size: int) -> bool:
    """Check if array is vector data (always 3 components)."""
    return (array.ndim == 2 and 
            array.shape[0] == expected_size and 
            array.shape[1] == 3)


def is_tensor_data(array: np.ndarray, expected_size: int, 
                   components: Optional[int] = None) -> Union[bool, int]:
    """
    Check if array is tensor data.
    
    Args:
        array: Input array
        expected_size: Expected number of points/cells
        components: If specified, check for exact component count.
                   If None, return component count if it's tensor data, False otherwise.
    
    Returns:
        If components specified: bool indicating if array matches
        If components is None: component count if tensor-like (>3), False otherwise
    """
    if array.ndim != 2 or array.shape[0] != expected_size:
        return False
    
    n_components = array.shape[1]
    
    if components is not None:
        return n_components == components
    else:
        # Consider anything > 3 components as potential tensor data
        # (3 components are vectors)
        return n_components if n_components > 3 else False


def create_adaptive_chunks(shape_c: tuple, component_size: int = 1, 
                          dtype: np.dtype = np.float64) -> tuple:
    """Create chunk shape adapted to grid asymmetry."""
    nz, ny, nx = shape_c[:3]
    
    # Target chunk size: 1-4 MB
    bytes_per_element = dtype.itemsize * component_size
    target_bytes = 2 * 1024 * 1024  # 2 MB
    max_bytes = 4 * 1024 * 1024     # 4 MB
    
    # Calculate slice size
    slice_bytes = ny * nx * bytes_per_element
    
    if slice_bytes <= max_bytes:
        # Use full XY slices
        chunks = (1, ny, nx)
    else:
        # Need to chunk within slices
        target_elements = target_bytes // bytes_per_element
        
        if nx > ny:
            chunk_nx = min(nx, int(np.sqrt(target_elements)))
            chunk_ny = min(ny, target_elements // chunk_nx)
        else:
            chunk_ny = min(ny, int(np.sqrt(target_elements)))
            chunk_nx = min(nx, target_elements // chunk_ny)
        
        chunk_nx = max(1, chunk_nx)
        chunk_ny = max(1, chunk_ny)
        
        chunks = (1, chunk_ny, chunk_nx)
    
    if component_size > 1:
        chunks = chunks + (component_size,)
    
    return chunks


def create_point_dataset(h5_file: h5py.File, var: str, 
                         is_vector: bool = False,
                         is_tensor: bool = False,
                         tensor_components: Optional[int] = None,
                         **kwargs) -> h5py.Dataset:
    """Create point dataset with adaptive chunking and flexible tensor support."""
    group = h5_file[VTKHDF][POINTDATA]
    shape_c = get_point_data_shape(h5_file)
    dtype = kwargs.get('dtype', np.float64)

    if is_tensor:
        if tensor_components is None:
            tensor_components = TENSOR_FULL  # Default to 9 components
        shape_c = shape_c + (tensor_components,)
        chunk_shape = create_adaptive_chunks(shape_c[:3], tensor_components, dtype)
        
        # Store tensor info in attributes
        if tensor_components == TENSOR_FULL and TENSORS not in group.attrs:
            # Standard 9-component tensor for backward compatibility
            group.attrs.create(TENSORS, np.bytes_(var))
        else:
            # Store component count for non-standard tensors
            attr_name = f"Tensor{tensor_components}"
            if attr_name not in group.attrs:
                group.attrs.create(attr_name, np.bytes_(var))
                
    elif is_vector:
        shape_c = shape_c + (3,)
        chunk_shape = create_adaptive_chunks(shape_c[:3], 3, dtype)
        if VECTORS not in group.attrs:
            group.attrs.create(VECTORS, np.bytes_(var))
    else:
        chunk_shape = create_adaptive_chunks(shape_c, 1, dtype)
        if SCALARS not in group.attrs:
            group.attrs.create(SCALARS, np.bytes_(var))

    return group.create_dataset(
        var, shape=shape_c, dtype=dtype,
        chunks=chunk_shape, **{k: v for k, v in kwargs.items() if k != 'dtype'}
    )


def create_cell_dataset(h5_file: h5py.File, var: str,
                       is_vector: bool = False,
                       is_tensor: bool = False,
                       tensor_components: Optional[int] = None,
                       **kwargs) -> h5py.Dataset:
    """Create cell dataset with adaptive chunking and flexible tensor support."""
    group = h5_file[VTKHDF][CELLDATA]
    shape_c = get_cell_data_shape(h5_file)
    dtype = kwargs.get('dtype', np.float64)
    
    if is_tensor:
        if tensor_components is None:
            tensor_components = TENSOR_FULL
        shape_c = shape_c + (tensor_components,)
        chunk_shape = create_adaptive_chunks(shape_c[:3], tensor_components, dtype)
        
        if tensor_components == TENSOR_FULL and TENSORS not in group.attrs:
            group.attrs.create(TENSORS, np.bytes_(var))
        else:
            attr_name = f"Tensor{tensor_components}"
            if attr_name not in group.attrs:
                group.attrs.create(attr_name, np.bytes_(var))
                
    elif is_vector:
        shape_c = shape_c + (3,)
        chunk_shape = create_adaptive_chunks(shape_c[:3], 3, dtype)
        if VECTORS not in group.attrs:
            group.attrs.create(VECTORS, np.bytes_(var))
    else:
        chunk_shape = create_adaptive_chunks(shape_c, 1, dtype)
        if SCALARS not in group.attrs:
            group.attrs.create(SCALARS, np.bytes_(var))
    
    return group.create_dataset(
        var, shape=shape_c, dtype=dtype,
        chunks=chunk_shape, **{k: v for k, v in kwargs.items() if k != 'dtype'}
    )


def get_data_order(h5_file: h5py.File) -> str:
    """Get data ordering from file metadata."""
    if DATA_ORDER in h5_file[VTKHDF].attrs:
        order = h5_file[VTKHDF].attrs[DATA_ORDER]
        if isinstance(order, bytes):
            order = order.decode()
        return order
    return "FortranOrder"


def write_scalar_slice(dset: h5py.Dataset, array: np.ndarray, index: int):
    """Write scalar data slice."""
    arr_c = f2c_reshape(array) if array.flags.f_contiguous else array
    dset[index, :, :] = arr_c[np.newaxis, :, :]


def write_vector_slice(dset: h5py.Dataset, array: np.ndarray, index: int,
                      data_order: str = "FortranOrder"):
    """Write vector data slice with explicit ordering."""
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected 3D vector array with shape (..., 3), got {array.shape}")
    
    expected_shape = dset.shape[1:3]
    
    if data_order == "FortranOrder":
        arr_to_write = array.transpose(1, 0, 2)
    else:
        arr_to_write = array
    
    if arr_to_write.shape[:2] != expected_shape:
        raise ValueError(
            f"Shape mismatch after reordering: got {arr_to_write.shape[:2]}, "
            f"expected {expected_shape}. Original array shape: {array.shape}, "
            f"data order: {data_order}"
        )
    
    arr_c = np.ascontiguousarray(arr_to_write)
    dset[index, :, :, :] = arr_c


def write_multicomponent_slice(dset: h5py.Dataset, array: np.ndarray, index: int,
                              data_order: str = "FortranOrder"):
    """Write multi-component data slice (vectors or tensors) with explicit ordering."""
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array with shape (..., n_components), got {array.shape}")
    
    n_components = array.shape[-1]
    expected_shape = dset.shape[1:3]
    expected_components = dset.shape[3]
    
    if n_components != expected_components:
        raise ValueError(f"Component mismatch: array has {n_components}, dataset expects {expected_components}")
    
    if data_order == "FortranOrder":
        arr_to_write = array.transpose(1, 0, 2)
    else:
        arr_to_write = array
    
    if arr_to_write.shape[:2] != expected_shape:
        raise ValueError(
            f"Shape mismatch after reordering: got {arr_to_write.shape[:2]}, "
            f"expected {expected_shape}. Original array shape: {array.shape}, "
            f"data order: {data_order}"
        )
    
    arr_c = np.ascontiguousarray(arr_to_write)
    dset[index, :, :, :] = arr_c


def get_point_array(image_data: pyvista.ImageData, var: str):
    """Get point array from ImageData."""
    arr = image_data.point_data[var]
    if arr.ndim == 2 and arr.shape[1] >= 3:  # Vectors or tensors
        return arr
    else:
        # For scalars, reshape to 3D
        return arr.reshape(image_data.dimensions, order='F')


def get_cell_array(image_data: pyvista.ImageData, var: str):
    """Get cell array from ImageData."""
    arr = image_data.cell_data[var]
    if arr.ndim == 2 and arr.shape[1] >= 3:  # Vectors or tensors
        return arr
    else:
        # For scalars, reshape to 3D
        cell_dims = tuple(max(1, d - 1) for d in image_data.dimensions)
        return arr.reshape(cell_dims, order='F')


def write_vtkhdf(h5_file: h5py.File, imagedata,
                 direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                 data_order="FortranOrder",
                 **kwargs): 
    """Write ImageData to VTK HDF format with flexible tensor support."""
    if isinstance(imagedata, vtk.vtkImageData):
        imagedata = pyvista.wrap(imagedata)

    initialize(h5_file, imagedata.extent, origin=imagedata.origin,
               spacing=imagedata.spacing, direction=direction,
               data_order=data_order)

    n_points = imagedata.n_points
    n_cells = imagedata.n_cells

    # Write point data
    for var in imagedata.point_data.keys():
        arr = imagedata.point_data[var]
        
        # Check what type of data this is
        tensor_components = is_tensor_data(arr, n_points)
        
        if tensor_components:  # Returns component count or False
            dset = create_point_dataset(h5_file, var, is_tensor=True, 
                                      tensor_components=tensor_components,
                                      dtype=arr.dtype, **kwargs)
            arr_4d = arr.reshape(*imagedata.dimensions, tensor_components, order='F')
            for k in range(imagedata.dimensions[2]):
                write_multicomponent_slice(dset, arr_4d[:, :, k, :], k, data_order)
                
        elif is_vector_data(arr, n_points):
            dset = create_point_dataset(h5_file, var, is_vector=True, 
                                      dtype=arr.dtype, **kwargs)
            arr_4d = arr.reshape(*imagedata.dimensions, 3, order='F')
            for k in range(imagedata.dimensions[2]):
                write_vector_slice(dset, arr_4d[:, :, k, :], k, data_order)
        else:
            dset = create_point_dataset(h5_file, var, is_vector=False,
                                      dtype=arr.dtype, **kwargs)
            arr_3d = get_point_array(imagedata, var)
            for k in range(imagedata.dimensions[2]):
                write_scalar_slice(dset, arr_3d[:, :, k], k)

    # Write cell data
    cell_dims = extent2dimensions(point2cell_extent(imagedata.extent))
    for var in imagedata.cell_data.keys():
        arr = imagedata.cell_data[var]
        
        tensor_components = is_tensor_data(arr, n_cells)
        
        if tensor_components:
            dset = create_cell_dataset(h5_file, var, is_tensor=True,
                                     tensor_components=tensor_components,
                                     dtype=arr.dtype, **kwargs)
            arr_4d = arr.reshape(*cell_dims, tensor_components, order='F')
            for k in range(cell_dims[2]):
                write_multicomponent_slice(dset, arr_4d[:, :, k, :], k, data_order)
                
        elif is_vector_data(arr, n_cells):
            dset = create_cell_dataset(h5_file, var, is_vector=True,
                                     dtype=arr.dtype, **kwargs)
            arr_4d = arr.reshape(*cell_dims, 3, order='F')
            for k in range(cell_dims[2]):
                write_vector_slice(dset, arr_4d[:, :, k, :], k, data_order)
        else:
            dset = create_cell_dataset(h5_file, var, is_vector=False,
                                     dtype=arr.dtype, **kwargs)
            arr_3d = get_cell_array(imagedata, var)
            for k in range(cell_dims[2]):
                write_scalar_slice(dset, arr_3d[:, :, k], k)

    # Write field data
    for var in imagedata.field_data.keys():
        fdat = get_field_array(imagedata, var)
        create_field_dataset(h5_file, var, data=fdat, **kwargs)


def read_vtkhdf(filename: str):
    """Read VTK HDF file with flexible tensor support."""
    reader = vtk.vtkHDFReader()
    reader.SetFileName(filename)
    reader.Update()
    output = reader.GetOutput()
    
    mesh = pyvista.wrap(output)
    
    # Set vector/tensor metadata
    with h5py.File(filename, 'r') as f:
        data_order = get_data_order(f)
        
        if POINTDATA in f[VTKHDF]:
            pd_group = f[VTKHDF][POINTDATA]
            
            # Check for vectors
            if VECTORS in pd_group.attrs:
                vector_name = pd_group.attrs[VECTORS]
                if isinstance(vector_name, bytes):
                    vector_name = vector_name.decode()
                if vector_name in mesh.point_data:
                    mesh.point_data.set_vectors(mesh.point_data[vector_name], vector_name)
            
            # Check for standard tensors
            if TENSORS in pd_group.attrs:
                tensor_name = pd_group.attrs[TENSORS]
                if isinstance(tensor_name, bytes):
                    tensor_name = tensor_name.decode()
                # Standard 9-component tensors might be recognized by ParaView
                
            # Check for non-standard tensors
            for attr_name in pd_group.attrs:
                if attr_name.startswith('Tensor') and attr_name != TENSORS:
                    var_name = pd_group.attrs[attr_name]
                    if isinstance(var_name, bytes):
                        var_name = var_name.decode()
                    # Non-standard tensors won't be recognized as tensors by ParaView
                    # but the data is preserved
        
        # Similar for cell data
        if CELLDATA in f[VTKHDF]:
            cd_group = f[VTKHDF][CELLDATA]
            if VECTORS in cd_group.attrs:
                vector_name = cd_group.attrs[VECTORS]
                if isinstance(vector_name, bytes):
                    vector_name = vector_name.decode()
                if vector_name in mesh.cell_data:
                    mesh.cell_data.set_vectors(mesh.cell_data[vector_name], vector_name)
    
    return mesh


def set_point_scalar(image_data: pyvista.ImageData, 
                    scalar_array: np.ndarray, var: str):
    """Add scalar field to point data."""
        
    if scalar_array.shape != image_data.dimensions:
        raise ValueError(f"Array dimensions {scalar_array.shape[:-1]} don't match "
                        f"ImageData dimensions {image_data.dimensions}")
    
    image_data.point_data[var] = scalar_array.reshape(-1, order='F')

def set_point_vector(image_data: pyvista.ImageData, 
                    vector_array: np.ndarray, var: str):
    """Add vector field to point data."""
    if vector_array.ndim != 4 or vector_array.shape[-1] != 3:
        raise ValueError(f"Expected 4D array with shape (..., 3), got {vector_array.shape}")
    
    if vector_array.shape[:-1] != image_data.dimensions:
        raise ValueError(f"Array dimensions {vector_array.shape[:-1]} don't match "
                        f"ImageData dimensions {image_data.dimensions}")
    
    image_data.point_data[var] = vector_array.reshape(-1, 3, order='F')


def set_point_tensor(image_data: pyvista.ImageData, 
                     tensor_array: np.ndarray, var: str,
                     components: Optional[int] = None):
    """
    Add tensor field to point data with flexible component count.
    
    Args:
        image_data: PyVista ImageData object
        tensor_array: 4D array with shape (nx, ny, nz, n_components)
        var: Variable name
        components: Expected number of components (for validation).
                   If None, accepts any component count > 3.
    """
    if tensor_array.ndim != 4:
        raise ValueError(f"Expected 4D tensor array, got {tensor_array.ndim}D")
    
    n_components = tensor_array.shape[-1]
    
    if components is not None and n_components != components:
        raise ValueError(f"Expected {components} components, got {n_components}")
    
    if n_components <= 3:
        raise ValueError(f"Tensor must have >3 components, got {n_components}. Use set_point_vector for 3-component data.")
    
    if tensor_array.shape[:-1] != image_data.dimensions:
        raise ValueError(f"Tensor dimensions {tensor_array.shape[:-1]} don't match ImageData dimensions {image_data.dimensions}")
    
    image_data.point_data[var] = tensor_array.reshape(-1, n_components, order='F')


def get_point_vector(image_data: pyvista.ImageData, var: str) -> np.ndarray:
    """Get vector field as 4D array."""
    vec_data = image_data.point_data[var]
    if vec_data.ndim != 2 or vec_data.shape[1] != 3:
        raise ValueError(f"Expected vector data with shape (N, 3), got {vec_data.shape}")
    
    return vec_data.reshape(*image_data.dimensions, 3, order='F')


def get_point_tensor(image_data: pyvista.ImageData, var: str,
                     components: Optional[int] = None) -> np.ndarray:
    """
    Get tensor field as 4D array with flexible component count.
    
    Args:
        image_data: PyVista ImageData object
        var: Variable name
        components: Expected number of components (for validation).
                   If None, accepts any component count > 3.
    """
    tensor_data = image_data.point_data[var]
    
    if tensor_data.ndim != 2:
        raise ValueError(f"Expected 2D tensor data, got {tensor_data.ndim}D")
    
    n_components = tensor_data.shape[1]
    
    if components is not None and n_components != components:
        raise ValueError(f"Expected tensor with {components} components, got {n_components}")
    
    if n_components <= 3:
        raise ValueError(f"Data has only {n_components} components. Use get_point_vector for 3-component data.")
    
    return tensor_data.reshape(*image_data.dimensions, n_components, order='F')


def validate_imagedata_consistency(imagedata: pyvista.ImageData, 
                                 verbose: bool = True) -> Dict[str, any]:
    """Validate ImageData consistency and return diagnostics."""
    diagnostics = {
        'valid': True,
        'issues': [],
        'spacing_ratio': max(imagedata.spacing) / min(imagedata.spacing),
        'dimension_ratio': max(imagedata.dimensions) / min(imagedata.dimensions),
        'dimensions': imagedata.dimensions,
        'spacing': imagedata.spacing,
        'is_square_xy': imagedata.dimensions[0] == imagedata.dimensions[1],
        'data_types': {}
    }
    
    # Check data types
    for var in imagedata.point_data.keys():
        arr = imagedata.point_data[var]
        if arr.ndim == 2:
            components = arr.shape[1]
            if components == 3:
                diagnostics['data_types'][var] = 'vector'
            elif components > 3:
                diagnostics['data_types'][var] = f'tensor{components}'
            else:
                diagnostics['data_types'][var] = 'scalar'
        else:
            diagnostics['data_types'][var] = 'scalar'
    
    # Check spacing ratio
    if diagnostics['spacing_ratio'] > 10:
        diagnostics['issues'].append(f"Large spacing ratio: {diagnostics['spacing_ratio']:.2f}")
        diagnostics['valid'] = False
    
    # Check dimension ratio
    if diagnostics['dimension_ratio'] > 100:
        diagnostics['issues'].append(f"Large dimension ratio: {diagnostics['dimension_ratio']:.2f}")
        diagnostics['valid'] = False
    
    # Check extent consistency
    extent = imagedata.extent
    for i in range(3):
        expected_extent = imagedata.dimensions[i] - 1
        actual_extent = extent[2*i+1] - extent[2*i]
        if actual_extent != expected_extent:
            diagnostics['issues'].append(
                f"Extent mismatch in dimension {i}: expected {expected_extent}, got {actual_extent}"
            )
            diagnostics['valid'] = False
    
    # Warn about square XY
    if diagnostics['is_square_xy']:
        diagnostics['issues'].append("Square XY dimensions - orientation ambiguity possible")
    
    if verbose:
        if diagnostics['issues']:
            print("ImageData validation results:")
            for issue in diagnostics['issues']:
                print(f"  - {issue}")
            print(f"  Recommendation: {'OK to proceed' if diagnostics['valid'] else 'Review data before writing'}")
        
        print("\nData types found:")
        for var, dtype in diagnostics['data_types'].items():
            print(f"  {var}: {dtype}")
    
    return diagnostics


# Convenience functions for common tensor types
def set_point_symmetric_tensor(image_data: pyvista.ImageData,
                              tensor_array: np.ndarray, var: str):
    """Add 6-component symmetric tensor to point data."""
    set_point_tensor(image_data, tensor_array, var, components=TENSOR_SYMMETRIC)


def set_point_4d_tensor(image_data: pyvista.ImageData,
                       tensor_array: np.ndarray, var: str):
    """Add 27-component 3x3x3 tensor to point data."""
    set_point_tensor(image_data, tensor_array, var, components=TENSOR_4D)


# Additional utility functions
def get_point_dataset(h5_file: h5py.File, var: str) -> h5py.Dataset:
    """Get point dataset from HDF5 file."""
    return h5_file[VTKHDF][POINTDATA][var]

def get_cell_dataset(h5_file: h5py.File, var: str) -> h5py.Dataset:
    """Get cell dataset from HDF5 file."""
    return h5_file[VTKHDF][CELLDATA][var]

def get_field_dataset(h5_file: h5py.File, var: str) -> h5py.Dataset:
    """Get field dataset from HDF5 file."""
    return h5_file[VTKHDF][FIELDDATA][var]

def read_slice(h5_file: h5py.File, var: str, index: int, 
               data_type: str = "point") -> np.ndarray:
    """Read a single slice from dataset."""
    if data_type == "point":
        dset = get_point_dataset(h5_file, var)
    elif data_type == "cell":
        dset = get_cell_dataset(h5_file, var)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return dset[index]