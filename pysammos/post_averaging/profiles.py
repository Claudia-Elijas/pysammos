"""
This module provides functionality to perform vertical integration of variables
in an xarray.Dataset along a specified profile dimension (x, y, or z).
The integration is done by grouping data points based on their coordinate values
along the profile dimension and summing the values of variables that have a 'point'
dimension. The summed values are then normalized by an area element factor calculated
from the spacing and extent of the other two spatial dimensions.

This module contains the `VerticalIntegrator` class with methods to:
    - :func:`get_position_data` to extract position data for the specified profile dimension.
    - :func:`get_area_factor` to calculate the area element factor for normalization.
    - :func:`integration` to perform the vertical integration of relevant variables in the dataset.

"""



import xarray as xr
import numpy as np



class VerticalIntegrator:
    def __init__(self, ds: xr.Dataset, profile_dim: str):
        self.ds = ds
        self.profile_dim = profile_dim
        self.profile_points, self.coords_1, self.coords_2 = self.get_position_data()
        self.area_element = self.get_area_factor()

    def get_position_data(self):
        
        if self.profile_dim == 'y':
            dim_prof = self.ds['positions'].sel(xyz=1)
            dim_1 = self.ds['positions'].sel(xyz=0)
            dim_2 = self.ds['positions'].sel(xyz=2)
        elif self.profile_dim == 'x':
            dim_prof = self.ds['positions'].sel(xyz=0)
            dim_1 = self.ds['positions'].sel(xyz=1)
            dim_2 = self.ds['positions'].sel(xyz=2)
        elif self.profile_dim == 'z':
            dim_prof = self.ds['positions'].sel(xyz=2)
            dim_1 = self.ds['positions'].sel(xyz=0)
            dim_2 = self.ds['positions'].sel(xyz=1)
        else:
            raise ValueError("profile_dim must be one of 'x', 'y', or 'z'")
        
        return dim_prof, dim_1, dim_2
        
        
    # get area factor: general version
    def get_area_factor(self): 
        '''

        Calculate area element factor for normalization.
        Inputs
        ------
        coords_1 : xarray.DataArray
            First coordinate array of coarse-grained data.
        coords_2 : xarray.DataArray
            Second coordinate array of coarse-grained data.
        Outputs
        -------
        area_element : float
            Area element factor for normalization.

        '''

        # Get spacing in each direction
        d_1 = np.mean(np.diff(np.unique(self.coords_1)))
        d_2 = np.mean(np.diff(np.unique(self.coords_2)))

        # Get domain size 
        delta_1 = self.coords_1.max().item() - self.coords_1.min().item()
        delta_2 = self.coords_2.max().item() - self.coords_2.min().item()

        # Calculate area element
        area_element = d_1 * d_2 / delta_1 / delta_2

        return area_element


    # general vertical integrate function
    def integration(self) -> xr.Dataset:
        """
        Integrate all variables with a 'point' dimension by grouping along the y coordinate
        (either exact values or bins) and summing over the grouped 'point's, then multiply by area.

        Inputs
        ------
        ds : xr.Dataset
            Input dataset containing variables, some of which have a 'point' dim.
        area_element : float
            Area element to multiply the integrated sums by.
        profile_dim : str
            Name of the profile dimension (e.g., 'y', 'z', 'x').

        Outputs
        -------
        xr.Dataset
            Dataset with variables (those that had 'point') integrated over vertical groups.
            Group dimension will be 'y' (exact).
        """

        print(f"Starting integration along profile dimension: {self.profile_dim}")

        # Ensure profile_dim is attached as a coordinate on the 'point' dimension
        ds = self.ds.assign_coords({self.profile_dim: ("point", self.profile_points.data)})

        # Select only variables that actually have the 'point' dimension
        vars_with_point = [name for name, var in ds.data_vars.items() if 'point' in var.dims]
        ds_sel = ds[vars_with_point]

        # Build the GroupBy over the selected variables
        gb = ds_sel.groupby(self.profile_dim)

        # Sum within each group (reduces the grouped 'point' dim), then scale by area
        out = gb.sum() * self.area_element

        print(f"Completed integration along profile dimension: {self.profile_dim}")

        return out
