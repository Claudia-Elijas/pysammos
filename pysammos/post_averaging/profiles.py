"""
This module provides functionality to perform vertical integration of variables
in an xarray.Dataset along a specified profile dimension (x, y, or z).
The integration is done by grouping data points based on their coordinate values
along the profile dimension and summing the values of variables that have a 'point'
dimension, each point weighted by the trapezoid rule over the other two spatial
dimensions (spacing over extent, half of it at the two ends of each range), so
that the result is the mean over the area of the grid.
On a periodic axis whose grid includes both faces of the box, the two end points are
the same point, and the trapezoid rule gives every point of one period the same weight.

This module contains the `VerticalIntegrator` class with methods to:
    - :func:`get_position_data` to extract position data for the specified profile dimension.
    - :func:`get_area_factor` to calculate the area element factor for normalization.
    - :func:`get_point_weights` to calculate the trapezoid weight of each point.
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
        self.point_weights = self.get_point_weights()

    def get_position_data(self):
        
        if self.profile_dim == 'y':
            dim_prof = self.ds['positions'].sel(xyz=1) # y
            dim_1 = self.ds['positions'].sel(xyz=0) # x
            dim_2 = self.ds['positions'].sel(xyz=2) # z
        elif self.profile_dim == 'x':
            dim_prof = self.ds['positions'].sel(xyz=0) # x
            dim_1 = self.ds['positions'].sel(xyz=1) # y
            dim_2 = self.ds['positions'].sel(xyz=2) # z
        elif self.profile_dim == 'z':
            dim_prof = self.ds['positions'].sel(xyz=2) # z
            dim_1 = self.ds['positions'].sel(xyz=0) # x
            dim_2 = self.ds['positions'].sel(xyz=1) # y
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

        factors = []
        for coords in (self.coords_1, self.coords_2):
            unique = np.unique(coords) # get unique coordinate values
            if unique.size < 2:
                continue  # collapsed dimension → ignore
            d = np.mean(np.diff(unique)) # average spacing between unique coordinates
            delta = unique.max() - unique.min() # total extent of the coordinate
            factors.append(d / delta) # factor for this dimension
        area_element = np.prod(factors) if factors else 1.0 # default to 1.0 if no factors

        return area_element


    # trapezoid weights of the points
    def get_point_weights(self) -> xr.DataArray:
        '''

        Calculate the weight of each point in the integration over the two dimensions other
        than the profile dimension: along each of them, the spacing over the extent of the
        coordinate, halved at the smallest and largest coordinate (trapezoid rule), for a grid of
        uniform spacing as Pysammos generates it. The weights of the points of one profile value
        sum to 1, so the integral is the mean over the area.
        Summing every point with the full weight instead (the factor of :func:`get_area_factor`)
        counts the two ends of each range in full, which raises the mean of a uniform field by
        n/(n-1) along each dimension of n points.

        Outputs
        -------
        point_weights : xarray.DataArray
            Weight of each point, along the 'point' dimension.

        '''
        weights = np.ones(self.profile_points.shape[0])
        for coords in (self.coords_1, self.coords_2):
            values = np.asarray(coords)
            unique = np.unique(values) # get unique coordinate values
            if unique.size < 2:
                continue  # collapsed dimension → ignore
            d = np.mean(np.diff(unique)) # average spacing between unique coordinates
            delta = unique.max() - unique.min() # total extent of the coordinate
            w = np.full(values.shape, d / delta)
            ends = np.isclose(values, unique.min(), rtol=0.0, atol=1e-6 * d) | \
                   np.isclose(values, unique.max(), rtol=0.0, atol=1e-6 * d)
            w[ends] *= 0.5
            weights *= w
        return xr.DataArray(weights, dims=("point",))

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

        # Build the GroupBy over the selected variables, each point with its trapezoid weight
        gb = (ds_sel * self.point_weights).groupby(self.profile_dim)

        # Sum within each group (reduces the grouped 'point' dim): the mean over the area
        out = gb.sum()

        print(f"Completed integration along profile dimension: {self.profile_dim}")

        return out
