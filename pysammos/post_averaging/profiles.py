r"""
Post-hoc spatial averages of coarse-grained fields: over a whole region, or as a profile along x, y or z.

The coarse-grained fields are split in primary fields (kernel-weighted sums over particles: volume
fraction, mixture density, momentum density, kinetic, fluctuation and contact tensors, ...) and
derived fields (nonlinear functions of them: velocity, shear rate, pressure, granular temperature,
inertial number, frictional coefficient, ...), see :mod:`pysammos.macroscopic_fields.gridded.derived_fields`.

As for the time averages, only the primary fields are averaged (with the same weights,
:func:`pysammos.time_average.accumulator.summable_terms`), and the derived fields are then computed
from the averages with the same code as for one time step:

- velocity: averaged momentum density over averaged mixture density (Favre average),
- velocity gradient: along a profile, finite differences of the profile velocity; across the averaged
  directions (and for a whole-region average), the slope of a least-squares fit of the grid-node velocities,
  weighted by the mixture density. On a 2D grid the collapsed direction has a zero gradient.
- granular temperature, stresses, pressure, shear rate, inertial number, frictional coefficients: from the above.

Averaging the derived fields themselves ("naive" average) is biased wherever they vary within the
averaged region, e.g. the mean of :math:`1/\sqrt{p}` exceeds :math:`1/\sqrt{\bar p}`. The primary fields
are written by :class:`~pysammos.coarse_graining.CoarseGraining` with ``export_primaries=True`` (default);
for files without them the naive average is only computed if the user agrees.

Note that the kinetic and fluctuation tensors at each grid node are measured relative to the velocity
at that node: the spatial variation of the coarse-grained velocity within the averaged region
(:math:`\rho (u - \bar{u}) \otimes (u - \bar{u})`) is not included in the averaged stress or temperature.

This module contains:
    - :class:`SpatialAverager` with :meth:`~SpatialAverager.mean` (one value per time step over a region)
      and :meth:`~SpatialAverager.profile` (profile along an axis, within a region),
    - :class:`VerticalIntegrator`, the former interface (a profile over the whole grid).

Example
-------
>>> ds = H5XarrayManager("CG_Lucy_Monodisperse.h5").h5_to_xarray()
>>> averager = SpatialAverager(ds, region={"x": (0.1, 0.3)})
>>> prof = averager.profile("y")       # profile along y of the region 0.1 <= x <= 0.3
>>> bulk = averager.mean()             # one value per time step over the region
"""

import warnings

import numpy as np
import xarray as xr

from ..macroscopic_fields.field_dependencies import get_fields_to_compute
from ..macroscopic_fields.gridded import derived_fields
from ..macroscopic_fields.gridded.derived_fields import EXPORT_GROUPS, PRIMARY_FIELDS, PHASE_INDEPENDENT_FIELDS
from ..time_average.accumulator import ARITHMETIC_FIELDS, summable_terms, finalise

AXES = ("x", "y", "z")

#: Primary fields stored in the .h5 output (fluctuation_weight only in time averages).
STORED_PRIMARIES = tuple(PRIMARY_FIELDS) + ("fluctuation_weight",)

#: Fields needed as weights by every spatial average, except of the arithmetic primaries.
WEIGHT_FIELDS = {"volume_fraction", "density_mixture"}


class SpatialAverager:
    """
    Spatial averages of the coarse-grained fields of an .h5 output, over a region of the grid.

    Inputs
    ------
    ds : xr.Dataset
        Coarse-grained output read with :meth:`H5XarrayManager.h5_to_xarray` (time series or time average).
    region : dict, optional
        Limits of the region to average, e.g. ``{"x": (0.1, 0.3), "z": (None, 0.05)}`` (inclusive,
        None for no limit). Axes not given are not limited. Default is the whole grid.
    naive : bool or None, optional
        What to do with requested fields whose primary fields are not in ``ds``:
        True: average the fields themselves (biased, listed in the attribute ``naive_average``);
        False: raise an error; None (default): ask the user.
    """

    def __init__(self, ds: xr.Dataset, region: dict = None, naive: bool = None):
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
        self.ds = ds
        self.naive = naive
        self.positions = np.asarray(ds["positions"].values, dtype=np.float64)

        # region mask
        self.region = dict(region or {})
        mask = np.ones(len(self.positions), dtype=bool)
        for axis, limits in self.region.items():
            if axis not in AXES:
                raise ValueError(f"region keys must be 'x', 'y' or 'z', got {axis!r}")
            lo, hi = limits
            coord = self.positions[:, AXES.index(axis)]
            if lo is not None:
                mask &= coord >= lo
            if hi is not None:
                mask &= coord <= hi
        if not mask.any():
            raise ValueError(f"No grid point in the region {self.region}.")
        self.mask = mask

        # information on the coarse graining (attributes written by CoarseGraining, inferred for older files)
        attrs = ds.attrs
        self.cg_calc_mode = attrs.get("cg_calc_mode", "Polydisperse" if "phase" in ds.coords else "Monodisperse")
        self.phases = (np.asarray(attrs["phases_diameter_density"], dtype=np.float64)
                       if "phases_diameter_density" in attrs else None)
        self.grid_dimension = int(attrs.get("grid_dimension",
                                            sum(np.unique(self.positions[:, a]).size > 1 for a in range(3))))
        self.available_primaries = [k for k in STORED_PRIMARIES if k in ds.data_vars]

    # ---------------------------------------------------------------- public
    def mean(self, fields_to_export: dict = None) -> xr.Dataset:
        """
        Average over the whole region: one value per time step.

        Inputs
        ------
        fields_to_export : dict, optional
            Flags as in the ``[fields_to_export]`` config section. Default: every field in ``ds``.

        Outputs
        -------
        xr.Dataset
            Averaged fields, dimensions (time, [phase], ...).
        """
        labels = np.zeros(self.mask.sum(), dtype=int)
        return self._average(labels, None, fields_to_export, name="mean")

    def profile(self, axis: str, fields_to_export: dict = None) -> xr.Dataset:
        """
        Profile along ``axis``: average over the grid points of the region with the same ``axis`` coordinate.

        Inputs
        ------
        axis : str
            Axis of the profile, 'x', 'y' or 'z'.
        fields_to_export : dict, optional
            Flags as in the ``[fields_to_export]`` config section. Default: every field in ``ds``.

        Outputs
        -------
        xr.Dataset
            Averaged fields, dimensions (time, ``axis``, [phase], ...).
        """
        if axis not in AXES:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")
        coord = np.round(self.positions[self.mask, AXES.index(axis)], 10)  # grid planes (round-off safe)
        levels, labels = np.unique(coord, return_inverse=True)
        return self._average(labels, (axis, levels), fields_to_export, name=f"profile_{axis}")

    # ---------------------------------------------------------------- which fields
    def _split_flags(self, fields_to_export):
        """Split the requested flags into those computed from the primaries and those averaged naively."""
        ds = self.ds
        if fields_to_export is None:   # everything that is in the file
            flags = [f for f, keys in EXPORT_GROUPS.items() if any(k in ds.data_vars for k in keys)]
        else:
            flags = [f for f, include in fields_to_export.items() if include and f in EXPORT_GROUPS]

        correct, missing = [], {}
        for flag in flags:
            needed = get_fields_to_compute({flag: True}) & set(PRIMARY_FIELDS)
            if flag not in ARITHMETIC_FIELDS:
                needed |= WEIGHT_FIELDS
            lacking = sorted(needed - set(self.available_primaries))
            if lacking:
                missing[flag] = lacking
            else:
                correct.append(flag)

        naive = []
        if missing:
            unavailable = [f for f in missing if not any(k in ds.data_vars for k in EXPORT_GROUPS[f])]
            message = ("These fields cannot be averaged correctly because their primary fields are not in the file:\n"
                       + "\n".join(f"  {flag}: missing {', '.join(lack)}" for flag, lack in missing.items())
                       + "\nRe-run the coarse graining with export_primaries=True ([output_options] export_primaries = True)"
                         " to average them correctly.")
            if unavailable:
                raise ValueError(message + f"\nNot in the file at all, so they cannot be averaged: {', '.join(unavailable)}.")
            if not self._accept_naive(message):
                raise ValueError(message)
            naive = list(missing)
        return correct, naive

    def _accept_naive(self, message):
        """Whether to average the fields without primaries naively (``naive``, or ask the user)."""
        if self.naive is not None:
            if self.naive:
                warnings.warn(message + "\nThey are averaged naively (biased where they vary in the region).",
                              stacklevel=4)
            return bool(self.naive)
        try:
            answer = input(message + "\nAverage them naively instead (biased where they vary in the region)? [y/N] ")
        except (EOFError, OSError):   # no interactive input (batch job)
            return False
        return answer.strip().lower() in ("y", "yes")

    # ---------------------------------------------------------------- averaging
    def _average(self, labels, profile, fields_to_export, name):
        correct, naive = self._split_flags(fields_to_export)
        ds, mask = self.ds, self.mask
        n_groups = labels.max() + 1
        counts = np.bincount(labels, minlength=n_groups).astype(np.float64)

        def group_mean(values):   # mean over the grid points of each group, first axis = point
            sums = np.zeros((n_groups,) + values.shape[1:])
            np.add.at(sums, labels, values)
            return sums / counts.reshape((-1,) + (1,) * (values.ndim - 1))

        correct_flags = dict.fromkeys(correct, True)
        fields_to_compute = get_fields_to_compute(correct_flags)
        if "inertial_number" in fields_to_compute and self.phases is None:
            if self.cg_calc_mode == "Polydisperse":
                raise ValueError("The phases are not in the file (attribute phases_diameter_density): the polydisperse "
                                 "inertial number cannot be computed. Re-run the coarse graining with the current version.")
        phases = self.phases if self.phases is not None else np.full((1, 2), np.nan)  # unused when Monodisperse
        grid_info = {"grid_dimension": self.grid_dimension}
        naive_keys = [k for flag in naive for k in EXPORT_GROUPS[flag] if k in ds.data_vars]

        results = []
        for t in range(ds.sizes["time"]):
            out = {}
            if correct:
                primary = {k: np.asarray(ds[k].values[t])[mask] for k in self.available_primaries}
                mean_terms = {k: group_mean(v) for k, v in summable_terms(primary).items()}
                averaged = finalise(mean_terms)
                fields = dict(averaged, velocity=None, velocity_gradient=None)
                if "velocity" in fields_to_compute:
                    fields["velocity"] = derived_fields.velocity_from_momentum(averaged["momentum_density"],
                                                                               averaged["density_mixture"])
                if "velocity_gradient" in fields_to_compute:
                    fields["velocity_gradient"] = self._velocity_gradient(primary, labels, profile, fields["velocity"])
                fields = derived_fields.compute_derived_fields(fields, fields_to_compute=fields_to_compute,
                                                               phases=phases, grid_info=grid_info)
                out.update(derived_fields.select_exports(fields, correct_flags))
            for key in naive_keys:
                values = np.asarray(ds[key].values[t], dtype=np.float64)[mask]
                sums = np.zeros((n_groups,) + values.shape[1:])
                weights = np.zeros_like(sums)
                np.add.at(sums, labels, np.nan_to_num(values))
                np.add.at(weights, labels, np.isfinite(values).astype(np.float64))
                out[key] = np.divide(sums, weights, out=np.full_like(sums, np.nan), where=weights > 0)
            results.append(out)

        return self._to_dataset(results, profile, name, naive)

    def _velocity_gradient(self, primary, labels, profile, velocity):
        """
        Gradient of the bulk velocity of each group, shape (Ngroups, 3, 3), [component, direction]:
        along the profile, finite differences of the profile velocity; in the other directions, the slope
        of a least-squares fit of the grid-node velocities weighted by the mixture density.
        """
        rho = primary["density_mixture"]
        rho_bulk = rho if rho.ndim == 1 else rho[:, 0]
        mom = primary["momentum_density"]
        mom_bulk = mom if mom.ndim == 2 else mom[:, 0, :]
        node_velocity = np.divide(mom_bulk, rho_bulk[:, None], out=np.zeros_like(mom_bulk), where=rho_bulk[:, None] > 0)
        positions = self.positions[self.mask]

        n_groups = labels.max() + 1
        gradient = np.zeros((n_groups, 3, 3))
        fit_axes = [a for a in range(3) if profile is None or a != AXES.index(profile[0])]
        for g in range(n_groups):
            in_group = labels == g
            pos, vel, w = positions[in_group], node_velocity[in_group], rho_bulk[in_group]
            axes = [a for a in fit_axes if np.ptp(pos[:, a]) > 0]   # directions with more than one grid plane
            if not axes:
                continue
            if np.count_nonzero(w) <= len(axes):
                gradient[g][:, axes] = np.nan    # too few occupied nodes to fit a gradient
                continue
            centred = pos[:, axes] - np.average(pos[:, axes], axis=0, weights=w)
            design = np.hstack([np.ones((len(pos), 1)), centred]) * np.sqrt(w)[:, None]
            coef = np.linalg.lstsq(design, vel * np.sqrt(w)[:, None], rcond=None)[0]   # (1+len(axes), 3)
            gradient[g][:, axes] = coef[1:].T

        if profile is not None:
            axis, levels = profile
            v_bulk = velocity if velocity.ndim == 2 else velocity[:, 0, :]
            column = np.gradient(v_bulk, levels, axis=0) if len(levels) > 1 else np.full((1, 3), np.nan)
            gradient[:, :, AXES.index(axis)] = column
        return gradient

    # ---------------------------------------------------------------- output
    def _to_dataset(self, results, profile, name, naive):
        ds = self.ds
        phase_dim = "phase" in ds.coords and self.cg_calc_mode == "Polydisperse"
        n_phases = ds.sizes["phase"] if phase_dim else None
        lead = ("time",) + ((profile[0],) if profile is not None else ())

        data_vars = {}
        for key in results[0]:
            values = np.stack([r[key] for r in results])                 # (time, Ngroups, ...)
            if profile is None:
                values = values[:, 0]                                    # (time, ...)
            trailing = list(values.shape[len(lead):])
            dims = []
            if phase_dim and key not in PHASE_INDEPENDENT_FIELDS and trailing and trailing[0] == n_phases:
                dims.append("phase")
                trailing = trailing[1:]
            if trailing == [3]:
                dims += ["dim1_3D"]
            elif trailing == [3, 3]:
                dims += ["dim1_3D", "dim2_3D"]
            elif trailing == [2, 2]:
                dims += ["dim1_2D", "dim2_2D"]
            else:
                dims += [f"dim_{i}_{key}" for i in range(len(trailing))]
            data_vars[key] = (lead + tuple(dims), values)

        coords = {"time": ds["time"].values}
        if profile is not None:
            coords[profile[0]] = profile[1]
        if phase_dim:
            coords["phase"] = ds["phase"].values
        attrs = {k: v for k, v in ds.attrs.items()
                 if k not in ("primary_fields", "derived_fields", "grid_nodes", "grid_spacing")}
        attrs.update({"spatial_average": name,
                      "region": str(self.region or "whole grid"),
                      "naive_average": [k for flag in naive for k in EXPORT_GROUPS[flag] if k in data_vars]})
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)


class VerticalIntegrator(SpatialAverager):
    """
    Profile along ``profile_dim`` over the whole grid (former interface, kept for existing scripts).
    Equivalent to ``SpatialAverager(ds, naive=naive).profile(profile_dim)``.
    """

    def __init__(self, ds: xr.Dataset, profile_dim: str, naive: bool = None):
        super().__init__(ds, naive=naive)
        self.profile_dim = profile_dim

    def integration(self) -> xr.Dataset:
        print(f"Starting integration along profile dimension: {self.profile_dim}")
        out = self.profile(self.profile_dim)
        print(f"Completed integration along profile dimension: {self.profile_dim}")
        return out
