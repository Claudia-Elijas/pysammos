r"""

Running time averages of coarse-grained fields.

Only the primary fields (kernel-weighted sums over particles, see
:data:`pysammos.macroscopic_fields.gridded.derived_fields.PRIMARY_FIELDS`) are averaged in time.
Every derived field (velocity, velocity gradient, stress, pressure, granular temperature, shear
rate, inertial number, frictional coefficient) is then computed from these averages, with the same
code as for a single time step (:func:`~pysammos.macroscopic_fields.gridded.derived_fields.compute_derived_fields`):

- velocity: time-averaged momentum density over time-averaged mixture density (Favre average),
- velocity gradient: finite differences of that velocity on the grid (whatever ``velocity_gradient_method``),
- granular temperature: time-averaged fluctuation tensor over time-averaged mixture density,
- shear rate, pressure, inertial number and frictional coefficients: from the above.

For a steady flow this is the standard procedure (e.g. Kim & Kamrin, 2020, Phys. Rev. Lett. 125,
088002: fields are averaged over time before the rheology is evaluated). Averaging the per-time-step
inertial number, shear rate magnitude or stress ratios instead is biased: the mean of a magnitude
exceeds the magnitude of the mean wherever the instantaneous gradients fluctuate, and the mean of
:math:`1/\sqrt{p}` exceeds :math:`1/\sqrt{\bar{p}}` where the local pressure fluctuates.

Note that the kinetic and fluctuation tensors of each time step are measured relative to the
velocity of that time step: fluctuations of the coarse-grained velocity *between* time steps
(:math:`\rho (u - \bar{u}) \otimes (u - \bar{u})`) are not included in the averaged stress or temperature.

"""

import numpy as np

from ..macroscopic_fields.gridded.derived_fields import PRIMARY_FIELDS

#: Primary fields averaged arithmetically. The fluctuation tensor and the grain sizes need weights.
ARITHMETIC_FIELDS = ("volume_fraction", "density_mixture", "momentum_density", "kinetic_tensor",
                     "contact_tensor", "coordination_number", "fabric_tensor", "sff_moments")


def _safe_divide(a, b):
    """a / b where b > 0, 0 elsewhere."""
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def summable_terms(primary_fields: dict) -> dict:
    """
    Terms of the primary fields that can be summed (over time steps or grid nodes) and then turned
    back into primary fields by :func:`finalise`. Shared by the time averages
    (:class:`TimeAverageAccumulator`) and the post-hoc spatial averages
    (:mod:`pysammos.post_averaging.profiles`), so both use the same weights:

    - arithmetic fields (:data:`ARITHMETIC_FIELDS`) as they are,
    - the fluctuation tensor weighted by the mixture density where it is defined
      (``fluctuation_weight``; at one time step, the mixture density where the tensor is finite),
    - d43 weighted by the bulk volume fraction, d32 as a volume-fraction-weighted harmonic mean.

    Inputs
    ------
    primary_fields : dict
        Primary fields at one time step, or already averaged (then ``fluctuation_weight`` is used if
        given). The first axis is the grid point. ``volume_fraction`` is required for the grain sizes,
        ``density_mixture`` for the fluctuation tensor.

    Outputs
    -------
    terms : dict
        Summable terms, all with the grid point as first axis.
    """
    terms = {}
    for key in ARITHMETIC_FIELDS:
        value = primary_fields.get(key)
        if value is not None:
            terms[key] = np.asarray(value, dtype=np.float64)

    # The fluctuation tensor is undefined where the velocity gradient is (next to empty grid nodes):
    # those samples are left out of the temperature average, at that node only.
    fluct = primary_fields.get("fluctuation_tensor")
    if fluct is not None:
        fluct = np.asarray(fluct, dtype=np.float64)
        rho = np.asarray(primary_fields["density_mixture"], dtype=np.float64)
        defined = np.isfinite(fluct).all(axis=(-2, -1))
        weight = primary_fields.get("fluctuation_weight")
        weight = np.where(defined, rho, 0.0) if weight is None else np.where(defined, weight, 0.0)
        # equals the fluctuation tensor where defined (one time step), or its weighted sum (averages)
        terms["fluctuation_tensor"] = (np.where(defined[..., np.newaxis, np.newaxis], fluct, 0.0)
                                       * _safe_divide(weight, rho)[..., np.newaxis, np.newaxis])
        terms["fluctuation_weight"] = weight

    # Mean grain sizes, weighted by the bulk volume fraction so that the averages are those of
    # sum(w d^4)/sum(w d^3) and sum(w d^3)/sum(w d^2) over all samples.
    if primary_fields.get("volume_fraction") is None:   # only arithmetic fields can be averaged
        return terms
    phi = np.asarray(primary_fields["volume_fraction"], dtype=np.float64)
    phi_bulk = phi if phi.ndim == 1 else phi[:, 0]
    terms["volume_fraction_bulk"] = phi_bulk
    if primary_fields.get("d43") is not None:
        terms["volume_fraction_x_d43"] = phi_bulk * np.nan_to_num(primary_fields["d43"])
    if primary_fields.get("d32") is not None:
        terms["volume_fraction_over_d32"] = _safe_divide(phi_bulk, np.nan_to_num(primary_fields["d32"]))
    return terms


def finalise(mean_terms: dict) -> dict:
    """
    Primary fields from the averages of the terms of :func:`summable_terms` (inverse of it for one sample).

    Outputs
    -------
    fields : dict
        Averaged primary fields, keys :data:`PRIMARY_FIELDS` (None where absent) plus
        ``fluctuation_weight`` (None if no fluctuation tensor), which keeps the averages exact if they
        are averaged again (e.g. a post-hoc spatial average of the time average).
    """
    m = mean_terms
    fields = dict.fromkeys(PRIMARY_FIELDS)
    fields["fluctuation_weight"] = None
    for key in ARITHMETIC_FIELDS:
        if key in m:
            fields[key] = m[key]

    if "fluctuation_tensor" in m:
        # scaled so that fluctuation_tensor / density_mixture (the temperature) is
        # sum(fluctuation tensor) / sum(mixture density) over the samples where it is defined
        scale = _safe_divide(m["density_mixture"], m["fluctuation_weight"])
        fields["fluctuation_tensor"] = m["fluctuation_tensor"] * scale[..., np.newaxis, np.newaxis]
        fields["fluctuation_weight"] = m["fluctuation_weight"]

    phi_bulk = m.get("volume_fraction_bulk")
    if "volume_fraction_x_d43" in m:
        fields["d43"] = _safe_divide(m["volume_fraction_x_d43"], phi_bulk)
    if "volume_fraction_over_d32" in m:
        fields["d32"] = _safe_divide(phi_bulk, m["volume_fraction_over_d32"])
    return fields


class TimeAverageAccumulator:
    """
    Running sums of the primary coarse-grained fields over time steps.

    Usage
    -----
    >>> acc = TimeAverageAccumulator()
    >>> for t in time_steps:
    ...     acc.add(primary_fields_t, t)              # only sums, no physics
    >>> mean_primary = acc.mean()                     # -> derived_fields, as for one time step

    The sliced granular temperature, if computed, is averaged alongside with
    :meth:`add_slices` / :meth:`mean_slices`.
    """

    def __init__(self):
        self._sums = {}
        self._times = []
        self._slice_sums = {}
        self._n_slices = 0
        self.slice_positions = None

    def __len__(self) -> int:
        return len(self._times)

    @property
    def time_steps(self) -> np.ndarray:
        """Time values of the time steps added so far."""
        return np.asarray(self._times, dtype=int)

    @property
    def has_slices(self) -> bool:
        """Whether any sliced granular temperature has been added."""
        return self._n_slices > 0

    # ---------------------------------------------------------------- accumulate
    @staticmethod
    def _accumulate(sums: dict, key: str, value):
        """sums[key] += value (copy on first add, in place afterwards); None is ignored."""
        if value is None:
            return
        value = np.asarray(value, dtype=np.float64)
        if key in sums:
            sums[key] += value
        else:
            sums[key] = value.copy()

    def add(self, primary_fields: dict, time_of_timestep: int):
        """
        Add the primary fields of one time step to the running sums.

        Inputs
        ------
        primary_fields : dict
            Primary fields of the time step (keys :data:`PRIMARY_FIELDS`, None if not computed),
            as returned by :meth:`CoarseGraining._compute_fields`. ``volume_fraction`` is required.
        time_of_timestep : int
            Time value of the time step.
        """
        for key, value in summable_terms(primary_fields).items():
            self._accumulate(self._sums, key, value)
        self._times.append(int(time_of_timestep))

    def add_slices(self, slice_fields: dict, positions: np.ndarray):
        """
        Add the sliced granular temperature of one time step.

        Inputs
        ------
        slice_fields : dict
            Sliced fields of the time step, e.g. granular_temperature_Kamrin, granular_temperature_LAMMPS.
        positions : np.ndarray, shape (Nslices, 3)
            Positions of the slices (stored from the first call).
        """
        if self.slice_positions is None:
            self.slice_positions = positions
        for key, value in slice_fields.items():
            self._accumulate(self._slice_sums, key, value)
        self._n_slices += 1

    # ---------------------------------------------------------------- finalise
    def mean(self) -> dict:
        """
        Time averages of the primary fields.

        Outputs
        -------
        fields : dict
            Averaged primary fields, keys :data:`PRIMARY_FIELDS` (None where never added) plus
            ``fluctuation_weight``, ready for
            :func:`~pysammos.macroscopic_fields.gridded.derived_fields.add_velocity_fields` and
            :func:`~pysammos.macroscopic_fields.gridded.derived_fields.compute_derived_fields`.
        """
        if not self._times:
            raise ValueError("No time step has been added to the time averages.")
        n = len(self._times)
        return finalise({key: value / n for key, value in self._sums.items()})

    def mean_slices(self) -> dict:
        """Time averages of the sliced granular temperature."""
        if not self.has_slices:
            raise ValueError("No sliced granular temperature has been added to the time averages.")
        return {key: value / self._n_slices for key, value in self._slice_sums.items()}