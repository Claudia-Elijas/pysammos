r"""
Periodic boundaries for the grid-particle search.

In a periodic domain, a grid node within the cutoff :math:`c` of a periodic face also has neighbours
on the opposite side of the domain. This module provides:

    - :func:`resolve_simulation_box`: the limits of the DEM simulation box given by the user, checked against
      the periodic axes.
    - :func:`check_particles_in_box`: a warning if particles lie outside the box on a periodic axis.
    - :func:`periodic_images`: the particle positions extended with their periodic images that
      lie within a margin of the box, and the index of the original particle for every position.

The images are **positions only**: after the neighbour search, matches are mapped back to the original
particles with the returned index array, so all particle properties (mass, velocity, ...) come from the
original particle, while the displacement to the grid node is measured to the closest periodic image.
"""
import itertools
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np

AXES = ("x", "y", "z")



def resolve_simulation_box(box_ranges: Optional[Sequence], periodic: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Limits of the DEM simulation box, as given by the user.

    The limits are required on every periodic axis: they set the period of the domain. On the other axes
    they are not used and may be None.

    Inputs
    ------
    box_ranges : sequence of 3 entries, or None
        ``[(xmin, xmax), (ymin, ymax), (zmin, zmax)]``, with None for an axis whose limits are not given.
        None for all three if no limits are given.
    periodic : sequence of 3 bool
        Periodicity of the x, y and z axes.

    Outputs
    -------
    box_lo, box_hi : np.ndarray, shape (3,)
        Lower and upper limits of the box; -inf and +inf on axes whose limits are not given.

    Raises
    ------
    ValueError
        If the limits of a periodic axis are missing, or if a lower limit is not below the upper limit.
    """
    periodic = np.asarray(periodic, dtype=bool)
    box_ranges = [None, None, None] if box_ranges is None else list(box_ranges)
    if len(box_ranges) != 3:
        raise ValueError(f"The simulation box needs 3 entries (x, y, z), got {len(box_ranges)}.")

    box_lo = np.full(3, -np.inf)
    box_hi = np.full(3, np.inf)
    missing = []
    for i, (axis, limits) in enumerate(zip(AXES, box_ranges)):
        if limits is None:
            if periodic[i]:
                missing.append(axis)
            continue
        lo, hi = (float(v) for v in limits)
        if not lo < hi:
            raise ValueError(f"Invalid simulation box on the {axis} axis: ({lo}, {hi}); the minimum must be below the maximum.")
        box_lo[i], box_hi[i] = lo, hi
    if missing:
        raise ValueError(
            f"The {', '.join(missing)} axis is periodic, so the limits of the simulation box are needed on it: "
            f"set {', '.join(a + '_range' for a in missing)} in the [simulation_box] section of config.ini "
            f"(or the simulation_box argument of CoarseGraining).")
    return box_lo, box_hi


def check_particles_in_box(data_bounds: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray,
                           periodic: Sequence[bool], tolerance: float) -> None:
    """
    Warn if particles lie outside the simulation box on a periodic axis, which usually means that the
    box limits given by the user do not match the simulation.

    Inputs
    ------
    data_bounds : np.ndarray, shape (3, 2)
        Range of the particle positions at this time step.
    box_lo, box_hi : np.ndarray, shape (3,)
        Limits of the simulation box.
    periodic : sequence of 3 bool
        Periodicity of the x, y and z axes.
    tolerance : float
        Distance outside the box that is still accepted (e.g. the largest particle diameter).
    """
    bounds = np.asarray(data_bounds, dtype=np.float64).reshape(3, 2)
    for i, axis in enumerate(AXES):
        if periodic[i] and (bounds[i, 0] < box_lo[i] - tolerance or bounds[i, 1] > box_hi[i] + tolerance):
            warnings.warn(
                f"Particles lie outside the simulation box on the periodic {axis} axis: particle range "
                f"[{bounds[i, 0]:.6g}, {bounds[i, 1]:.6g}], box [{box_lo[i]:.6g}, {box_hi[i]:.6g}]. "
                f"Check {axis}_range in [simulation_box].", RuntimeWarning, stacklevel=2)


def periodic_images(positions: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray,
                    periodic: Sequence[bool], margin: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Extend the positions with their periodic images that lie within ``margin`` of the box.

    Every position is shifted by :math:`\mathbf{s} \cdot \mathbf{L}`, for all combinations of
    :math:`s_i \in \{-1, 0, 1\}` on the periodic axes (:math:`s_i = 0` on the others), and kept if it
    lies within ``[box_lo - margin, box_hi + margin]`` on every axis. With one periodic axis this gives
    the images across its two faces; with two or three, also the images across edges and corners.

    Inputs
    ------
    positions : np.ndarray, shape (N, 3)
        Particle (or contact) positions.
    box_lo, box_hi : np.ndarray, shape (3,)
        Limits of the simulation box (see :func:`resolve_simulation_box`); may be infinite on non-periodic axes.
    periodic : sequence of 3 bool
        Periodicity of the x, y and z axes.
    margin : float
        Distance from the box within which images are kept: the cutoff :math:`c` of the kernel.
        Must be smaller than the box length on every periodic axis.

    Outputs
    -------
    positions_ext : np.ndarray, shape (N + N_images, 3)
        The original positions followed by the images, same dtype as ``positions``.
    original_index : np.ndarray, shape (N + N_images,)
        Index of the original position for every row of ``positions_ext``
        (``0 ... N-1`` for the originals).

    Notes
    -----
    If no axis is periodic, the input positions are returned unchanged (no copy).
    """
    periodic = np.asarray(periodic, dtype=bool)
    n = positions.shape[0]
    if not periodic.any():
        return positions, np.arange(n)

    box_lo = np.asarray(box_lo, dtype=np.float64)
    box_hi = np.asarray(box_hi, dtype=np.float64)
    L = np.where(periodic, box_hi - box_lo, 0.0)       # period; 0 on non-periodic axes (never shifted)
    if not np.all(np.isfinite(L)):
        raise ValueError("The simulation box limits must be finite on every periodic axis.")
    if np.any(L[periodic] <= margin):
        raise ValueError(f"The box length on every periodic axis ({L[periodic]}) must be larger "
                         f"than the cutoff ({margin}).")

    ext_pos, ext_idx = [positions], [np.arange(n)]
    for shift in itertools.product(*[(-1, 0, 1) if p else (0,) for p in periodic]):
        if not any(shift):
            continue                                    # (0, 0, 0): the originals
        shifted = positions + np.asarray(shift, dtype=np.float64) * L
        keep = np.all((shifted >= box_lo - margin) & (shifted <= box_hi + margin), axis=1)
        if keep.any():
            ext_pos.append(shifted[keep])
            ext_idx.append(np.nonzero(keep)[0])

    positions_ext = np.ascontiguousarray(np.concatenate(ext_pos), dtype=positions.dtype)
    return positions_ext, np.concatenate(ext_idx)