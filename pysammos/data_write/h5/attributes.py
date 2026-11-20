"""
Root attributes of the .h5 output, read by the post-hoc spatial averages
(:class:`pysammos.post_averaging.profiles.SpatialAverager`).
"""

import numpy as np


def output_attributes(results: dict, primary_names, *, export_primary_fields: bool, cg_calc_mode: str,
                      phases, grid_dimension: int, nodes, spacing) -> dict:
    """
    Attributes describing the fields of an .h5 output.

    Inputs
    ------
    results : dict
        Fields written to the file (keys are the dataset names).
    primary_names : iterable of str
        Names of the primary fields (linear in the particle data, can be averaged directly),
        e.g. ``PRIMARY_FIELDS`` plus ``"fluctuation_weight"``. All other fields are derived
        (must be recomputed after averaging).
    export_primary_fields : bool
        Whether the primary fields needed by the derived fields were exported.
    cg_calc_mode : str
        'Monodisperse' or 'Polydisperse'.
    phases : array_like, shape (Nphases, 2)
        Diameter and density of each phase (needed for the polydisperse inertial number).
    grid_dimension : int
        2 or 3.
    nodes, spacing : array_like
        Number of grid nodes and grid spacing along x, y, z.

    Outputs
    -------
    attrs : dict
        Attributes for :meth:`H5XarrayManager.add_attrs`.
    """
    primary_names = set(primary_names)
    primary = [k for k in results if k in primary_names]
    derived = [k for k in results if k not in primary_names]
    return {"primary_fields": primary,
            "derived_fields": derived,
            "export_primary_fields": bool(export_primary_fields),
            "cg_calc_mode": cg_calc_mode,
            "phases_diameter_density": np.asarray(phases, dtype=np.float64),
            "grid_dimension": int(grid_dimension),
            "grid_nodes": np.asarray(nodes, dtype=np.int64),
            "grid_spacing": np.asarray(spacing, dtype=np.float64)}