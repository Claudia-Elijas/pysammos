r"""
Stress-force-fabric (SFF) friction of the contact stress, with the cross terms between the anisotropies.

The contact stress of a granular material is an angular average over the contact normals
:math:`\mathbf{n}`. Let :math:`\mathbf{g} = \ell \mathbf{f}` be a contact force :math:`\mathbf{f}`
weighted by its branch length :math:`\ell`, with normal part :math:`g_n` and tangential part
:math:`\mathbf{g}_t`. If the contact probability, the normal force and the tangential force depend on
:math:`\mathbf{n}` through three symmetric traceless anisotropy tensors,

.. math::
    p(\mathbf{n}) = \frac{1 + \mathbf{A}_c : \mathbf{n}\mathbf{n}}{4\pi}, \qquad
    g_n(\mathbf{n}) = g_0\,(1 + \mathbf{A}_n : \mathbf{n}\mathbf{n}), \qquad
    \mathbf{g}_t(\mathbf{n}) = g_0\,[\,\mathbf{A}_t \mathbf{n} - (\mathbf{A}_t : \mathbf{n}\mathbf{n})\,\mathbf{n}\,],

then the contact stress is :math:`\boldsymbol{\sigma}_c = 3 P_c \mathbf{Q} / \mathrm{tr}\,\mathbf{Q}` with the
contact pressure :math:`P_c` taken from the measured contact tensor and

.. math::
    \mathbf{Q} = \mathbf{I} + \mathbf{L} + \mathbf{C}, \qquad
    \mathbf{L} = \tfrac{2}{5}(\mathbf{A}_c + \mathbf{A}_n) + \tfrac{3}{5}\mathbf{A}_t,

.. math::
    \mathbf{C} = \tfrac{2}{35}\Big\{ [\mathbf{A}_c : (\mathbf{A}_n - \mathbf{A}_t)]\,\mathbf{I}
    + 2(\mathbf{A}_c\mathbf{A}_n + \mathbf{A}_n\mathbf{A}_c)
    + \tfrac{3}{2}(\mathbf{A}_c\mathbf{A}_t + \mathbf{A}_t\mathbf{A}_c) \Big\}.

:math:`\mathbf{L}` is the usual linear SFF relation. :math:`\mathbf{C}` holds the cross terms, the products
between the three anisotropies that the linear relation drops; they follow from the sixth-order moments of
the unit sphere. The three tensors are full 3x3 tensors, so their principal axes need not coincide.
The relation reconstructs the friction of the contact stress; the contact pressure is not predicted.
Adding the measured kinetic stress gives

.. math::
    \boldsymbol{\sigma}^{SFF} = \boldsymbol{\sigma}_c^{SFF} + \boldsymbol{\sigma}_k, \qquad
    P = P_c + P_k, \qquad
    \mu_{SFF} = \frac{\sqrt{\tfrac12\, \mathrm{dev}\,\boldsymbol{\sigma}^{SFF} : \mathrm{dev}\,\boldsymbol{\sigma}^{SFF}}}{P},

which has the convention of ``frictional_coefficient_Dxyz_Pxyz``.

.. rubric:: Use in a run

Set ``sff_friction = True`` in the ``[fields_to_export]`` section of ``config.ini``. It needs the contact
forces and the branch vectors, like the contact tensor, and it is computed for the bulk (all phases together).
The group exports

- ``fabric_anisotropy_tensor``, ``normal_force_anisotropy_tensor`` and ``tangential_force_anisotropy_tensor``
  (:math:`\mathbf{A}_c`, :math:`\mathbf{A}_n`, :math:`\mathbf{A}_t`),
- ``contact_stress_tensor_SFF`` (:math:`\boldsymbol{\sigma}_c^{SFF}`, with the cross terms),
- ``frictional_coefficient_SFF_Dxyz_Pxyz`` (:math:`\mu_{SFF}` with the cross terms) and
  ``frictional_coefficient_SFF_linear_Dxyz_Pxyz`` (the same with :math:`\mathbf{L}` only).

.. rubric:: How the anisotropies are measured

All sums below run over the contact entries with the kernel weights of the contact tensor.

- :math:`\mathbf{A}_c = \tfrac{15}{2}\,\mathrm{dev}\,\langle \mathbf{n}\mathbf{n} \rangle`.
- :math:`(g_0, \mathbf{B})` minimise :math:`\sum_c w_c\,[\,g_{n,c} - g_0 - \mathbf{B} : \mathbf{n}_c\mathbf{n}_c\,]^2`
  over a scalar :math:`g_0` and a symmetric traceless :math:`\mathbf{B}`, and :math:`\mathbf{A}_n = \mathbf{B}/g_0`.
- :math:`\mathbf{T}` minimises :math:`\sum_c w_c\,\| \mathbf{g}_{t,c} - (\mathbf{I} - \mathbf{n}_c\mathbf{n}_c)\,\mathbf{T}\mathbf{n}_c \|^2`
  over a symmetric traceless :math:`\mathbf{T}`, and :math:`\mathbf{A}_t = \mathbf{T}/g_0`.

Both fits are linear least squares whose normal equations contain only kernel-weighted sums of products of
the contact normal and the contact force. Those 35 sums are the *primary* field ``sff_moments``: they are
computed at every time step by :func:`coarse_grain_moments`, they are the only SFF quantity that is
averaged in time, and the anisotropies and the friction are derived from them by :func:`sff_fields`, as for
every other derived field. A node where the fits are undetermined (no contacts, too few contact
directions) is set to NaN.

References
----------
Rothenburg and Bathurst (1989), Geotechnique 39(4), 601-614; Guo and Zhao (2013), Computers and Geotechnics 47, 1-15.
"""

import itertools

import numpy as np
from numba import njit, prange, int32, float64

#: Smallest ratio of the smallest to the largest eigenvalue of the normal equations for a node to be fitted.
MIN_RCOND = 1.0e-8

#: Components of a symmetric 3x3 tensor, in the order xx, yy, zz, xy, xz, yz.
SYM6 = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

#: Independent components of the symmetric fourth-order tensor n_i n_j n_k n_l (i <= j <= k <= l).
M4_INDEX = tuple((i, j, k, l) for i in range(3) for j in range(i, 3) for k in range(j, 3) for l in range(k, 3))

# Layout of the moment vector of one node:
#   0         sum w                       number of contacts
#   1..6      sum w n n                   (xx, yy, zz, xy, xz, yz)
#   7..21     sum w n n n n               (15 independent components)
#   22        sum w g_n                   g_n = g . n
#   23..28    sum w g_n n n               (6 components)
#   29..34    sum w sym(g_t (x) n)        g_t = g - g_n n  (6 components)
_I_COUNT, _I_M2, _I_M4, _I_G0, _I_G2, _I_H = 0, 1, 7, 22, 23, 29
N_MOMENTS = 35

_SYM6 = np.array(SYM6, dtype=np.int64)
_M4_IDX = np.array(M4_INDEX, dtype=np.int64)

# ---------------------------------------------------------------------------------------------
# symmetric traceless 3x3 tensors: orthonormal basis, constant contraction tables
# ---------------------------------------------------------------------------------------------
_S2, _S6 = np.sqrt(2.0), np.sqrt(6.0)

#: Orthonormal basis (Frobenius product) of the symmetric traceless 3x3 tensors, shape (5, 3, 3).
BASIS = np.array([
    np.diag([1.0, -1.0, 0.0]) / _S2,
    np.diag([-1.0, -1.0, 2.0]) / _S6,
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float) / _S2,
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float) / _S2,
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float) / _S2,
])

# (E_k E_l)_ab, used by the tangential normal equations
_EE = np.einsum("kab,lbc->klac", BASIS, BASIS)


def _basis_contraction_m4():
    """Table T[k, l, m] with E_k : M4 : E_l = sum_m T[k, l, m] M4_m for the 15 independent components."""
    table = np.zeros((5, 5, len(M4_INDEX)))
    for m, idx in enumerate(M4_INDEX):
        for a, b, c, d in set(itertools.permutations(idx)):   # every distinct index tuple once
            table[:, :, m] += BASIS[:, a, b][:, None] * BASIS[:, c, d][None, :]
    return table


_T4 = _basis_contraction_m4()


def _sym6_to_tensor(v):
    """(..., 6) components xx, yy, zz, xy, xz, yz -> (..., 3, 3) symmetric tensors."""
    out = np.empty(v.shape[:-1] + (3, 3))
    for s, (a, b) in enumerate(SYM6):
        out[..., a, b] = v[..., s]
        out[..., b, a] = v[..., s]
    return out


def _from_coefficients(c):
    """(..., 5) coefficients on the basis -> (..., 3, 3) symmetric traceless tensors."""
    return np.einsum("...k,kab->...ab", c, BASIS)


def _deviator(a):
    return a - (np.trace(a, axis1=-2, axis2=-1) / 3.0)[..., np.newaxis, np.newaxis] * np.eye(3)


def _swap(a):
    return np.swapaxes(a, -1, -2)


# ---------------------------------------------------------------------------------------------
# primary field: kernel-weighted moments of the contacts
# ---------------------------------------------------------------------------------------------
def contact_vectors(center_to_center, force, branch):
    """
    Contact normals and branch-length weighted forces of the contact entries.

    Inputs
    ------
    center_to_center : np.ndarray, shape (M, 3)
        Vector between the two particle centres of each contact entry.
    force : np.ndarray, shape (M, 3)
        Contact force of each entry.
    branch : np.ndarray, shape (M, 3)
        Branch vector of each entry (contact point to particle centre); only its length is used.

    Outputs
    -------
    normals : np.ndarray, shape (M, 3)
        Unit vectors along ``center_to_center``.
    g : np.ndarray, shape (M, 3)
        ``force`` times the branch length.
    """
    c2c = np.asarray(center_to_center, dtype=np.float64)
    length = np.linalg.norm(c2c, axis=-1)
    normals = c2c / np.where(length > 0.0, length, 1.0)[:, np.newaxis]
    ell = np.linalg.norm(np.asarray(branch, dtype=np.float64), axis=-1)
    g = np.asarray(force, dtype=np.float64) * ell[:, np.newaxis]
    return np.ascontiguousarray(normals), np.ascontiguousarray(g)


@njit(float64[:,:](float64[:], int32[:], int32[:], float64[:,:], float64[:,:]), parallel=True)
def _moments_csr(weights, visibility, grid_indices, normals, g):
    """
    Kernel-weighted moments of the contacts at every grid node (see :func:`coarse_grain_moments`).

    Inputs
    ------
    weights : (N_vis,) float64 array
        Coarse-graining weights for each visible contact entry.
    visibility : (N_vis,) int32 array
        Contact-entry indices corresponding to each entry in the weights array.
    grid_indices : (N_points + 2,) int32 array
        Start/end indices into the visibility array for each grid point.
        Assumes padding: first = 0, last = len(visibility).
    normals : (M, 3) float64 array
        Unit normal of each contact entry.
    g : (M, 3) float64 array
        Branch-length weighted force of each contact entry.

    Outputs
    -------
    out : (N_points, 35) float64 array
        The 35 moments of each node, in the layout read back by :func:`anisotropies`.
    """
    n_points = len(grid_indices) - 2
    out = np.zeros((n_points, N_MOMENTS))
    for p in prange(n_points):
        acc = np.zeros(N_MOMENTS)
        for i in range(grid_indices[p], grid_indices[p + 1]):
            c = visibility[i]
            w = weights[i]
            nv = normals[c]
            gv = g[c]
            gn = gv[0] * nv[0] + gv[1] * nv[1] + gv[2] * nv[2]
            acc[_I_COUNT] += w
            acc[_I_G0] += w * gn
            for s in range(6):
                a = _SYM6[s, 0]
                b = _SYM6[s, 1]
                nn = nv[a] * nv[b]
                acc[_I_M2 + s] += w * nn
                acc[_I_G2 + s] += w * gn * nn
                acc[_I_H + s] += w * (0.5 * (gv[a] * nv[b] + gv[b] * nv[a]) - gn * nn)
            for m in range(15):
                acc[_I_M4 + m] += w * nv[_M4_IDX[m, 0]] * nv[_M4_IDX[m, 1]] * nv[_M4_IDX[m, 2]] * nv[_M4_IDX[m, 3]]
        out[p, :] = acc
    return out


def coarse_grain_moments(weights, visibility, grid_indices, center_to_center, force, branch):
    """
    Kernel-weighted moments of the contacts at every grid node (the primary field ``sff_moments``).

    The weights, visibility and grid indices are those of the contact tensor (``Wint_c``, ``part_ind_c``
    and ``grid_ind_c`` of :class:`~pysammos.coarse_graining.CoarseGraining`); the contact arrays are those
    of :func:`contact_vectors`.

    Outputs
    -------
    moments : np.ndarray, shape (Npoints, 35)
        Sums over the contacts visible from each node; the layout is given in the module source and
        read back by :func:`anisotropies`.
    """
    normals, g = contact_vectors(center_to_center, force, branch)
    return _moments_csr(np.asarray(weights, dtype=np.float64), np.asarray(visibility, dtype=np.int32),
                        np.asarray(grid_indices, dtype=np.int32), normals, g)


def moments_from_contacts(normals, g, weights=None):
    """
    Moments of one set of contacts, in plain numpy (a whole box, or a reference for the grid kernel).

    Inputs
    ------
    normals : np.ndarray, shape (M, 3)
        Unit contact normals.
    g : np.ndarray, shape (M, 3)
        Branch-length weighted contact forces.
    weights : np.ndarray, shape (M,), optional
        Weight of each contact (default 1).

    Outputs
    -------
    moments : np.ndarray, shape (35,)
    """
    n = np.asarray(normals, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    w = np.ones(len(n)) if weights is None else np.asarray(weights, dtype=np.float64)
    gn = np.einsum("ci,ci->c", g, n)
    gt = g - gn[:, np.newaxis] * n
    m = np.empty(N_MOMENTS)
    m[_I_COUNT] = w.sum()
    m[_I_G0] = (w * gn).sum()
    for s, (a, b) in enumerate(SYM6):
        nn = n[:, a] * n[:, b]
        m[_I_M2 + s] = (w * nn).sum()
        m[_I_G2 + s] = (w * gn * nn).sum()
        m[_I_H + s] = (w * 0.5 * (gt[:, a] * n[:, b] + gt[:, b] * n[:, a])).sum()
    for k, (a, b, c, d) in enumerate(M4_INDEX):
        m[_I_M4 + k] = (w * n[:, a] * n[:, b] * n[:, c] * n[:, d]).sum()
    return m


# ---------------------------------------------------------------------------------------------
# derived: anisotropies from the moments
# ---------------------------------------------------------------------------------------------
def _well_conditioned(matrices, rcond):
    """Boolean mask of the (finite, positive semi-definite) matrices whose smallest eigenvalue exceeds rcond * largest."""
    ok = np.zeros(len(matrices), dtype=bool)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    if finite.any():
        eig = np.linalg.eigvalsh(matrices[finite])
        ok[finite] = eig[:, 0] > rcond * eig[:, -1]
    return ok


def anisotropies(moments, rcond=MIN_RCOND):
    """
    Contact, normal-force and tangential-force anisotropy tensors from the moments.

    Inputs
    ------
    moments : np.ndarray, shape (Npoints, 35)
        Moments of :func:`coarse_grain_moments` (or their time average).
    rcond : float, optional
        A node is fitted only if the smallest eigenvalue of both normal-equation matrices exceeds
        ``rcond`` times the largest one; otherwise its tensors are NaN.

    Outputs
    -------
    Ac, An, At : np.ndarray, shape (Npoints, 3, 3)
        Symmetric traceless anisotropy tensors; NaN where undefined (no contacts, too few directions,
        or a non-positive mean normal force ``g_0``).
    """
    m = np.atleast_2d(np.asarray(moments, dtype=np.float64))
    npts = len(m)
    count = m[:, _I_COUNT]
    M2 = _sym6_to_tensor(m[:, _I_M2:_I_M2 + 6])
    M4 = m[:, _I_M4:_I_M4 + 15]
    G2 = _sym6_to_tensor(m[:, _I_G2:_I_G2 + 6])
    H = _sym6_to_tensor(m[:, _I_H:_I_H + 6])

    e_M2 = np.einsum("kab,pab->pk", BASIS, M2)
    e_M4 = np.einsum("klm,pm->pkl", _T4, M4)

    # normal force: unknowns (g0, B_1 ... B_5), the matrix is the Gram matrix of the features (1, E_k : nn)
    A = np.zeros((npts, 6, 6))
    A[:, 0, 0] = count
    A[:, 0, 1:] = e_M2
    A[:, 1:, 0] = e_M2
    A[:, 1:, 1:] = e_M4
    rhs_n = np.concatenate([m[:, _I_G0:_I_G0 + 1], np.einsum("kab,pab->pk", BASIS, G2)], axis=1)

    # tangential force: unknowns T_1 ... T_5, the matrix is the Gram matrix of (E_k n) - (n . E_k n) n
    C = np.einsum("klab,pab->pkl", _EE, M2) - e_M4
    rhs_t = np.einsum("kab,pab->pk", BASIS, H)

    Ac = np.full((npts, 3, 3), np.nan)
    An = np.full((npts, 3, 3), np.nan)
    At = np.full((npts, 3, 3), np.nan)

    fit = np.flatnonzero((count > 0.0) & _well_conditioned(A, rcond) & _well_conditioned(C, rcond))
    if len(fit) == 0:
        return Ac, An, At
    u = np.linalg.solve(A[fit], rhs_n[fit][..., np.newaxis])[..., 0]
    t = np.linalg.solve(C[fit], rhs_t[fit][..., np.newaxis])[..., 0]
    g0 = u[:, 0]
    keep = g0 > 0.0
    fit, u, t, g0 = fit[keep], u[keep], t[keep], g0[keep]

    scale = g0[:, np.newaxis, np.newaxis]
    Ac[fit] = 7.5 * _deviator(M2[fit] / count[fit][:, np.newaxis, np.newaxis])
    An[fit] = _from_coefficients(u[:, 1:]) / scale
    At[fit] = _from_coefficients(t) / scale
    return Ac, An, At


# ---------------------------------------------------------------------------------------------
# derived: contact stress and friction
# ---------------------------------------------------------------------------------------------
def contact_stress_shape(Ac, An, At, cross_terms=True):
    r"""
    Contact stress divided by the contact pressure, :math:`\boldsymbol{\sigma}_c / P_c`, which has trace 3.

    Inputs
    ------
    Ac, An, At : np.ndarray, shape (..., 3, 3)
        Anisotropy tensors (symmetric, traceless).
    cross_terms : bool, optional
        True: :math:`3\mathbf{Q}/\mathrm{tr}\,\mathbf{Q}` with the cross terms. False: the linear relation
        :math:`\mathbf{I} + \mathbf{L}`.
    """
    eye = np.eye(3)
    linear = 0.4 * (Ac + An) + 0.6 * At
    if not cross_terms:
        return eye + linear
    AcAn = Ac @ An
    AcAt = Ac @ At
    scalar = np.einsum("...ij,...ij->...", Ac, An - At)
    cross = (2.0 / 35.0) * (scalar[..., np.newaxis, np.newaxis] * eye
                            + 2.0 * (AcAn + _swap(AcAn)) + 1.5 * (AcAt + _swap(AcAt)))
    Q = eye + linear + cross
    return 3.0 * Q / np.trace(Q, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis]


def _bulk(tensor):
    """Bulk tensor: the field itself (Monodisperse) or its phase index 0 (Polydisperse)."""
    return tensor[:, 0] if tensor.ndim == 4 else tensor


def friction_from_stress(sigma_contact, sigma_kinetic, pressure):
    """mu = sqrt(1/2 dev(sigma) : dev(sigma)) / pressure, with sigma = sigma_contact + sigma_kinetic."""
    dev = _deviator(sigma_contact + sigma_kinetic)
    return np.sqrt(0.5 * np.einsum("...ij,...ij->...", dev, dev)) / pressure


def sff_fields(moments, contact_tensor, kinetic_tensor, rcond=MIN_RCOND):
    """
    Anisotropies, SFF contact stress and SFF friction coefficients (bulk, all three dimensions).

    Inputs
    ------
    moments : np.ndarray, shape (Npoints, 35)
        Primary field ``sff_moments`` of one time step or its time average.
    contact_tensor, kinetic_tensor : np.ndarray, shape (Npoints, 3, 3) or (Npoints, Nphases+1, 3, 3)
        Primary stress tensors; the bulk (phase index 0) is used in Polydisperse mode.
    rcond : float, optional
        See :func:`anisotropies`.

    Outputs
    -------
    fields : dict
        ``fabric_anisotropy_tensor``, ``normal_force_anisotropy_tensor``, ``tangential_force_anisotropy_tensor``,
        ``contact_stress_tensor_SFF`` (with the cross terms), ``frictional_coefficient_SFF_Dxyz_Pxyz`` (with the
        cross terms) and ``frictional_coefficient_SFF_linear_Dxyz_Pxyz`` (linear relation only). The pressure
        is the measured one, so the denominator equals that of ``frictional_coefficient_Dxyz_Pxyz``.
    """
    sigma_c = _bulk(contact_tensor)
    sigma_k = _bulk(kinetic_tensor)
    contact_pressure = np.trace(sigma_c, axis1=-2, axis2=-1) / 3.0
    pressure = np.trace(sigma_c + sigma_k, axis1=-2, axis2=-1) / 3.0

    Ac, An, At = anisotropies(moments, rcond)
    shape = contact_stress_shape(Ac, An, At, cross_terms=True)
    shape_linear = contact_stress_shape(Ac, An, At, cross_terms=False)
    sigma_sff = contact_pressure[:, np.newaxis, np.newaxis] * shape
    sigma_sff_linear = contact_pressure[:, np.newaxis, np.newaxis] * shape_linear
    return {
        "fabric_anisotropy_tensor": Ac,
        "normal_force_anisotropy_tensor": An,
        "tangential_force_anisotropy_tensor": At,
        "contact_stress_tensor_SFF": sigma_sff,
        "frictional_coefficient_SFF_Dxyz_Pxyz": friction_from_stress(sigma_sff, sigma_k, pressure),
        "frictional_coefficient_SFF_linear_Dxyz_Pxyz": friction_from_stress(sigma_sff_linear, sigma_k, pressure),
    }
