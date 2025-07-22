# calculate w
def calc_half_width(Average_part_diam, w_mult=0.75):
    
    r"""Calculate the half-width parameter `w` based on average particle diameter.

    This function computes the smoothing or weighting half-width `w` used in kernel
    functions for back-projection or filtering processes. The value of `w` is
    computed by scaling the average particle diameter by a multiplicative factor.

    .. math::
        w = w_{\text{mult}} \cdot D

    where :math:`D` is the average particle diameter and :math:`w_{\text{mult}}` is
    the user-defined multiplicative scaling factor.

    Parameters
    ----------
    Average_part_diam : float
        The average particle diameter :math:`D`.

    w_mult : float, optional
        The multiplicative factor for calculating the half-width `w`.
        Default is 0.75.

    Returns
    -------
    w : float
        The computed half-width `w`.
    """
    
    w = w_mult * Average_part_diam
    return w

# calcualte c
def calc_cutoff(w, function):
    
    r"""Calculate the cutoff radius `c` for a given kernel function.

    This function determines the cutoff distance :math:`c` used to truncate the kernel
    or filter function, based on the half-width `w` and the specified kernel type.

    The cutoff `c` is calculated as:

    - For the Lucy kernel: :math:`c = 2w`
    - For the Gaussian kernel: :math:`c = 3w`
    - For the Heaviside (step) function: :math:`c = w`

    Parameters
    ----------
    w : float
        The half-width parameter used in the kernel.

    function : str
        The name of the kernel function. Must be one of:
        `'Lucy'`, `'Gaussian'`, or `'HeavySide'`.

    Returns
    -------
    c : float
        The calculated cutoff value.

    Raises
    ------
    ValueError
        If the `function` argument is not one of the supported types.
    """

    if function == 'Lucy':
        c = 2*w
    elif function == 'Gaussian':
        c = 3*w
    elif function == 'HeavySide':
        c = 1*w
    else:
        raise ValueError(f"Unsupported function type: {function}")

    return c

####################################################################################