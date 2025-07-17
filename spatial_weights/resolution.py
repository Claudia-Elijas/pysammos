
# calculate w
def calc_half_width(Average_part_diam, w_mult=0.75):
    '''
        Compute the half-width of the cut-off distance
        ---------------------------------------------------------------------
        Inputs:
            -----------------------------------------------------------------
            `Average_part_diam (float)`: average diameter of the particles
            `w_mult (float)`: multiplier for the half-width

        Output:
            ------------------------------------------------------------------
            `w (float)`: half-width of the cut-off distance
    '''
    w = w_mult * Average_part_diam
    return w

# calcualte c
def calc_cutoff(w, function):

    '''
        Compute the cut-off distance
        ---------------------------------------------------------------------
        Inputs:
            -----------------------------------------------------------------
            `w (float)`: half-width of the cut-off distance
            `function (str)`: type of weighting function to be used

        Output:
            ------------------------------------------------------------------
            `c (float)`: cut-off distance
    
    '''

    if function == 'Lucy':
        c = 2*w
    elif function == 'Gaussian':
        c = 3*w
    elif function == 'HeavySide':
        c = 1*w

    return c

####################################################################################