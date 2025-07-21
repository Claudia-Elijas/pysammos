####################################################################################
# ---------------------------------------------------------------------------------#
#                  Functions to calculate  Visibility and Weights                  #
# ---------------------------------------------------------------------------------#
####################################################################################

####################################################################################
import numpy as np
import math
####################################################################################


##################################################################################
def H(dis): 
    if dis > 0:
        return 1
    else:
        return 0
    
def ComputeHeavySideWeight(c, dist):
    # given that c = w
    if dist > c:
        return 0
    else:
        
        Omega = (4/3)*math.pi*c**3 
        HS  = (1/Omega) * H(c - dist)
        return HS


####################################################################################

def ComputeGaussianWeight(c, dist):
    '''
        Compute the CG weight using a Gaussian function
        ---------------------------------------------------------------------
        Inputs:
            -----------------------------------------------------------------
            `w (float)`: half-width of cuteoff distance (scalar)
            `dist (float)`: distance between the CG and discrete points (scalar)
            `c (float)`: cut-off distance (scalar)

        Output:
            -----------------------------------------------------------------
            `W (float)`: interpolation weight
    '''
    
    if dist > c:
        W = 0
    else:
        w = c/3
        Vw = 2 * math.sqrt(2) * w**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*w)) - 4 * c * w**2 * math.pi * math.exp(-(c**2/(2*w**2))) # Matlab-Carrara
        
        #Vw = 2 * math.sqrt(2) * (c/3)**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*(c/3))) - 4 * c * (c/3)**2 * math.pi * math.exp(-(c**2/(2*(c/3)**2))) # Matlab-Carrara
        
        #Vw = math.erf(c/(math.sqrt(2)*w))- math.sqrt(2/math.pi) * (c/w) * math.exp(-(c**2)/(2*w**2)) # Weinhart 2016
        #Vw = (np.sqrt(2 * np.pi) * w)**3 # Tunuguntla 2016
        if Vw == 0:
            W = 0
        else:
            W = (1/Vw) * math.exp(-((dist**2))/(2*w**2))

    return W

####################################################################################

def ComputeLucyWeight(c,dist):
    """
        Compute the CG weight using a Lucy function
        ---------------------------------------------------------------------
        Inputs:
            -----------------------------------------------------------------
            `c (float)`: cut-off distance (scalar)
            `dist (float)`: distance between the CG and discrete points (scalar)

        Output:
            ------------------------------------------------------------------
            `W (float)`: interpolation weight
    """
    if dist > c:
        W = 0
    else:
        W = 105/(16*math.pi*c**3) * (-3*(dist/c)**4+8*(dist/c)**3-6*(dist/c)**2+1)
    return W

####################################################################################

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

####################################################################################

def calc_cutoff_distance(w, function):

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