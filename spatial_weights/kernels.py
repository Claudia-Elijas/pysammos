from scipy.special import erf
import numpy as np


# Lucy function
def Lucy(c,dist):
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
        W = 105/(16*np.pi*c**3) * (-3*(dist/c)**4+8*(dist/c)**3-6*(dist/c)**2+1)
    return W


# top hat function
def H(dis): 
    if dis > 0:
        return 1
    else:
        return 0

# Heavy side function
def HeavySide(c, dist):
    # given that c = w
    if dist > c:
        return 0
    else:
        
        Omega = (4/3)*np.pi*c**3 
        HS  = (1/Omega) * H(c - dist)
        return HS


# Gaussian function
def Gaussian(c, dist):
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
        Vw = 2 * np.sqrt(2) * w**3 * np.pi**(3/2) * erf((c*np.sqrt(2))/(2*w)) - 4 * c * w**2 * np.pi * np.exp(-(c**2/(2*w**2))) # Matlab-Carrara
        
        #Vw = 2 * math.sqrt(2) * (c/3)**3 * math.pi**(3/2) * math.erf((c*math.sqrt(2))/(2*(c/3))) - 4 * c * (c/3)**2 * math.pi * math.exp(-(c**2/(2*(c/3)**2))) # Matlab-Carrara
        
        #Vw = math.erf(c/(math.sqrt(2)*w))- math.sqrt(2/math.pi) * (c/w) * math.exp(-(c**2)/(2*w**2)) # Weinhart 2016
        #Vw = (np.sqrt(2 * np.pi) * w)**3 # Tunuguntla 2016
        if Vw == 0:
            W = 0
        else:
            W = (1/Vw) * np.exp(-((dist**2))/(2*w**2))

    return W


