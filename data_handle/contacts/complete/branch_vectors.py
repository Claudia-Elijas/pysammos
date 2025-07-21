import numpy as np

# Calculate branch vectors
def from_contacts(r_A, r_B, contact_A, L, periodic):
    """
    Compute branch vectors (BV) from contact points relative to particle centers.

    Parameters:
    r_A        : ndarray (N, dim), Positions of particles A
    contact_A  : ndarray (N, dim), Contact points on particles A
    L          : ndarray (dim,), Box dimensions
    periodic   : list of bool, Periodic boundary conditions in each dimension

    Returns:
    BV         : ndarray (2N, dim), Branch vectors for both particles
    d         : ndarray (2N, dim), Center-to-center vectors for both particles
    """
    
    # Compute displacement vector from center to contact point
    BV = r_A - contact_A
    d = r_A - r_B  # Center-to-center vector
    
    # Adjust displacement for periodic boundaries
    for i in range(len(L)):  # Loop over each dimension
        if periodic[i]:  
            BV[:, i] = BV[:, i] - L[i] * np.round(BV[:, i] / L[i])  # Periodic correction
            d[:,i] = d[:,i] - L[i] * np.round(d[:,i] / L[i]) # correct for particles close to boundary
    
    return BV, d

# Calculate branch vectors from diameters
def from_diameters(r_A, r_B, d_A, d_B, L, periodic):

    '''
        Function to calculate the branch vector for two particles with cyclic boundaries
        ---------------------------------------------------------------------
        Input:

            -----------------------------------------------------------------
            `r_A (array)`: position of particle A (shape = (3,))
            `r_B (array)`: position of particle B (shape = (3,))
            `d_A (float)`: diameter of particle A
            `d_B (float)`: diameter of particle B
            `L (array)`: domain dimensions (shape = (3,))
            `periodic (array)`: periodic boundaries (shape = (3,))

        Output:
            -----------------------------------------------------------------
            `BV (array)`: branch vector for particle A (shape = (3,))
            `d (array)`: center-to-center vector between particle A and B (shape = (3,))
    
    
    '''

    # Initialize displacement vector
    d = r_A - r_B
    
    # Adjust displacement for periodic boundaries selectively
    for i in range(len(L)):  # Loop over each dimension
        if periodic[i]:  # Apply periodic adjustment if enabled for the dimension
            d[:,i] = d[:,i] - L[i] * np.round(d[:,i] / L[i]) # correct for particles close to boundary
    BV =  d * d_A[:, np.newaxis] / (d_A[:, np.newaxis] + d_B[:, np.newaxis])
  
    return BV , d    