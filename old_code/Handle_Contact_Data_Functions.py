import numpy as np
from collections import defaultdict
from numba import njit, prange
# Get unique pairs of two arrays 
def Get_Unique_Pairs(LL_py, I_py):
    pair_indices = defaultdict(list) # Store indices of each pair, treating (A,B) and (B,A) as the same
    for idx, (ll, i) in enumerate(zip(LL_py, I_py)):
        sorted_pair = tuple(sorted((ll, i))) # Sorting ensures (2,6) == (6,2)
        pair_indices[sorted_pair].append(idx)
    keep_mask = np.ones(len(LL_py), dtype=bool) # Create a boolean mask to track which elements to keep
    for pair, indices in pair_indices.items(): 
        if len(indices) > 1: # If duplicates exist
            #print(f"First Pair: {LL_py[indices[0]], I_py[indices[0]]}. Second Pair: {LL_py[indices[1]], I_py[indices[1]]}")
            for idx in indices[1:]:# Mark duplicate occurrences for removal (except the first one)
                keep_mask[idx] = False
    keep_indices = np.where(keep_mask)[0] ; #print('Keep indices: ', len(keep_indices))
    #remove_indices = np.where(~keep_mask)[0] ; #print('Remove indices: ', len(remove_indices))  
    return keep_indices
# Check if there is duplicate pairs
def Check_for_Duplicate_Pairs(Particle_LL_OG, Particle_I_OG, Fij_OG, Contacts_OG):
    # Find Unique Pairs
    keep_these = Get_Unique_Pairs(Particle_LL_OG, Particle_I_OG)
    #Select the Contact data corresponding to Unique Pairs
    Particle_LL = Particle_LL_OG[keep_these]
    Particle_I = Particle_I_OG[keep_these]
    Fij = Fij_OG[keep_these]
    Contacts = Contacts_OG[keep_these] if Contacts_OG is not None else None
    print('Repeated pairs in contact data: ', len(Particle_LL_OG) - len(Particle_LL))
    return Particle_LL, Particle_I, Fij, Contacts

# Calculate coordination number
@njit
def Calc_Coordination_Number(particle_inContacts_dup, global_id_all):
  
    # Step 1: Count occurrences using np.bincount
    max_id = np.max(particle_inContacts_dup)
    count_array = np.zeros(max_id + 1, dtype=np.int64)
    count_array[:len(np.bincount(particle_inContacts_dup))] = np.bincount(particle_inContacts_dup)

    # Step 2: Map counts to global_id_all
    CN = np.zeros(len(global_id_all), dtype=np.int64)
    for i in range(len(global_id_all)):
        pid = global_id_all[i]
        if pid <= max_id:
            CN[i] = count_array[pid]  # 0 if not found in contacts

    # Step 3: Filter out rattlers (particles with CN == 0)
    CN_no_rattlers = CN[CN > 1]

    return CN, CN_no_rattlers

# Calculate branch vectors
def Calc_BranchVector_CyclicBoundaries__contacts(r_A, r_B, contact_A, L, periodic):
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
def Calc_BranchVector_CyclicBoundaries__diameters(r_A, r_B, d_A, d_B, L, periodic):

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
# Produce a phase array
def Produce_Phase_Index_Array(UniquePhases, Density, Diameter):
    # Use broadcasting to match each particle's density and diameter with the phase types
    
    phase_mask = (Density[:, np.newaxis] == UniquePhases[:, 1]) & (Diameter[:, np.newaxis] == UniquePhases[:, 0])
    Phases_ar = np.argmax(phase_mask, axis=1) # Find the index of the matching phase for each particle
    del phase_mask

    print("Phase array made")
    return Phases_ar

# def Produce_Phase_Index_Array(UniquePhases, Density, Diameter): # new with no broadcasting (needs to be checked)
#     # Create lookup: (diameter, density) → phase index
#     phase_lookup = { (d, rho): idx for idx, (d, rho) in enumerate(UniquePhases) }

#     # Build phase index array without broadcasting
#     Phases_ar = np.array([
#         phase_lookup.get((d, rho), -1)  # -1 if no match found
#         for d, rho in zip(Diameter, Density)
#     ])
#     print("Phase array made (no broadcasting)")
#     return Phases_ar
# Load and Arrange Contact data
def Arange_ContactData(Global_ID, Position, Diameter, Density, Volume, # Particle data
                                    Particle_LL, Particle_I, Fij, Contact_Points,# Contact data
                                    ModelAxesRanges, AxesPeriodicity, # Information for Branch Vector Calculation
                                    Return_Volume, # Is volume data needed to calculate fabric tensor?
                                    Particle_Phase_Array_t):  # if True, does NOT return a phase array
    
    # --------------------------------------------------------------------- #

    
      
    # --------------------------------------------------------------------- #

    # Find equivalence in ID indices
    inds_glob_LL = np.searchsorted(Global_ID, Particle_LL)
    inds_glob_I  = np.searchsorted(Global_ID, Particle_I) 
    
    valid_LL = Global_ID[inds_glob_LL] == Particle_LL
    valid_I = Global_ID[inds_glob_I] == Particle_I
    inds_glob_LL = inds_glob_LL[valid_LL]
    inds_glob_I = inds_glob_I[valid_I] 

    print("Found relationship between contact data and particle data")

    # Get arrays from VELOCITY files
    Pos_LL = Position[inds_glob_LL] # Positions (to calculate branch vector)
    Pos_I = Position[inds_glob_I] # Positions (to calculate branch vector)
    Diam_LL = Diameter[inds_glob_LL] #  Diameters (to filter the phase  +  to calculate branch vector)
    Diam_I = Diameter[inds_glob_I] # (to calculate branch vector)   
    Density_LL = Density[inds_glob_LL] #  Density (to filter the phase)
    Density_I = Density[inds_glob_I] #  Density (to filter the phase)
    
    # Account for both particles involved in interaction by concatenating the arrays 
    Position_LL_dup = np.concatenate((Pos_LL, Pos_I))
    Pos_I_dup = np.concatenate((Pos_I, Pos_LL))
    Force_LL_dup = np.concatenate((Fij, -Fij))
    Density_LL_dup = np.concatenate((Density_LL, Density_I))
    Diam_LL_dup = np.concatenate((Diam_LL, Diam_I))

    if Particle_Phase_Array_t is not None:
        Phases_array_LL = Particle_Phase_Array_t[inds_glob_LL] # Phases (to filter the phase)
        Phases_array_I = Particle_Phase_Array_t[inds_glob_I]
        Phases_array = np.concatenate((Phases_array_LL, Phases_array_I))
    else: 
        Phases_array = None

    # Mean diameter
    Mean_Diameter = np.mean(Diam_LL_dup)
  
    # --------------------------------------------------------------------- #

    # Calculate the Branch Vectors
    if Contact_Points is not None:
        Contact_Points_LL = Contact_Points
        Contact_Points_LL_dup = np.concatenate((Contact_Points_LL, Contact_Points_LL))
        BranchVector_LL_dup, CenterToCenterVector_LL_dup = Calc_BranchVector_CyclicBoundaries__contacts(Position_LL_dup, Pos_I_dup,
                                                                            Contact_Points_LL_dup, 
                                                                           ModelAxesRanges, AxesPeriodicity)
    elif Contact_Points is None:
        Diam_I_dup = np.concatenate((Diam_I, Diam_LL))
        BranchVector_LL_dup, CenterToCenterVector_LL_dup = Calc_BranchVector_CyclicBoundaries__diameters(Position_LL_dup, Pos_I_dup, 
                                                                            Diam_LL_dup, Diam_I_dup, 
                                                                            ModelAxesRanges, AxesPeriodicity)
        
    # --------------------------------------------------------------------- #  
    # Get data for fabric tensor
    if Return_Volume == True:
        Volume_LL = Volume[inds_glob_LL] #  Volumes (to calulate fabric tensor)
        Volume_I = Volume[inds_glob_I] #  Volumes (to calulate fabric tensor)
        Volume_LL_dup = np.concatenate((Volume_LL, Volume_I))
    elif Return_Volume == False:
        Volume_LL_dup = None

    # --------------------------------------------------------------------- #

    
    return Position_LL_dup, Force_LL_dup, BranchVector_LL_dup, CenterToCenterVector_LL_dup, Volume_LL_dup, Phases_array, Mean_Diameter



# OLD FUNCTION THAT WORKS 
# def Arange_ContactData__(Global_ID, Position, Diameter, Density, Volume, # Particle data
#                                     Particle_LL_OG, Particle_I_OG, Fij_OG, Contact_Points, # Contact data
#                                     AxesRanges, AxesPeriodicity, # Information for Branch Vector Calculation
#                                     Phases, # Phases in data to make phase array
#                                     FabricTensor): 

#         # Find Unique Pairs
#         keep_these = Get_Unique_Pairs(Particle_LL_OG, Particle_I_OG)
#         # Select the Contact data corresponding to Unique Pairs
#         Particle_LL = Particle_LL_OG[keep_these]
#         Particle_I = Particle_I_OG[keep_these]
#         Fij = Fij_OG[keep_these]
        
#         # --------------------------------------------------------------------- #

#         # Find equivalence in ID indices
#         inds_glob_LL = np.argmin(np.abs(Global_ID[:, None] - Particle_LL[None, :]), axis=0)
#         inds_glob_I = np.argmin(np.abs(Global_ID[:, None] - Particle_I[None, :]), axis=0)

#         # global ids
#         Id_LL = np.concatenate((Particle_LL, Particle_I))
    
#         # Get arrays from VELOCITY files
#         Pos_LL = Position[inds_glob_LL] # Positions (to calculate branch vector)
#         Pos_I = Position[inds_glob_I] # Positions (to calculate branch vector)
#         Diam_LL = Diameter[inds_glob_LL] #  Diameters (to filter the phase  +  to calculate branch vector)
#         Diam_I = Diameter[inds_glob_I] # (to calculate branch vector)   
#         Density_LL = Density[inds_glob_LL] #  Density (to filter the phase)
#         Density_I = Density[inds_glob_I] #  Density (to filter the phase)
#         # Account for both particles involved in interaction by concatenating the arrays 
#         Diam_LL_dup = np.concatenate((Diam_LL, Diam_I)); print("Test Diam: ", Diam_LL_dup[0])
#         Position_LL_dup = np.concatenate((Pos_LL, Pos_I))
#         Force_LL_dup = np.concatenate((Fij, -Fij))
#         Density_LL_dup = np.concatenate((Density_LL, Density_I)); print("Test Density: ", Density_LL_dup[0])
        
#         # --------------------------------------------------------------------- #

#         # Calculate the Branch Vectors
#         if Contact_Points is not None:
#             Contact_Points_LL = Contact_Points[keep_these]
#             BranchVector_LL_dup = Calc_BranchVector_CyclicBoundaries__contacts(Position_LL_dup, Contact_Points_LL, AxesRanges, AxesPeriodicity)
#         elif Contact_Points is None:
#             Pos_I_dup = np.concatenate((Pos_I, Pos_LL))
#             Diam_I_dup = np.concatenate((Diam_I, Diam_LL))
#             BranchVector_LL_dup = Calc_BranchVector_CyclicBoundaries__diameters(Position_LL_dup, Pos_I_dup, Diam_LL_dup, Diam_I_dup, AxesRanges, AxesPeriodicity)
            
#         # --------------------------------------------------------------------- #

#         # Calculate the Phase Array
#         if len(Phases) == 1:
#             Phases_array = None
#         elif len(Phases) > 1:
#             Phases_array = Produce_Phase_Index_Array(Phases, Density_LL_dup, Diam_LL_dup)

#         # --------------------------------------------------------------------- #
        
#         # Get data for fabric tensor
#         if FabricTensor == True:
#             Volume_LL = Volume[inds_glob_LL] #  Volumes (to calulate fabric tensor)
#             Volume_I = Volume[inds_glob_I] #  Volumes (to calulate fabric tensor)
#             Volume_LL_dup = np.concatenate((Volume_LL, Volume_I))
#         elif FabricTensor == False:
#             Volume_LL_dup = None

#         # --------------------------------------------------------------------- #
        
#         return Position_LL_dup, Force_LL_dup, BranchVector_LL_dup, Volume_LL_dup, Phases_array