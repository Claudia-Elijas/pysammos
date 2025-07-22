import numpy as np
from .complete import branch_vectors

def map_contact_data(Global_ID, Position, Diameter, Density, Volume,# Particle data
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
        BranchVector_LL_dup, CenterToCenterVector_LL_dup = branch_vectors.from_contacts(Position_LL_dup, Pos_I_dup,
                                                                            Contact_Points_LL_dup, 
                                                                           ModelAxesRanges, AxesPeriodicity)
    elif Contact_Points is None:
        Diam_I_dup = np.concatenate((Diam_I, Diam_LL))
        BranchVector_LL_dup, CenterToCenterVector_LL_dup = branch_vectors.from_diameters(Position_LL_dup, Pos_I_dup, 
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

