"""
This module provides functionality to map global particle data (positions, diameters, etc.) to contact
pair data, duplicates the arrays to account for both directions (A→B and B→A),
and computes the branch vectors using either contact points or particle diameters.
It also includes options to return particle volumes for fabric tensor calculations.

It contains one main function:
    1. :func:`map_contact_data`: Maps particle and contact data for branch vector and fabric tensor analysis.
  Returns positions, forces, branch vectors, center-to-center vectors, and optionally volumes and phases
  for each contact pair, accounting for both directions of interaction.

"""


import numpy as np
from .complete import branch_vectors
from typing import Tuple, Optional

def map_contact_data(Global_ID:np.ndarray, Position:np.ndarray, Diameter:np.ndarray, Density:np.ndarray, Volume:np.ndarray,# Particle data
                                    Particle_LL:np.ndarray, Particle_I:np.ndarray, Fij:np.ndarray, Contact_Points:np.ndarray,# Contact data
                                    ModelAxesRanges:np.ndarray, AxesPeriodicity:np.ndarray, # Information for Branch Vector Calculation
                                    Return_Volume:bool, # Is volume data needed to calculate fabric tensor?
                                    Particle_Phase_Array_t:np.ndarray)-> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,
                                                                        Optional[np.ndarray],Optional[np.ndarray], float]: # if None, no phase data is returned
    r"""
    
    Maps global particle data to contact pair data, duplicates arrays for both interaction directions,
    and computes branch vectors for fabric tensor analysis.

    Inputs
    ------

    Global_ID : np.ndarray, shape (N,)
        Array of global particle IDs.

    Position : np.ndarray, shape (N, 3)
        Particle positions.

    Diameter : np.ndarray, shape (N,)
        Particle diameters.

    Density : np.ndarray, shape (N,)
        Particle densities.

    Volume : np.ndarray, shape (N,)
        Particle volumes, used for fabric tensor calculations if `Return_Volume` is True.

    Particle_LL : np.ndarray, shape (M,)
        Arrays of particle IDs involved in contacts.

    Particle_I : np.ndarray, shape (M,)
        Arrays of particle IDs involved in contacts.

    Fij : np.ndarray, shape (M, 3)
        Contact forces between particle pairs.

    Contact_Points : np.ndarray or None, shape (M, 3), optional
        Contact points for each interaction, used to compute branch vectors.
        If None, branch vectors are computed using particle diameters.

    ModelAxesRanges : np.ndarray, shape (3,)
        Ranges of the model axes for periodic boundary conditions.

    AxesPeriodicity : np.ndarray of bool, shape (3,)
        Indicates which axes are periodic.

    Return_Volume : bool
        If True, the function returns particle volumes for fabric tensor calculations.

    Particle_Phase_Array_t : np.ndarray or None
        Array containing phase information for each particle. If None, phase data is not returned.

        
    Outputs
    -------
    Position_LL_dup : np.ndarray, shape (2M, 3)
        Positions for each duplicated contact (both A→B and B→A).
    
    Force_LL_dup : np.ndarray, shape (2M, 3)
        Forces for each duplicated contact (both A→B and B→A).  

    BranchVector_LL_dup : np.ndarray, shape (2M, 3)
        Branch vectors for each duplicated contact (both A→B and B→A).

    CenterToCenterVector_LL_dup : np.ndarray, shape (2M, 3)
        Center-to-center vectors for each duplicated contact (both A→B and B→A).

    Volume_LL_dup : np.ndarray or None, shape (2M,)
        Duplicated particle volumes if `Return_Volume` is True, else None.

    Phases_array : np.ndarray or None, shape (2M,)
        Duplicated particle phase data if provided, else None.
    
    Notes
    -----
    - The function assumes that `Global_ID` is sorted for efficient mapping.
    - Arrays are duplicated to account for both directions of each contact pair.
    - Branch vectors are computed using either contact points (if provided) or particle diameters.
    - Designed for use in granular material simulations and fabric tensor analysis.

    """

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

