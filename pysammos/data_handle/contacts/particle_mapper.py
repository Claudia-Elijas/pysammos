"""
Particle Mapper Module
===================

This module provides functionality to map global particle data (positions, diameters, etc.) to contact
pair data, duplicates the arrays to account for both directions (A→B and B→A),
and computes the branch vectors using either contact points or particle diameters.
It also includes options to return particle volumes for fabric tensor calculations.

Functions
--------
- `map_contact_data`: Maps particle and contact data for branch vector and fabric tensor analysis.
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
    
    Map particle and contact data for branch vector and fabric tensor analysis.

    This function maps global particle data (positions, diameters, etc.) to contact
    pair data, duplicates the arrays to account for both directions (A→B and B→A),
    and computes the branch vectors using either contact points or particle diameters.

    If `Return_Volume` is `True`, particle volumes will also be included in the output
    for use in computing a weighted fabric tensor.

    Parameters
    ----------
    Global_ID : ndarray, shape (N,).
        Array of global particle IDs, used to map contact pairs to particle data.

    Position : ndarray, shape (N, 3).
        Particle positions.

    Diameter : ndarray, shape (N,).
        Particle diameters.

    Density : ndarray, shape (N,).
        Particle densities.

    Volume : ndarray, shape (N,).
        Particle volumes, used for fabric tensor calculations if `Return_Volume` is `True`.

    Particle_LL : ndarray, shape (M,).
        Particle A IDs for contact pairs.

    Particle_I : ndarray, shape (M,).
        Particle B IDs for contact pairs.

    Fij : ndarray, shape (M, 3).
        Contact forces between particle pairs (A, B).

    Contact_Points : ndarray or None, shape (M, 3), optional.
        Contact point coordinates. If `None`, diameters are used to estimate branch vectors.

    ModelAxesRanges : ndarray, shape (3,).
        Domain dimensions.

    AxesPeriodicity : ndarray of bool, shape (3,).
        Periodicity flags for each axis (True/False).

    Return_Volume : bool
        Whether to include volume data in the output.

    Particle_Phase_Array_t : ndarray or None
        Optional array of phase identifiers per particle.

    Returns
    -------
    Position_LL_dup : ndarray, shape (2M, 3).
        Positions for each duplicated contact.

    Force_LL_dup : ndarray, shape (2M, 3).
        Contact forces, duplicated (including negative counterparts).

    BranchVector_LL_dup : ndarray, shape (2M, 3).
        Computed branch vectors from each particle to a contact point.

    CenterToCenterVector_LL_dup : ndarray, shape (2M, 3).
        Center-to-center vectors between particle pairs in both directions.

    Volume_LL_dup : ndarray or None, shape (2M,).
        Duplicated particle volumes if `Return_Volume` is True.

    Phases_array : ndarray or None, shape (2M,).
        Duplicated particle phase data if provided.

    Mean_Diameter : float
        Average diameter across all duplicated particles.
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

