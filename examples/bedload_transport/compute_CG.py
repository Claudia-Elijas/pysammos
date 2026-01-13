# Terminal print output
import sys
sys.stdout = open("out.txt", "w")

# Set the number of threads for Numba to use
import os
os.environ["NUMBA_NUM_THREADS"] = "8"

# Import packages
from pysammos.utils.config_loader import load_config
from pysammos.coarse_graining import CoarseGraining
import numba 
print(f">>> Numba is using {numba.get_num_threads()} cores")
import time

# Start the timer
start_time = time.time()

# Load the configuration from the ini file
cfg = load_config("config.ini")  
print("-------------------- config.ini file read --------------------")

# Initialize the CoarseGraining class with the loaded configuration
CG = CoarseGraining(
    particle_path=cfg["particles_path"],
    contacts_path=cfg["contacts_path"],
    output_path=cfg["output_path"],
    start_timestep=cfg["t0"],
    end_timestep=cfg["tf"],
    dt_time_step=cfg["dt"],
    DEM_keymap=cfg["key_mapping"],
    grid_info=cfg["grid_info"],
    weight_function=cfg["smoothing_function"],
    fields_to_export=cfg["fields_to_export"],
    ignore_phases=cfg["partialignore"],
    vtk_hdf_output=cfg["vkthdf_output"],
    h5_output=cfg["h5_output"],

                    ) 
print("  ") ; print("-------------------- CoarseGraining class initialised --------------------")
                        
# -------------------  COARSE-GRAINING WORK FLOW  ------------------------- # 

# 1. Load the size-relevant particle data for the first time step
Bounds_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0 = CG.data_sampling()

# 2. Calculate the particle size range
CG.get_particle_size_statistics(Diameter_t0, Mass_t0)
print(">> Particle size statistics: ") 
print("       d43: ", CG.d43)
print("       dmax: ", CG.dmax)
print("       d50: ", CG.d50)
print("       d32: ", CG.d32)
print("       drms: ", CG.drms)

# 3. Get the phases
CG.get_particle_phases(Diameter_t0, Density_t0, GlobalID_t0, 8)
print(">> Phases: ")
print("       Diameters: ", CG.phases[:,0])
print("       Densities: ", CG.phases[:,1])

# 4. Calculate the CG grid spacing
CG.set_resolution(CG.d43) # here you can input different number, to make w and c bigger or smaller
print(">> Coarse Graining resolution: ")
print("       c:", CG.c)
print("       w:", CG.w)

# 5. Generate the CG grid
CG.generate_grid()
print(">> Grid: ")
print("       Grid Points: ", CG.GridPoints.shape, "First Point: ", CG.GridPoints[0])
print("       Nodes: ", CG.Nodes)
print("       Spacing: ", CG.Spacing)
 
# 6. Calculate the CG fields
CG.fields_in_time()

# end the timer
end_time = time.time()
elapsed_time = end_time - start_time
print(" ")
print(f"Coarse Graining completed in {elapsed_time:.2f} seconds")
print(" ")

print("------------------------------------------------------------")
print(">>> Coarse Graining completed successfully <<<") 
print("------------------------------------------------------------")