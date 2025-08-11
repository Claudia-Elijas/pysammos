from coarse_graining import CoarseGraining
from utils.config_loader import load_config
import sys

# Terminal print output
sys.stdout = open("out.txt", "w")  

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
    dt_time_step=1,
    DEM_keymap=cfg["key_mapping"],
    grid_info=cfg["grid_info"],
    weight_function='Lucy',
    fields_to_export=cfg["fields_to_export"],
    ignore_phases=cfg["partialignore"]
                    )
print("  ") ; print("-------------------- CoarseGraining class initialised --------------------")
                        
# -------------------  COARSE-GRAINING WORK FLOW  ------------------------- # 

# 1. Load the size-relevant particle data for the first time step
Bounds_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0 = CG.data_sampling()

# 2. Calculate the particle size range
d43, d32 = CG.get_particle_size_statistics(Diameter_t0, Mass_t0)
print(">> Particle size statistics: ") 
print("       d43: ", d43)
print("       dmax: ", CG.dmax)
print("       d50: ", CG.d50)
print("       d32: ", d32)
print("       drms: ", CG.drms)

# 3. Get the phases
CG.get_particle_phases(Diameter_t0, Density_t0, GlobalID_t0, 8)
print(">> Phases: ")
print("       Diameters: ", CG.phases[:,0])
print("       Densities: ", CG.phases[:,1])

# 4. Calculate the CG grid spacing
CG.set_resolution(d43) # here you can input different number, to make w and c bigger or smaller 
print(">> Coarse Graining resolution: ")
print("       c:", CG.c)
print("       w:", CG.w)

# 5. Generate the CG grid
CG.generate_grid(smoothing_length=CG.c)
print(">> Grid: ")
print("       Grid Points: ", CG.GridPoints.shape, "First Point: ", CG.GridPoints[0])
print("       Nodes: ", CG.Nodes)
print("       Spacing: ", CG.Spacing)

# 6. Calculate the CG fields
CG.fields_in_time()

print("------------------------------------------------------------")
print(">>> Coarse Graining completed successfully <<<") 
print("------------------------------------------------------------")