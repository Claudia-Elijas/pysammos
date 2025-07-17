# particle clustering
from particle_phase.clustering import test
#from particle_phase.clustering import test
test()

# spatial weights
import spatial_weights.kernels as kernels
c = 1.0
d = 0.5
print("Lucy Weight:", kernels.Lucy(c, d))

# neighbour search
from neighbour_search.grid_particle_search import *

# grid generation
from grid_generation import regular_cuboid 
g = regular_cuboid.Grid_Generation ; print(g)

# macroscopic field calculation

# contact data handling

# data read

# data write

