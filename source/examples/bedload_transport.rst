Bedload Transport Example
=========================


In this example, we demonstrate how to use the `CoarseGraining` to perform coarse graining on a bedload transport simulation and plot the results.
In order to do that, you can use the `compute_CG.ipynb` Jupyter notebook provided in the `examples/bedload_transport` directory. 

The files required for this example can be found in the `examples\bedload_transport` directory of the repository:

- The DEM simualtion data files in the `examples\bedload_transport\VTU` directory
- The configuration file `config.ini` in the `examples\bedload_transport` directory

Coarse Graining Steps
----------------------

First, ensure you have the necessary libraries installed:

.. code-block:: python

   
    import numpy as np
    from pysammos.utils.config_loader import load_config
    from pysammos.coarse_graining import CoarseGraining


We should initialize the configuration and the coarse graining object:

.. code-block:: python
    
    # Load the configuration from the ini file
    cfg = load_config("config.ini")  

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
        ignore_phases=cfg["partialignore"]
                        ) 

Next, we can perform the coarse graining on the simulation data:

1. Load the size-relevant particle data for the first time step

.. code-block:: python
    
    Bounds_t0, Diameter_t0, Density_t0, Mass_t0, GlobalID_t0 = CG.data_sampling()


2. Calculate the particle size range

.. code-block:: python
    
    CG.get_particle_size_statistics(Diameter_t0, Mass_t0)
    print(">> Particle size statistics: ") 
    print("       d43: ", CG.d43)
    print("       dmax: ", CG.dmax)
    print("       d50: ", CG.d50)
    print("       d32: ", CG.d32)
    print("       drms: ", CG.drms)


3. Get the phases

.. code-block:: python
    
    CG.get_particle_phases(Diameter_t0, Density_t0, GlobalID_t0, 8)
    print(">> Phases: ")
    print("       Diameters: ", CG.phases[:,0])
    print("       Densities: ", CG.phases[:,1])

4. Calculate the CG grid spacing

.. code-block:: python
    
    CG.set_resolution(CG.d43) # here you can input different number, to make w and c bigger or smaller 
    print(">> Coarse Graining resolution: ")
    print("       c:", CG.c)
    print("       w:", CG.w)

5. Generate the CG grid

.. code-block:: python
    
    CG.generate_grid()
    print(">> Grid: ")
    print("       Grid Points: ", CG.GridPoints.shape, "First Point: ", CG.GridPoints[0])
    print("       Nodes: ", CG.Nodes)
    print("       Spacing: ", CG.Spacing)

6. Calculate the CG fields

.. code-block:: python
    
    CG.fields_in_time()


Plotting the Results
--------------------

After running the above code, the coarse-grained fields will be saved in the specified output directory.
To visualize the results, you can use the provided Jupyter notebook `visualization_bedload_transport.ipynb` in the same directory.

1. Import necessary libraries

.. code-block:: python
    
    from pysammos.data_write.h5.writer import H5XarrayManager
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
    import pyvista as pv
    from matplotlib.lines import Line2D

2. Initialize the H5XarrayManager to read the coarse-grained data

.. code-block:: python
    
    # gridded data
    manager = H5XarrayManager("./PysammosCG/CG_Lucy_Monodisperse.h5")
    bedload_CG = manager.h5_to_xarray()

    # sliced granular temperature data
    manager_gt = H5XarrayManager("./PysammosCG/CG_GranularTemperature_slices.h5")
    slices_gt = manager_gt.h5_to_xarray()

3. Plotting the mixture density field at a specific time step

.. code-block:: python
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

    # Plot DEM particle positions 
    bedload_DEM = pv.read('./VTU/DES_FB1_0150.vtp')
    bedload_particles = bedload_DEM.points
    x_particle = bedload_particles[:, 0]; y_particle = bedload_particles[:, 1]
    ax.scatter(x_particle, y_particle, s=10, c='k', alpha=1, label='Particle Positions', zorder=3)

    # Plot the CG data
    # CG coordinates at z = 0.0
    x = np.asarray(bedload_CG.positions[:, 0]) ; y = np.asarray(bedload_CG.positions[:, 1]) ; z = np.asarray(bedload_CG.positions[:, 2])
    var = np.asarray(bedload_CG.density_mixture.values[0, :])
    mask = np.asarray(bedload_CG.positions[:, 2]) == 0.0 # get the points that have z = 0.0
    x_slice = x[mask] ; y_slice = y[mask] ; var_slice = var[mask]   
    # plot the CG data
    sc = ax.scatter(x_slice, y_slice, s=65, c=var_slice, 
                    cmap='viridis', alpha=0.6, label='CG Density', 
                    zorder=1, marker='s')

    plt.colorbar(sc, ax=ax, label='Density')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(' Coarse-Grained Mixture Density')
    plt.show()

.. image:: /_static/mixdens.png
   :width: 600px
   :align: center
   :alt: Granular temperature vertical profile

4. Plotting the granular temperature field at a specific time step

.. code-block:: python
    
    # plot first time step vertical profile comparing Kamrin and LAMMPS
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(slices_gt.granular_temperature_Kamrin[0,:,0], slices_gt.positions[:, 1], label='Granular Temperature Kamrin', color='blue')
    ax.plot(slices_gt.granular_temperature_LAMMPS[0,:,0], slices_gt.positions[:, 1], label='Granular Temperature LAMMPS', color='orange')
    ax.set_xlabel('Granular Temperature')
    ax.set_ylabel('Y Position')
    ax.set_title('Granular Temperature Vertical Profile at z=0, Timestep 0')
    #gridlines
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    # log scale x axis
    ax.set_xscale('log')
    ax.legend()
    plt.show()  

.. image:: /_static/output.png
   :width: 600px
   :align: center
   :alt: Granular temperature vertical profile