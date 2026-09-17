Example 2: using Pysammos's coarse-graining capabilities with LAMMPS output
===========================================================================

In this example, we demonstrate how to use the ``CoarseGraining`` to perform coarse graining using LAMMPS output.
In order to do that, you can use the :mod:`compute_CG.ipynb` Jupyter notebook provided in the `examples/LAMMPS_reading` directory. 

The files required for this example can be found in the `examples/LAMMPS_reading` directory of the repository:

- The DEM simualtion data files in the `examples/LAMMPS_reading/DEM_data` directory
- The configuration file `config.ini` in the `examples/LAMMPS_reading` directory


The Configuration File
----------------------

The configuration file `config.ini` contains the parameters for the coarse-graining process. You can modify the parameters in this file to suit your needs. 
For further details on the parameters, please refer to the :doc:`the bedload transport example <bedload_transport>`. 


.. code-block:: ini

    [paths]
    particles_path = ../data_examples/lammps_reading/DEM_data/dump_particles/particles.
    contacts_path = ../data_examples/lammps_reading/DEM_data/dump_forcechain/forcechain.
    output_path = ./PysammosCG/

    [timesteps] 
    t0 = 11991
    tf = 11991 
    dt = 1
    t_pad = 9 

    [smoothing_function]
    type = Lucy

    [flags]
    partialignore = True

    [key_mapping]
    Global_ID = id
    Particle_Position = [x, y, z]
    Particle_Velocity = [vx, vy, vz]
    Particle_Diameter = diameter
    Particle_Density = v_pdens
    Particle_Volume = v_pvol
    Particle_Mass = mass
    Particle_Radius = radius
    Coordination_Number = c_Z
    Particle_i_ID = c_pl[1]
    Particle_j_ID = c_pl[2]
    Force_ij = [c_pr[1], c_pr[2], c_pr[3]]
    Contact_ij = None

    [DEM_software]
    name = lammps
    
    [grid_info]
    grid_dimension = 3
    grid_axes = xyz
    automatic_grid = True
    x_min = 0.005
    x_max = 0.10
    y_min = 0.005
    y_max = 0.44
    z_min = 0.005
    z_max = 0.05
    x_transect = None
    y_transect = None
    z_transect = None
    x_axis_periodic = True
    y_axis_periodic = False
    z_axis_periodic = True

    [fields_to_export]
    volume_fraction = True
    density_particle = False
    density_mixture = True
    momentum_density = False
    velocity = True
    velocity_gradient = False
    kinetic_tensor = False
    contact_tensor = False
    total_stress_tensor = True
    pressure = True
    granular_temperature = True
    granular_temperature_slices = True
    fabric_tensor = False
    inertial_number = True
    coordination_number = False
    d43 = False
    d32 = False
    frictional_coefficient = True
    shear_rate_tensor = False

    [output_options]
    vkthdf_output = True
    h5_output = True


Initialising the CoarseGraining Class
-------------------------------------

First load the relevant modules:

.. code-block:: python

    from pysammos.utils.config import load_config
    from pysammos.coarse_graining import CoarseGraining


Then initialise the ``CoarseGraining`` class with the parameters from the configuration file:

.. code-block:: python

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
        t_pad=cfg["t_pad"],
        DEM_keymap=cfg["key_mapping"],
        DEM_software=cfg["DEM_software"],
        grid_info=cfg["grid_info"],
        weight_function=cfg["smoothing_function"],
        fields_to_export=cfg["fields_to_export"],
        ignore_phases=cfg["partialignore"],
        h5_output=cfg["h5_output"],
        vkthdf_output=cfg["vkthdf_output"]) 
    

And finally, run the coarse-graining process:

.. code-block:: python

    CG.run()



Alternatively, you can run each step of the coarse-graning workflow separately, as shown in the :doc:`the bedload transport example <bedload_transport>`.