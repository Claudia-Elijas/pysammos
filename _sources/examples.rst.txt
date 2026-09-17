Tutorials and Examples
======================

Here are some example usages of Pysammos's coarse-graining framework, from the building and explanation of the configuration files to the visualisation of the coarse-grained fields.

The first Pysammos release (v1.0.0) includes the various examples for DEM models in MFiX. Below is an example of a 2D bedload transport simulation. 

.. toctree::
   :maxdepth: 2

   examples/bedload_transport



In the second Pysammos release (v1.1.0), the examples for DEM models in LAMMPS are included. Below is an example of a 3D granular column collapse simulation.
This example shows how to enter the inputs to the ``CoarseGraining`` class. The subsequent visualisation of the coarse-grained fields is not included in the example,
but can be implemented from other MFiX examples found in the documentation or the `examples` directory.

.. toctree::
   :maxdepth: 2

   examples/cg_from_lammps_files


.. note:: 
   The data files for the examples are not included in the repository due to their large size (~1.3GB). 

   You can either:

      - Download the folder data_examples.zip manually from Zenodo (https://doi.org/10.5281/zenodo.22813684) and place the data_examples/ folder inside examples/. 

      - Let pysammos fetch it automatically by running the data_fetch.py script in this folder, or:

      - Use the following Python code to fetch the data:

         ..   code-block:: python

            from pysammos.data import fetch_example_data
            fetch_example_data()
