Tutorials and Examples
======================

Here are some example usages of Pysammos's coarse-graining framework, from the building and explanation of the configuration files to the visualisation of the coarse-grained fields.

.. note::
   These examples are kept up to date with the ``main`` branch on GitHub, not with a
   specific release. If something doesn't match your installation, update to the
   latest code with ``git pull`` and ``pip install -e .``. 


The first Pysammos release (v1.0.0) includes the various examples for DEM models in MFiX, they have been updated to run with the latest status of pysammos. 

.. nbgallery::

   examples/bedload_docs.ipynb
   examples/erodible_bed_docs.ipynb
   examples/janssen_pile_docs.ipynb


In the second Pysammos release (v1.1.0), the examples for DEM models in LAMMPS are included. Below is an example of a 3D granular column collapse simulation.
This example shows how to enter the inputs to the ``CoarseGraining`` class. The subsequent visualisation of the coarse-grained fields is not included in the example,
but can be implemented from other MFiX examples found in the documentation or the `examples` directory.

.. nbgallery::

   examples/lammps_reading_docs.ipynb


.. warning::
   Once you have run the coarse-graining workflow, re-running it with a different
   number of time steps, spatial scale or set of variables requires a **new output
   folder**: change ``output_path`` in ``config.ini``, or delete the previous results first.

.. note:: 
   The data files for the examples are not included in the repository due to their large size (~1.3GB). 

   You can either:

      - Download the folder data_examples.zip manually from Zenodo (https://doi.org/10.5281/zenodo.22813684) and place the data_examples/ folder inside examples/. 

      - Let pysammos fetch it automatically by running the data_fetch.py script in this folder, or:

      - Use the following Python code to fetch the data:

         ..   code-block:: python

            from pysammos.data import fetch_example_data
            fetch_example_data()
