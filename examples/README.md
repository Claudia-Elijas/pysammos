# Example Data

The example dataset (~1.3GB) is not bundled with the package.
You can either:

1. Download it manually from `GitHub Releases <link>`_ and place it in ``examples/data_examples/``
2. Let pysammos fetch it automatically by running the ``data_fetch.py`` script in this folder, or: 

    ```
    from pysammos.data import fetch_example_data
    fetch_example_data()

    ```

## Example Simulations

In this folder you will find the following example simulations: 

1. Bedload transport:
    - 2D
    - non-spherical particles
    - polydisperse
    - DEM-CFD coupling
    - data analysis in Python

2. Erodible bed
    - 3D
    - spherical particles
    - polydisperse
    - no fluid
    - data analysis in Python

3. Impact cratering
    - 3D 
    - spherical particles
    - polydisperse
    - no fluid

4. Jannsen pile
    - 2D
    - spherical particles
    - polydisperse
    - no fluid
    - data analysis in Python

5. Magma conduit
    - 2D
    - non-spherical particles
    - monodisperse
    - viscous fluid
    


