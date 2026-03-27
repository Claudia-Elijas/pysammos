# Pysammos

Granular flows are widespread in nature and industry, yet poorly understood. Discrete particle simulations offer detailed insight into their behaviour, but only provide properties of the individual particles, such as velocity and force. To obtain bulk flow quantities relevant to science and engineering, such as pressure or strain rate, 
we employ a mathematical framework called Coarse-Graining (CG).

Pysammos is a Python package designed with the outlook of providing a user-friendly Coarse-Graining workflow to post-process data form the MFiX open-source DEM software, and provide a streamlined visualisation in widely-used open-source visualisation software, Paraview. This code package provides flexibility in the output variable selection, mesh parametrisation, and data analysis of the obtained results. Pysammmos is also designed to invite geoscientists to incorporate DEM in their research, as many processes studied in geosciences involve discrete elements that are currently modelled as a continuum could benefit from a discrete insight. Similarly, to enhance DEM analysis by extracting continuum fields without the need to handle inner-level source code. The efficient algorithmic complexity exhibited by Pysammos avoids the requirement of extensive computational resources, making it a programme that can be ran on at the same time as other processes.

This library is distributed under the GNU General Public License v3.0. Contributions are welcome, and can be made via pull requests on the GitHub repository.

## Documentation

Find detailed documentation of the code here: 

https://Claudia-Elijas.github.io/pysammos/

## Quick Installation:

To install Pysammos in your local machine from GitHub, follow the steps below:

1. Create a conda environment from the `pysammos_env.yml` file:

   ```
   `conda env create -f pysammos_env.yml`
   ```
2. Activate the environment:

   ```
   conda activate Pysammos_Env
   ```
3. Install `pysammos` locally in the `./pysammos/` directory, which contains the code folder `pysammos` and other folders (e.g., `examples`). 

   ```
   pip install -e .
   ```
Note that the -e allows you run Pysammos after editting the source code. You may ommit the -e if you are not going to be developing the code.

4. Now you are ready to run the example in `./examples/bedload_transport/`. Note that you will need to fetch the example data as described in the README.md file. There's the option to run it in a Jupyter notebook (`.ipynb`) or in a Python script (`.py`). To run the Python script, simply:

   ```
   python3 ./examples/bedload_transport/compute_CG.py
   ```


