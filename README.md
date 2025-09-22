# Pysammos

**Pysammos** is a Python package to compute macroscopic continuum fields from discrete element method simulations.

This library is distributed under the GNU General Public License v3.0. Contributions are welcome, and can be made via pull requests on the GitHub repository.

## Quick Installation:

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
4. Now you are ready to run the example in `./examples/bedload_transport/`. There's the option to run it in a Jupyter notebook (`.ipynb`) or in a Python script (`.py`). To run the Python script, simply:

   ```
   python3 ./examples/bedload_transport/compute_CG.py
   ```

## Documentation:

I have started creating documentation html from the docstrings in my code, you can find it in:

`./build/html/index.html`
