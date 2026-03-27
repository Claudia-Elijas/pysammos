# Pysammos

Granular flows are widespread in nature and industry, yet poorly understood. Discrete particle simulations offer detailed insight into their behaviour, but only provide properties of the individual particles, such as velocity and force. To obtain bulk flow quantities relevant to science and engineering, such as pressure or strain rate, 
we employ a mathematical framework called Coarse-Graining (CG).

Pysammos is a Python package designed with the outlook of providing a user-friendly Coarse-Graining workflow to post-process data form the MFiX open-source DEM software, and provide a streamlined visualisation in widely-used open-source visualisation software, Paraview. This code package provides flexibility in the output variable selection, mesh parametrisation, and data analysis of the obtained results. Pysammmos is also designed to invite geoscientists to incorporate DEM in their research, as many processes studied in geosciences involve discrete elements that are currently modelled as a continuum could benefit from a discrete insight. Similarly, to enhance DEM analysis by extracting continuum fields without the need to handle inner-level source code. The efficient algorithmic complexity exhibited by Pysammos avoids the requirement of extensive computational resources, making it a programme that can be ran on at the same time as other processes.

This library is distributed under the GNU General Public License v3.0. Contributions are welcome, and can be made via pull requests on the GitHub repository.

## Documentation

Find detailed documentation of the code here: 

https://Claudia-Elijas.github.io/pysammos/

## Installation

There are several ways to install Pysammos, depending on your preferences and needs.

### 1. Using the GitHub Repository

You can clone the Pysammos repository from GitHub and install it locally. This method allows you to access the latest version of the code and contribute to its development if you wish.

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Claudia-Elijas/pysammos.git
cd pysammos
```

Create a conda environment from the `pysammos_env.yml` file:
```bash
conda env create -f pysammos_env.yml
```

Activate the environment:
```bash
conda activate Pysammos_Env
```

Install the package in editable mode:
```bash
pip install -e .
```

> **Note:** The `-e` flag installs the package in editable mode, meaning any changes you make to the source code will be reflected immediately without needing to reinstall. You may omit `-e` if you are not developing the code.

### 2. Using pip *(coming soon)*

Pysammos will be available on PyPI soon. Once available, you can install it with:
```bash
pip install --upgrade pip
pip install pysammos
```

---

## Quick Start 

Once installed, you are ready to run the examples. Note that you will need to download the example data separately — see the [Examples README](examples/README.md) for instructions.

Navigate to the bedload transport example:
```bash
cd examples/bedload_transport
```

Run as a Python script:
```bash
python3 compute_CG.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook compute_CG.ipynb
```

---


