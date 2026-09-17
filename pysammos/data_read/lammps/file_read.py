"""
This module provides functions to read LAMMPS dump files, which are text files containing simulation data. The main function is `read_dump`, which parses the file and returns the timestep, box bounds, and a DataFrame of atom data. The function is designed to be flexible and can handle both particle and contact data by specifying the `type` argument.
"""

import numpy as np
import pandas as pd

def read_dump(path, type="particles"):
    """Parse a LAMMPS dump file by searching for ITEM: keywords."""

    if type == "particles":
        expected_items = ["TIMESTEP", "NUMBER", "BOX", "ATOMS"]
    elif type == "contacts":
        expected_items = ["TIMESTEP", "NUMBER", "BOX", "ENTRIES"]
    else:
        raise ValueError("type must be 'particles' or 'contacts'.")
    

    # raise errors if file cannot be opened
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except PermissionError:
        raise PermissionError(f"Permission denied: {path}")
    except Exception as e:
        raise IOError(f"Error reading file {path}: {e}. Check if the path is correct, including the padding of zeros in the filename.")

    # Find key line indices
    item_map = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("ITEM:"):
            key = stripped.split("ITEM:")[1].strip().split()[0]  # e.g. TIMESTEP, BOX, ATOMS, NUMBER
            item_map[key] = i


    # Number of atoms
    n_atoms = int(lines[item_map[expected_items[1]] + 1].strip())

    # Box bounds — line is like "ITEM: BOX BOUNDS pp pp fm"
    bounds_start = item_map[expected_items[2]] + 1
    bounds = np.array([[float(v) for v in lines[bounds_start + i].split()] for i in range(3)])

    # Column headers — line is like "ITEM: ATOMS id type diameter ..."
    atoms_header_line = lines[item_map[expected_items[3]]].strip()
    headers = atoms_header_line.split(f"ITEM: {expected_items[3]}")[1].split()

    # Atom data
    atoms_start = item_map[expected_items[3]] + 1
    data = np.genfromtxt(lines[atoms_start:atoms_start + n_atoms], dtype=float)

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    return bounds, df