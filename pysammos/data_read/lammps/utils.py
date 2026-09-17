"""
This module contains utility functions for reading and processing LAMMPS data, such as checking for required/optional fields, sorting data, and extracting 
relevant columns for particle properties. 
It is used by the main data reading functions to ensure that the input data is properly formatted and contains the necessary information for further analysis.

"""


def _check_fields(data, required_fields, optional_fields):

    '''
    Checks that the required and optional fields are present in the data. If a required field is missing, raises an error. If an optional field is missing, prints a warning.

    Inputs
    ------
    data: pandas DataFrame containing the data to check.
    required_fields: dict mapping field names to column names (or lists of column names) that are required in the data. If a field is required, it must be present in the data.
    optional_fields: dict mapping field names to column names (or lists of column names) that are optional in the data. If a field is optional, it may be present in the data, but if it is not, a warning will be printed. 

    Outputs
    -------
    None. Raises an error if a required field is missing, and prints a warning if an optional field is missing.
    
    
    '''
    for field_name, s in required_fields.items():
        if s is None:
            print(f" >>> WARNING: '{field_name}' was not provided (None). \n  This means you have not provided it in the data. Please check the documentation to see if you will need to provide it or if it can be computed from other columns.")
            continue
        cols = s if isinstance(s, list) else [s]
        for col in cols:
            if col not in data.columns:
                raise ValueError(f" >>> ERROR: '{field_name}': column '{col}' not found in data. Available: {list(data.columns)}")

    for field_name, s in optional_fields.items():
        if s is None:
            print(f" >>> WARNING: '{field_name}' was not provided (None). \n  If you do have it, please provide it in the data. If not, it will be calculated in the next steps.")
            continue
        cols = s if isinstance(s, list) else [s]
        for col in cols:
            if col not in data.columns:
                raise ValueError(f" >>> ERROR: '{field_name}': column '{col}' not found in data. Available: {list(data.columns)}")
            