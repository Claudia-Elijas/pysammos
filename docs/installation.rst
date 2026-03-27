Installation
============

Welcome to the installation of Pysammos!

There are several ways to install Pysammos, depending on your preferences and needs. 


1. **Using GitHub repository**: You can clone the Pysammos repository from GitHub and install it locally. This method allows you to access the latest version of the code and contribute to its development if you wish.
    
    First, clone the repository and navigate to the project directory:

    .. code-block:: bash
        
        git clone https://github.com/Claudia-Elijas/pysammos.git
        cd pysammos
    
    Then, you can install the required dependencies from the environment yml file:

    .. code-block:: bash

        conda env create -f pysammos_env.yml
    
    And finally, install the package:

    .. code-block:: bash

        pip install -e .
    
    Note that the `-e` flag allows you to install the package in editable mode, which means that any changes you make to the code will be reflected in the installed package without needing to reinstall it. This is particularly useful for development and testing purposes.

For more details, please refer to the `Pysammos GitHub repository <https://github.com/Claudia-Elijas/pysammos>`_   


2. **Using pip**: You can install Pysammos using pip, which **will be available soon on PyPI**. This method is straightforward and allows you to easily manage the package and its dependencies.
     
    First make sure you have pip installed and updated:
    
    .. code-block:: bash

        pip install --upgrade pip
    
    Then, you can install Pysammos using pip:

    .. code-block:: bash 

        pip install pysammos





