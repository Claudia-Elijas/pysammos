# setup file
from setuptools import setup

setup(
    name='pysammos',
    version='0.1',
    description='Python package to compute macroscopic continuum fields from discrete element method simulations.',
    author='Claudia Elijas-Parra',
    author_email='claudia.elijas.parra@gmail.com',
    packages=['pysammos'],
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.2.3",
        "scipy>=1.15.3",
        "scikit-learn>=1.6.1",
        "matplotlib>=3.10.0",
        "pyvista>=0.45.2",
        "vtk>=9.4.2",
        "vtk-hdf>=0.2.0",
        "requests>=2.32.3",
        "pooch>=1.8.2",
        "scooby>=0.10.1",
        "typing-extensions>=4.14.0"
    ],
    license="GPL-3.0",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # supported Python versions
        "Programming Language :: Python :: 3",
        # license
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        # operating systems 
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        # indicate the type of application
        "Topic :: Scientific/Engineering :: Physics"
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.12',

)