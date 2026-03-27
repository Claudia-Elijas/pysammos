Particle Phase
==============

pysammos.particle_phase package

Subpackage for particle phase analysis.


Mixture theory conceives granular mixtures to be populated by all phases with associated partial macroscopic fields contributing to a bulk field, where a phase can be defined as any particle property, 
or combination of properties, such as density, size, shape, elasticity, roughness.  
In Pysammos we consider a phase to be defined by particle diameter and density, since they two properties directly related to the two main competing mechanisms in segregation: kinetic sieving and 
buoyancy, two ubiquitous processes in natural flows. 

In Pysammos the phase detection is carried out by clustering particle data using the k-means algorithm (see figure below). 
The optimum number of clusters is chosen to be that with the absolute maximum in silhouette value, a measure for how well each point matches its assigned cluster relative to the nearest neighbouring cluster (d in figure below).
Afterwards, the k-means algorithm is applied to the dataset with the optimum number of clusters (e in figure below) to obtain the phase-ascribed arrays, which are stored as attributes of the class instance. 


.. figure:: _static/Clustering.png
   :alt: Clustering
   :align: left
   :width: 600px

   **Example of particle phase clustering.** Illustration of the approach to detect particle phases. Synthetic data set of a bimodal particle diameter distribution (a). 
   Synthetic data set of a bimodal particle density distribution (b). Cross-plot corresponding to the diameter and density distributions in (a) and (b), respectively (c). Cross-plot containing the same particle samples as (c) coloured by the phase they are classified to according to the k-means algorithm using the optimum number of clusters (d). Silhouette score for a given number of clusters employed in the k-means algorithm (grey), and the corresponding second derivative (black) (e). The absolute maximum in (e) is chosen to be the optimum number of clusters (dashed red).


.. automodule:: pysammos.particle_phase
   :members:
   :undoc-members:
   :show-inheritance:

Clustering module
-----------------
pysammos.particle\_phase.clustering module

.. automodule:: pysammos.particle_phase.clustering
   :members:
   :undoc-members:
   :show-inheritance:
