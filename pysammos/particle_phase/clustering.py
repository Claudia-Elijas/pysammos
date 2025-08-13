"""
Particle Phase Identification and Visualization Module
======================================================

This module provides tools to identify distinct particle phases based on their physical properties,
specifically particle diameter and density, using unsupervised clustering techniques.

Functions:
----------
- find_phases:
    Performs KMeans clustering on scaled particle diameter and density data to determine the
    optimal number of phases (clusters) and assigns each particle to a phase.

- plot_phases:
    Visualizes the clustered particles in diameter-density space, color-coded by phase,
    along with the cluster centroids for easy interpretation.

The clustering approach incorporates silhouette score evaluation and curvature analysis to 
select the most meaningful number of phases, aiding in material characterization, mixture 
analysis, or other particulate system studies.
"""


# import relevant libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def find_phases(particle_diameters, particle_densities, n_max):
    """
    Determine particle phases by clustering diameter and density data using KMeans.

    This function scales the input data, estimates the optimal number of clusters 
    based on silhouette scores and the curvature of the score plot, then applies KMeans
    clustering to assign particles to phases.

    Parameters
    ----------
    particle_diameters : float array, shape(N_particles,) 
        Array of particle diameters.
    particle_densities : float array, shape(N_particles,) 
        Array of particle densities.
    n_max : int
        Maximum number of clusters to test for optimal clustering.

    Returns
    -------
    phases : float array, shape(n_clusters, 2) 
        Cluster centroids in the original feature space (diameter, density).
    phase_array : float array, shape(N_particles,)
        Cluster labels (phase indices) assigned to each particle.
    """
    
    # 1. data scaling
    X = np.column_stack((particle_diameters, particle_densities))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. find optimum number of clusters
    km_scores = []
    for i in range(2, n_max):
        labels = KMeans(n_clusters=i, random_state=200).fit_predict(X_scaled)
        score = metrics.silhouette_score(X_scaled, labels, metric='euclidean', sample_size=1000, random_state=200)
        km_scores.append(score)
    deriv2 = np.gradient(np.gradient(km_scores)) # second derivative
    deriv2_negative = np.where(deriv2 < 0)[0] # args with -tive 2nd deriv (i.e., maxima)
    optimum_clust = deriv2_negative[np.argmax(np.array(km_scores)[deriv2_negative])]+2 # args of maxima with highest score (i.e, absolute maximum)
    
    # 3. apply KMeans with optimum number of clusters
    km = KMeans(n_clusters=optimum_clust, random_state=200, n_init=100)
    phase_array = km.fit_predict(X_scaled)
    phases = scaler.inverse_transform(km.cluster_centers_)
        
    return phases, phase_array

def plot_phases(particle_diameters, particle_densities, phases, phase_array):
    """
    Plot particle diameters vs densities colored by phase clusters with cluster centroids.

    Parameters
    ----------
    particle_diameters :  float array, shape(N_particles,) 
        Array of particle diameters.
    particle_densities : float, shape(N_particles,) 
        Array of particle densities.
    phases : float arra, shape(n_clusters, 2) 
        Cluster centroids in diameter-density space.
    phase_array : int array, shape(N_particles,) 
        Cluster labels for each particle.

    Returns
    -------
    None
        Displays a scatter plot showing clustered particles and centroids.
    """

    plt.figure(figsize=(5, 6))
    plt.scatter(particle_diameters, particle_densities, c=phase_array, cmap='Set1', alpha=0.6, zorder=3)
    plt.scatter(phases[:, 0], phases[:, 1], c='black', s=200, marker='X', label='Centroids', zorder=4)
    plt.title('KMeans Clustering of Particle Density and Diameter')
    plt.xlabel('Diameter (m)')
    plt.ylabel('Density (kg/m^3)')
    plt.legend()
    plt.grid(zorder=0)
    plt.show()  # Display the plot