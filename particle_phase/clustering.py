import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def test():
    print("This is a test function for particle phase clustering.")
    print(np.arange(10))  # Example operation to test the import

def find_phases(particle_diameters, particle_densities, n_max):
    
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

    plt.figure(figsize=(5, 6))
    plt.scatter(particle_diameters, particle_densities, c=phase_array, cmap='Set1', alpha=0.6, zorder=3)
    plt.scatter(phases[:, 0], phases[:, 1], c='black', s=200, marker='X', label='Centroids', zorder=4)
    plt.title('KMeans Clustering of Particle Density and Diameter')
    plt.xlabel('Diameter (m)')
    plt.ylabel('Density (kg/m^3)')
    plt.legend()
    plt.grid(zorder=0)
    plt.show()  # Display the plot