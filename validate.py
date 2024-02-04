import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score

def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
    - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
    - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
    """

    # TODO: Compute pairwise distances
#     m,n = data.shape
#     pairwise_distances = np.linalg.norm(np.tile(data.reshape((m,1,n)),(1,m,1)) - data,axis=2)
    n = data.shape[0]
    affinity_matrix = np.zeros((n, n))
    pairwise_distances = np.linalg.norm(data[:, None] - data, axis=2)

    if affinity_type == 'knn':
        # differences = data[:, :, np.newaxis] - data[:, np.newaxis, :]
        # pairwise_distances = np.sqrt(np.sum(differences**2, axis=1))
        pairwise_distances = np.einsum('ij,ij->i',data,data)[:,None] + np.einsum('ij,ij->i',data,data) - 2*np.dot(data,data.T)

        # Ensure symmetry
        # symmetric_distances = 0.5 * (pairwise_distances + pairwise_distances.T)

#         differences = data[:, :, np.newaxis] - data[:, np.newaxis, :]
#         pairwise_distances = np.sqrt(np.sum(differences**2, axis=1))
#         print(pairwise_distances.shape)
#         differences = data[:, :, np.newaxis] - data[:, np.newaxis, :]
#         pairwise_distances = np.sqrt(np.sum(differences**2, axis=0))
        # Get indices of k nearest neighbors for each vector
        k_neighbors_indices = np.argsort(pairwise_distances, axis=1)[:, 1:k+1]

        # Create a binary matrix indicating the neighbors
        binary_matrix = np.zeros_like(pairwise_distances, dtype=int)
        rows, cols = np.indices(binary_matrix.shape)
        binary_matrix[rows, k_neighbors_indices] = 1

        return binary_matrix
        # return affinity_matrix
#         # Compute pairwise distances without using sklearn
# #         distances = np.sqrt(((data[:, np.newaxis] - data) ** 2).sum(axis=2))
#         np.fill_diagonal(pairwise_distances, np.inf)  # Exclude self-distance
#         indices = np.argsort(pairwise_distances, axis=1)[:, :k]
# 
#         # Compute affinity matrix using broadcasting
#         d_ij = pairwise_distances[np.arange(n)[:, None], indices]  # Distance to the k-nearest neighbors
#         affinity_matrix[np.arange(n)[:, None], indices] = np.exp(-d_ij / (2 * sigma ** 2))
#         affinity_matrix[indices, np.arange(n)[:, None]] = affinity_matrix[np.arange(n)[:, None], indices]
        # TODO: Find k nearest neighbors for each point

        # TODO: Construct symmetric affinity matrix

        # TODO: Return affinity matrix
        
        # TODO: Find k nearest neighbors for each point
#         for i in range(len(data)):
#             
#             distances = LA.norm(data - data[i], axis=1) 
#             k_neares_neighbors_indexes = np.argpartition(distances, k)[:k]
# 
#             affinity_matrix[i, k_neares_neighbors_indexes] = 1
#             affinity_matrix[k_neares_neighbors_indexes, i] = 1
#         return affinity_matrix


    elif affinity_type == 'rbf':
        # Compute Gaussian-based similarity using broadcasting
        affinity_matrix = np.exp(-((pairwise_distances**2) / (2 * sigma ** 2)))
        # affinity_matrix = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # TODO: Apply RBF kernel

        # TODO: Return affinity matrix
        return affinity_matrix
#         for i in range(n):
#             for j in range(n):
#                 affinity_matrix[i, j] = np.exp(-((LA.norm(data[i] - data[j]))**2) / (2 * (sigma**2)))
#         return affinity_matrix

    else:
        raise Exception("invalid affinity matrix type")

if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']
    # TODO: Create and configure plot
    fig, plots = plt.subplots(3,4,figsize=(14,10))
    for i, ds_name in enumerate(datasets):
        X = np.load("./datasets/%s/data.npy" % ds_name)
        y = np.load("./datasets/%s/target.npy" % ds_name)
        
        n = len(np.unique(y)) # number of clusters
        k = 4
        sigma = 1.0

        y_km, _ = k_means_clustering(X, n)

        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)

        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))

        
        plots[i][0].scatter(X[:,0],X[:,1],c= y,marker= 'o')
        plots[i][0].set_title("ground truth for %s" %ds_name)

        plots[i][1].scatter(X[:,0],X[:,1],c= y_km,marker= 'o')
        plots[i][1].set_title("kmeans clusteration for %s" %ds_name)

        plots[i][2].scatter(X[:,0],X[:,1],c= y_rbf,marker= 'o')
        plots[i][2].set_title("rbf clusteration for %s" %ds_name)

        plots[i][3].scatter(X[:,0],X[:,1],c= y_knn,marker= 'o',)
        plots[i][3].set_title("knn clusteration for %s" %ds_name)   

        # TODO: Create subplots
    plt.show()
    # TODO: Show subplots

# if __name__ == "__main__":
#     datasets = ['blobs', 'circles', 'moons']
# 
#     # TODO: Create and configure plot
# 
#     for ds_name in datasets:
#         dataset = np.load("datasets/%s.npz" % ds_name)
#         X = dataset['data']     # feature points
#         y = dataset['target']   # ground truth labels
#         n = len(np.unique(y))   # number of clusters
# 
#         k = 3
#         sigma = 1.0
# 
#         y_km, _ = k_means_clustering(X, n)
#         Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
#         y_rbf = spectral_clustering(Arbf, n)
#         Aknn = construct_affinity_matrix(X, 'knn', k=k)
#         y_knn = spectral_clustering(Aknn, n)
# 
#         print("K-means on %s:" % ds_name, clustering_score(y, y_km))
#         print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
#         print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))
# 
#         # TODO: Create subplots
# 
#     # TODO: Show subplots
