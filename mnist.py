import numpy as np
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score


def chamfer_distance(point_cloud1, point_cloud2):
    pairwise_distances1 = np.sum((point_cloud1[:, None] - point_cloud2)**2, axis=-1)  
    nearest_neighbor_indices1 = np.argmin(pairwise_distances1, axis=1)  
    nearest_neighbor_distances1 = np.min(pairwise_distances1, axis=1) 
    pairwise_distances2 = np.sum((point_cloud2[:, None] - point_cloud1)**2, axis=-1)  
    nearest_neighbor_indices2 = np.argmin(pairwise_distances2, axis=1)  
    nearest_neighbor_distances2 = np.min(pairwise_distances2, axis=1)  
    chamfer_distance = np.mean(nearest_neighbor_distances1) + np.mean(nearest_neighbor_distances2)
    return chamfer_distance

def rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centered_A = A - centroid_A
    centered_B = B - centroid_B
    H = centered_A.T@centered_B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T@ U.T
    t = centroid_B - R@ centroid_A
    return R, t

# def chamfer_distance(point_cloud1, point_cloud2):
#     """
#     Calculate the Chamfer distance between two point clouds.
# 
#     Parameters:
#     - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
#     - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.
# 
#     Returns:
#     - dist: float, the Chamfer distance between the two point clouds.
#     """
#     
#     # TODO: Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2
#     pairwise_distance = np.einsum('ij,ij->i',point_cloud1,point_cloud1)[:,None] + np.einsum('ij,ij->i',point_cloud2,point_cloud2) - 2*np.dot(point_cloud1,point_cloud2.T)
#     distances1 = min((((point_cloud1[:, None, :] - point_cloud2)  2).sum(axis=-1)), axis=1)
#     distances2 = min(sqrt(((point_cloud2[:, None, :] - point_cloud1)  2).sum(axis=-1)), axis=1)
#     distance_1_to_2 = 0
#     distance_2_to_1 = 0
#     points1 = np.column_stack((point_cloud1['X'], point_cloud1['Y'], point_cloud1['Z']))
#     points2 = np.column_stack((point_cloud2['X'], point_cloud2['Y'], point_cloud2['Z']))
#     # points2 = np.column_stack((arr2['X'], arr2['Y'], arr2['Z']))
# 
#     # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1
#     for p1 in points1:
#         distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
#         min_distance = np.min(distances)
#         distance_1_to_2 += min_distance
# 
#     for p2 in points2:
#         distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
#         min_distance = np.min(distances)
#         distance_2_to_1 += min_distance
# 
#     # TODO: Return Chamfer distance, sum of the average distances in both directions
#     return (distance_1_to_2 / (len(arr1))) + (distance_2_to_1 / (len(arr2)))

# def rigid_transform(A, B):
#     """
#     Find the rigid (translation + rotation) transformation between two sets of points.
# 
#     Parameters:
#     - A: numpy array, mxn representing m points in an n-dimensional space.
#     - B: numpy array, mxn representing m points in an n-dimensional space.
# 
#     Returns:
#     - R: numpy array, n x n rotation matrix.
#     - t: numpy array, translation vector.
#     """
#     A = np.array(A)
#     B = np.array(B)
#     n, dim = A.shape
#     # TODO: Subtract centroids to center the point clouds A and B
#     centeredA = A - A.mean(axis=0)
#     centeredB = B - B.mean(axis=0)
# 
#     # TODO: Construct Cross-Covariance matrix
#     C = np.dot(np.transpose(centeredA), centeredB) / n
# 
#     # TODO: Apply SVD to the Cross-Covariance matrix
#     V, S, W = np.linalg.svd(C)
# 
#     # TODO: Calculate the rotation matrix
#     d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
# 
#     if d:
#         S[-1] = -S[-1]
#         V[:, -1] = -V[:, -1]
# 
#     R = np.dot(V, W)
# 
#     varP = np.var(A, axis=0).sum()
#     c = 1/varP * np.sum(S) # scale factor
# 
#     # TODO: Calculate the translation vector
#     t = A.mean(axis=0) - B.mean(axis=0).dot(c*R)
# 
#     # TODO: Return rotation and translation matrices
#     return R, t, c


# def icp(source, target, max_iterations=100, tolerance=1e-5):
#     """
#     Perform ICP (Iterative Closest Point) between two sets of points.
# 
#     Parameters:
#     - source: numpy array, mxn representing m source points in an n-dimensional space.
#     - target: numpy array, mxn representing m target points in an n-dimensional space.
#     - max_iterations: int, maximum number of iterations for ICP.
#     - tolerance: float, convergence threshold for ICP.
# 
#     Returns:
#     - R: numpy array, n x n rotation matrix.
#     - t: numpy array, translation vector.
#     - transformed_source: numpy array, mxn representing the transformed source points.
#     """
#     # TODO: Iterate until convergence
#     for _ in range(max_iterations):
#         # TODO: Find the nearest neighbors of target in the source
#         dist = np.einsum('ij,ij->i',source,source)[:,None] + np.einsum('ij,ij->i',target,target) - 2*np.dot(source,target.T)
#         k_indices = np.argpartition(dist, 2, axis=1)[:, :2]
#         nearest = [source[k_indices[i, :2]] for i in range(source.shape[0])]
#         nearest = np.array(nearest)
#         print("nearest.shape: ", nearest.shape)
#         # TODO: Calculate rigid transformation
#         R, t = rigid_transform(source,nearest)
#         # TODO: Apply transformation to source points
#         print("R.shape: ", R.shape)
#         print("sourse.shape: ", source.shape)
#         print("t.shape: ", t.shape)
#         trans_source = R @ source + t
#         # TODO: Calculate Chamfer distance
#         cham_dist = chamfer_distance(trans_source, target)
# 
#         # TODO: Check for convergence
#         if cham_dist < tolerance:
#             break
#     # TODO: Return the transformed source
#     return trans_source

def icp(source, target, max_iterations=100, tolerance=1e-5):
    transformed_source = source.copy()
    for iteration in range(max_iterations):
        # Find the nearest neighbors of target in the source
        distances = np.linalg.norm(target[:, np.newaxis, :] - transformed_source, axis=2)
        nearest_neighbors = np.argmin(distances, axis=1)
        # Extract corresponding points from the source and target
        corresponding_source = transformed_source[nearest_neighbors]
        corresponding_target = target
        # Calculate rigid transformation
        R, t = rigid_transform(corresponding_source, corresponding_target)
        # Apply transformation to source points
        transformed_source = np.dot(transformed_source, R.T) + t
        # Calculate Chamfer distance
        chamfer_dist = chamfer_distance(transformed_source, target)
        # Check for convergence
        if iteration > 0 and np.abs(chamfer_dist - prev_chamfer_dist) < tolerance:
            break
        prev_chamfer_dist = chamfer_dist
    return transformed_source

def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """
    num_clouds = len(point_clouds)
    affinity_matrix = np.zeros((num_clouds, num_clouds))
    for i in range(num_clouds):
        for j in range(i+1, num_clouds):
            cloud_i = point_clouds[i]
            cloud_j = point_clouds[j]
            # Register the point clouds with each other using ICP
            registered_cloud_i = icp(cloud_i, cloud_j)
            # Calculate symmetric Chamfer distance between registered clouds
            chamfer_dist_ij = chamfer_distance(registered_cloud_i, cloud_j)
            chamfer_dist_ji = chamfer_distance(cloud_j, registered_cloud_i)
            # Set affinity values in the matrix
            affinity_matrix[i, j] = chamfer_dist_ij
            affinity_matrix[j, i] = chamfer_dist_ji
    return affinity_matrix

# # def construct_affinity_matrix(point_clouds):
# #     """
# #     Construct the affinity matrix for spectral clustering based on the given data.
# # 
# #     Parameters:
# #     - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.
# # 
# #     Returns:
# #     - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
# #     """
# #     # TODO: Iterate over point clouds to fill affinity matrix
# #     m, n, d = point_clouds.shape
# #     affinity_matrix = np.zeros(shape=(m, m))
# #     for i, pc1 in enumerate(point_clouds):
# #         idx = 0
# #         min_cham_pc1 = -sys.maxsize - 1
# #         for j, pc2 in enumerate(point_clouds):
# #                 if (pc1 == pc2).all():
# #                     continue
# #                 # TODO: For each pair of point clouds, register them with each other
# #                 trans_source = icp(pc1, pc2)
# #                 chamdist = chamfer_distance(trans_source, pc2)
# #                 if chamdist < min_cham_pc1:
# #                     min_cham_pc1 = chamdist
# #                     idx = j
# #         affinity_matrix[i, idx] = min_cham_pc1
# #         affinity_matrix[idx, i] = min_cham_pc1
# #     # TODO: Calculate symmetric Chamfer distance between registered clouds
# #     return affinity_matrix

if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']  # feature points
    y = dataset['target']  # ground truth labels
    n = len(np.unique(y))  # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # Plot Ach using its first 3 eigenvectors
    _, eigenvectors = np.linalg.eigh(Ach)

    fig = plt.figure(figsize=(12, 6))

    # Plot the first three eigenvectors
    for i in range(3):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(eigenvectors[:, i], eigenvectors[:, i + 1], eigenvectors[:, i + 2], c=y_pred, cmap='viridis')
        ax.set_title(f'Eigenvectors {i + 1} to {i + 3}')

    # Plot the affinity matrix
    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(Ach, cmap='viridis')
    ax.set_title('Affinity Matrix')
    fig.colorbar(im)

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     dataset = "mnist"
#     fig, plots = plt.subplots(3,1,figsize=(14,10))
#     dataset = np.load("datasets/%s.npz" % dataset)
#     X = dataset['data']     # feature points
#     y = dataset['target']   # ground truth labels
#     n = len(np.unique(y))   # number of clusters
# 
#     Ach = construct_affinity_matrix(X)
#     y_pred = spectral_clustering(Ach, n)
# 
#     print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))
# 
#     # TODO: Plot Ach using its first 3 eigenvectors


