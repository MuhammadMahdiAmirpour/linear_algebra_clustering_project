import numpy as np
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score

def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer distance between two point clouds.

    Parameters:
    - point_cloud1: numpy array, shape (N1, D), representing the first point cloud.
    - point_cloud2: numpy array, shape (N2, D), representing the second point cloud.

    Returns:
    - dist: float, the Chamfer distance between the two point clouds.
    """
    
    # TODO: Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2
    distance_1_to_2 = 0
    distance_2_to_1 = 0
    points1 = np.column_stack((arr1['X'], arr1['Y'], arr2['Z']))
    points2 = np.column_stack((arr2['X'], arr2['Y'], arr2['Z']))

    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1
    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    # TODO: Return Chamfer distance, sum of the average distances in both directions
    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(arr1) + len(arr2))

def rigid_transform(A, B):
    """
    Find the rigid (translation + rotation) transformation between two sets of points.

    Parameters:
    - A: numpy array, mxn representing m points in an n-dimensional space.
    - B: numpy array, mxn representing m points in an n-dimensional space.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    """
    # TODO: Subtract centroids to center the point clouds A and B
    A = np.array(A)
    B = np.array(B)

    n, dim = A.shape

    centeredA = A - A.mean(axis=0)
    centeredB = B - B.mean(axis=0)

    # TODO: Construct Cross-Covariance matrix
    C = np.dot(np.transpose(centeredA), centeredB) / n

    # TODO: Apply SVD to the Cross-Covariance matrix
    V, S, W = np.linalg.svd(C)
    # TODO: Calculate the rotation matrix
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # TODO: Calculate the translation vector
    R = np.dot(V, W)

    varP = np.var(a1, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    # TODO: Return rotation and translation matrices
    return R, t, c


def icp(source, target, max_iterations=100, tolerance=1e-5):
    """
    Perform ICP (Iterative Closest Point) between two sets of points.

    Parameters:
    - source: numpy array, mxn representing m source points in an n-dimensional space.
    - target: numpy array, mxn representing m target points in an n-dimensional space.
    - max_iterations: int, maximum number of iterations for ICP.
    - tolerance: float, convergence threshold for ICP.

    Returns:
    - R: numpy array, n x n rotation matrix.
    - t: numpy array, translation vector.
    - transformed_source: numpy array, mxn representing the transformed source points.
    """
    # TODO: Iterate until convergence
    for _ in range(max_iterations):
        # TODO: Find the nearest neighbors of target in the source
        dist = np.einsum('ij,ij->i',source,source)[:,None] + np.einsum('ij,ij->i',target,target) - 2*np.dot(source,target.T)
        k_indices = np.argpartition(dist, 2, axis=1)[:, :2]
        nearest = [source[k_indices[i, :2]] for i in range(source.shape[0])]
        # TODO: Calculate rigid transformation
        R, t , _ = rigid_transform(source,nearest)
        # TODO: Apply transformation to source points
        trans_source = R @ source + t
        # TODO: Calculate Chamfer distance
        cham_dist = chamfer_distance(trans_source, target)

        # TODO: Check for convergence
        if cham_dist < tolerance:
            break
    # TODO: Return the transformed source
    return trans_source


def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """

    # TODO: Iterate over point clouds to fill affinity matrix
    m, n, d = point_clouds.shape
    affinity_matrix = np.zeros(shape=(m, n))
    for i, pc1 in enumerate(point_clouds):
        idx = 0
        min_cham_pc1 = -sys.maxsize - 1
        for j, pc2 in enumerate(point_clouds):
                if (pc1 == pc2).all():
                    continue
                # TODO: For each pair of point clouds, register them with each other
                trans_source = icp(pc1, pc2)
                chamdist = chamfer_distance(trans_source, pc2)
                if chamdist < min_cham_pc1:
                    min_cham_dist = chamdist
                    idx = j
        affinity_matrix[i, idx] = min_cham_dist
        affinity_matrix[idx, i] = min_cham_dist

    # TODO: Calculate symmetric Chamfer distance between registered clouds
    return affinity_matrix



if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']     # feature points
    y = dataset['target']   # ground truth labels
    n = len(np.unique(y))   # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Plot Ach using its first 3 eigenvectors
