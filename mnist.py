import numpy as np
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

    # TODO: Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1

    # TODO: Return Chamfer distance, sum of the average distances in both directions

    pass

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

    # TODO: Construct Cross-Covariance matrix

    # TODO: Apply SVD to the Cross-Covariance matrix

    # TODO: Calculate the rotation matrix

    # TODO: Calculate the translation vector

    # TODO: Return rotation and translation matrices

    pass

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

    # TODO: Find the nearest neighbors of target in the source

    # TODO: Calculate rigid transformation

    # TODO: Apply transformation to source points

    # TODO: Calculate Chamfer distance

    # TODO: Check for convergence

    # TODO: Return the transformed source

    pass

def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """

    # TODO: Iterate over point clouds to fill affinity matrix

    # TODO: For each pair of point clouds, register them with each other

    # TODO: Calculate symmetric Chamfer distance between registered clouds

    pass


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
