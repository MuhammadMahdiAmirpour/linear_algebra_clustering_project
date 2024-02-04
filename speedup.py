from spectral import spectral_clustering as spectral_clustering_old
from mnist import construct_affinity_matrix as construct_affinity_matrix_old

from metrics import clustering_score

from numba import jit, njit, prange, vectorize, guvectorize, cuda

import numpy as np
from numpy import linalg as LA
from timeit import timeit

# TODO: Rewrite the k_means_clustering function

# TODO: Rewrite the laplacian function

# TODO: Rewrite the spectral_clustering function

# TODO: Rewrite the chamfer_distance function

# TODO: Rewrite the rigid_transform function

# TODO: Rewrite the icp function

# TODO: Rewrite the construct_affinity_matrix function


if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']     # feature points
    y = dataset['target']   # ground truth labels
    n = len(np.unique(y))   # number of clusters

    # TODO: Run both the old and speed up version of your algorithms and capture running time

    print("Old Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred_old))
    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # TODO: Compare the running time using timeit module

    pass
