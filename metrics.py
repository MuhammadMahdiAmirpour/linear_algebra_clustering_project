import numpy as np
import math


def create_contingency_table(X,Y):

    unique_labels = np.unique(X)
    XS = []
    XS = [np.where(X == label)[0].tolist() for label in unique_labels]
    XS_vectorized = np.where(X[:, None] == unique_labels[None, :])[1].tolist()
    unique_labels = np.unique(Y)
    YS = [np.where(Y == label)[0].tolist() for label in unique_labels]
    contingency_table = []
    for i in range(len(XS)):
        Xi = []
        for j in range(len(YS)):
            Xi.append(np.intersect1d(XS[i],YS[j]).size)
        contingency_table.append(Xi)
    return contingency_table

def clustering_score(true_labels, predicted_labels):
    """
    Calculate the clustering score to assess the accuracy of predicted labels compared to true labels.

    Parameters:
    - true_labels: List or numpy array, true cluster labels for each data point.
    - predicted_labels: List or numpy array, predicted cluster labels for each data point.

    Returns:
    - score: float, clustering score indicating the accuracy of predicted labels.
    """

    contingency_table = create_contingency_table(true_labels,predicted_labels)
    A = np.sum(contingency_table,axis= 0)
    B = np.sum(contingency_table,axis= 1)
    r = np.sum(np.vectorize(math.comb)(contingency_table, 2))
    combinations = np.frompyfunc(math.comb, 2, 1)
    u = np.sum(combinations(A, 2))
    v = np.sum(np.frompyfunc(math.comb, 2, 1)(B, 2))
    e = u*v/math.comb(len(true_labels),2)
    m = (u + v)/2
    return (r - e)/(m - e)

if __name__ == "__main__":
    # Example usage:
    true_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    predicted_labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    print(clustering_score(true_labels, predicted_labels))



