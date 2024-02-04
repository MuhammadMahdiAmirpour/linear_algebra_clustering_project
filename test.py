import numpy as np
# Assuming 'data' is your matrix with shape (m, n)
# For example, m=3, n=4
data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

# Calculate pairwise distances using broadcasting
differences = data[:, :, np.newaxis] - data[:, np.newaxis, :]
pairwise_distances = np.sqrt(np.sum(differences**2, axis=0))

print("Pairwise Distance Matrix:")
print(pairwise_distances)


