"""
    build an affinity matric and apply Spectral Clustering
"""

import numpy as np
from sklearn.cluster import SpectralClustering

adjacency_matrix = np.random.randint(0, 2, (10, 10))
adjacency_matrix = np.zeros((14, 14))

# Implement the adjacency matrix
# cluster 1
adjacency_matrix[0, [1, 2]] = 1
adjacency_matrix[1, [0, 3, 2, 5]] = 1
adjacency_matrix[2, [0, 1, 5, 4]] = 1
adjacency_matrix[3, [1, 4]] = 1
adjacency_matrix[4, [2, 3]] = 1
adjacency_matrix[5, [1, 2]] = 1

# cluster 2
adjacency_matrix[6, [7, 8]] = 1
adjacency_matrix[7, [6, 8]] = 1
adjacency_matrix[8, [7, 6]] = 1

# cluster 3
adjacency_matrix[9, [12, 13]] = 1
adjacency_matrix[10, [11, 12]] = 1
adjacency_matrix[11, [10, 12, 13]] = 1
adjacency_matrix[12, [9, 10, 11, 13]] = 1
adjacency_matrix[13, [9, 11, 12]] = 1

# check the symmetry of the adjecency matrix
transp = np.transpose(adjacency_matrix)
if np.array_equal(adjacency_matrix - transp, np.zeros((14, 14))):
    print("symmetric adjacency matrix")
else:
    print("warning! adjacency matrix not symmetric")
print(adjacency_matrix)

nb_datapoints = adjacency_matrix.shape[0]
dataset = [x for x in range(nb_datapoints)]

nb_clusters = 3

sc = SpectralClustering(nb_clusters, affinity="precomputed")

# apply the Spectral Clustering to the adjacency matrix
sc.fit_predict(adjacency_matrix)

# print the clusters
for cluster_index in range(nb_clusters):
    cluster = np.where(sc.labels_ == cluster_index)[0]
    print(f"cluster {cluster_index}")
    print(cluster)
