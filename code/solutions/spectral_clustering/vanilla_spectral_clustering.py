"""
    Build an affinity matric and apply Spectral Clustering
"""

from sklearn.cluster import SpectralClustering
import numpy as np


def main() -> None:
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

    transp = np.transpose(adjacency_matrix)
    print(np.where(adjacency_matrix-transp))
    print(adjacency_matrix)

    # choose a relevant number of clusters
    n_clusters = 3
    sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            )

    # apply the Spectral Clustering to the adjacency matrix
    sc.fit_predict(adjacency_matrix)

    # print the clusters
    for cluster_index in range(n_clusters):
        cluster = np.where(sc.labels_ == cluster_index)[0]
        print(f"cluster {cluster_index}")
        print(cluster)

if __name__ == "__main__":
    main()
