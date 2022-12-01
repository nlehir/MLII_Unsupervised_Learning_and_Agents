"""
    Generate data for the normalized cut heuristic
"""

import matplotlib.pyplot as plt
import numpy as np

std_1 = 1 * 0.1
std_2 = 1 * 0.2
std_3 = 1 * 1

cluster_1 = np.random.normal((1, 4), std_1, (10, 2))
cluster_2 = np.random.normal((2, 2), std_2, (20, 2))
cluster_3 = np.random.normal((-2, -2), std_3, (30, 2))

data = np.concatenate((cluster_1, cluster_2))
data = np.concatenate((data, cluster_3))
np.random.shuffle(data)

nb_points = data.shape[0]
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, "o")
plt.title("data to cluster")
plt.savefig("images/data_to_cluster.pdf")
plt.close()

# build a matrix of distances
distances = np.zeros((nb_points, nb_points))
for i in range(nb_points):
    for j in range(nb_points):
        point_i = data[i, :]
        point_j = data[j, :]
        distances[i, j] = np.linalg.norm(point_i - point_j)


# we use the standard deviation to compute the similarity
similarity = np.exp(-distances / distances.std())
print(f"distances standard deviation: {distances.std()}")
print(similarity)
plt.imshow(similarity)
plt.savefig("images/similarity.pdf")
plt.close()


threshold = distances.std()
selection = np.where(similarity > 0.5)
adjacency_matrix = np.zeros(distances.shape)
adjacency_matrix[selection] = 1
plt.imshow(adjacency_matrix)
plt.savefig("images/adjacency_matrix.pdf")
plt.close()

# save the data
np.save("data/similarity", similarity)
np.save("data/adjacency_matrix", adjacency_matrix)
np.save("data/distances", distances)
