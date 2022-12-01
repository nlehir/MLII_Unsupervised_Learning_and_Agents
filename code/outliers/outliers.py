"""
    Z score
"""

import matplotlib.pyplot as plt
import numpy as np

data = np.load("data.npy")

plt.plot(data[:, 0], data[:, 1], "o", markersize=1)
plt.savefig("scatter plot.pdf")
nb_points = data.shape[0]

"""
    Compute center
"""
center = np.mean(data, axis=0)

"""
    Compute mean distance to center with euclidean metric
"""
metric = "euclidean"
differences = data - center
distances_to_center = np.linalg.norm(differences, axis=1)
mean_distance_to_center = distances_to_center.sum() / len(distances_to_center)

nb_outliers = 0
for index in range(nb_points):
    datapoint = data[index]
    vector_to_center = datapoint - center
    distance_to_center = np.linalg.norm(vector_to_center)
    if distance_to_center > 3 * mean_distance_to_center:
        print(datapoint)
        print(distance_to_center / mean_distance_to_center)
        print("")
        plt.plot(datapoint[0], datapoint[1], "o", color="red", markersize=1.5)
        nb_outliers += 1
print(f"{nb_outliers} outliers")


plt.title(f"outliers in red, {metric} metric")
plt.savefig(f"outliers, metric={metric}.pdf")
plt.close()


plt.plot(data[:, 0], data[:, 1], "o", markersize=1)
"""
    Compute mean distance to center with a more adapted metric
"""
metric = "custom"
std_x = np.std(data[:, 0])
std_y = np.std(data[:, 1])
# factor = 5
differences = data - center
differences[:, 0] /= std_x
differences[:, 1] /= std_y
abs_differences = np.abs(differences)
distances = np.sum(abs_differences, axis=1)
sum_distances = np.sum(distances)
mean_distance_to_center = sum_distances / nb_points

nb_outliers = 0
for index in range(nb_points):
    datapoint = data[index]
    vector_to_center = datapoint - center
    distance_to_center = abs(vector_to_center[0] / std_x) + abs(
        vector_to_center[1] / std_y
    )
    if distance_to_center > 2.5 * mean_distance_to_center:
        print(datapoint)
        print(distance_to_center / mean_distance_to_center)
        print("")
        plt.plot(datapoint[0], datapoint[1], "o", color="red", markersize=1.5)
        nb_outliers += 1
print(f"{nb_outliers} outliers")


plt.title(f"outliers in red, {metric} metric")
plt.savefig(f"outliers, metric={metric}.pdf")
plt.close()
