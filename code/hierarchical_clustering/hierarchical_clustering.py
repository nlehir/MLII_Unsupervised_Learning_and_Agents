"""
    Exercise illustrating hierarchical clustering.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from sklearn.cluster import linkage_tree

# open file
file_name = "addresses.csv"

x_coordinates = list()
y_coordinates = list()

# load the data
with open(file_name, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        x_coordinates.append(float(row[0]))
        y_coordinates.append(float(row[1]))

x_coordinates = np.asarray(x_coordinates)
y_coordinates = np.asarray(y_coordinates)
data = np.column_stack((x_coordinates, y_coordinates))

"""
    Part A : show a scatter plot of the data
"""
# # first method manual plot
nb_datapoints = x_coordinates.shape[0]
for datapoint in range(nb_datapoints):
    plt.plot(x_coordinates[datapoint], y_coordinates[datapoint], "o")
plt.title("scatter plot of addresses")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
fig_path = os.path.join("images", "scatter_plot_manual.pdf")
plt.savefig(fig_path)
plt.close()

# second method
plt.plot(x_coordinates, y_coordinates, "o")
plt.title("scatter plot of addresses")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
fig_path = os.path.join("images", "scatter_plot_manual_2.pdf")
plt.savefig(fig_path)
plt.close()

# third method : matplotlib function
plt.scatter(x_coordinates, y_coordinates)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
fig_path = os.path.join("images", "scatter_plot_matplotlib.pdf")
plt.savefig(fig_path)
plt.close()


"""
    Part B : hierarchical clustering.
    We will use agglomerative clustering.
"""
# choose metric
metric = "euclidean"

# build distance matrix between points
condensed_distance = scipy.spatial.distance.pdist(data, metric=metric)
distance_matrix = scipy.spatial.distance.squareform(condensed_distance)

# initializes classes
# each data point is its own class.
classes = [[i] for i in range(nb_datapoints)]


def find_closest_classes(classes: list[list], linkage: str = "single") -> tuple:
    """
    In the list of classes, find the two closest classes

    :param classes (list): list of clsses
    :returns returned_classes (tuple): two closest classes
    """
    min_dist = np.max(distance_matrix)
    returned_classes = (0, 1)
    """
    add lines here
    """
    return returned_classes


def distance_between_classes_single_linkage(class_1: list, class_2: list) -> float:
    """
    single linkage clustering.
    The distance is the minimum distance between
    any point in class_1 and any point in class_2

    :param class_1 (list): list of houses
    :param class_2 (list): list of houses
    :returns min_dist: minimum disance between points between the two
    classes
    """
    min_dist = np.max(distance_matrix)
    """
    add lines here
    """
    return min_dist


def distance_between_classes_average_linkage(class_1: list, class_2: list) -> float:
    """
    average linkage clustering
    The distance is the average of distances between
    apoint in class_1 a point in class_2

    :param class_1 (list): list of houses
    :param class_2 (list): list of houses
    :returns average_distances: minimum average between points between the two
    classes
    """
    min_dist = np.max(distance_matrix)
    average_distances = min_dist
    """
    add lines here
    """
    return average_distances


def plot_clustering(step: int, classes: list[list], linkage: str) -> None:
    """
    Plots the clustering resutls
    """
    plt.plot(x_coordinates, y_coordinates, "o")
    for index in range(len(classes)):
        class_to_plot = classes[index]
        x_coord = x_coordinates[class_to_plot]
        y_coord = y_coordinates[class_to_plot]
        plt.plot(x_coord, y_coord, "o")
    plt.title(
        "hierarchical clustering"
        f"\nstep {step}"
        f"\n{len(classes)} clusters"
        f"\n{metric} metric"
        f"\n{linkage} linkage"
    )
    plt.tight_layout()
    fig_name = f"{linkage}_linkage_step_{step}.pdf"
    fig_path = os.path.join("clusterings", fig_name)
    plt.savefig(fig_path)
    plt.close()


def main():
    step = 0
    LINKAGE = "single"
    print("hierarchical clustering")
    print(f"use {LINKAGE} linkage")
    while len(classes) > 1:
        print(f"step: {step}")
        step += 1
        index_1, index_2 = find_closest_classes(classes, linkage=LINKAGE)
        classes[index_1] = classes[index_1] + classes[index_2]
        classes.remove(classes[index_2])
        plot_clustering(step, classes, linkage=LINKAGE)


if __name__ == "__main__":
    main()
