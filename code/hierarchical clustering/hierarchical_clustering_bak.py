"""
    Exercise illustrating hierarchical clustering.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

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
plt.savefig("scatter_plot_manual.pdf")
plt.close()

# second method
plt.plot(x_coordinates, y_coordinates, "o")
plt.title("scatter plot of addresses")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.savefig("scatter_plot_manual_2.pdf")
plt.close()

# third method : matplotlib function
plt.scatter(x_coordinates, y_coordinates)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.savefig("scatter_plot_matplotlib.pdf")
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


def find_closest_classes(classes: list[list]) -> tuple:
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
    """
    add lines here
    """
    return average_distances


def define_color(index):
    int_1 = (3 * index) % 9 + 1
    int_2 = (3 * index) % 5 + 1
    int_3 = (3 * index) % 2 + 1


def plot_clustering(step: int, classes: list[list]) -> None:
    """
    Plots the clustering resutls
    """
    plt.plot(x_coordinates, y_coordinates, "o")
    for index in range(len(classes)):
        class_to_plot = classes[index]
        x_coord = x_coordinates[class_to_plot]
        y_coord = y_coordinates[class_to_plot]
        color = define_color(index)
        plt.plot(x_coord, y_coord, "o", color=color)
    plt.title(
        f"hierarchical clustering\nstep {step}"
        f"\n{len(classes)} clusters\n"
        f"{metric} metric"
    )
    plt.tight_layout()
    plt.savefig(f"clustering/clustering_step_{step}.pdf")
    plt.close()


def main():
    """
    main loop
    """
    step = 0
    while len(classes) > 1:
        print(f"step: {step}")
        step += 1
        index_1, index_2 = find_closest_classes(classes)
        classes[index_1] = classes[index_1] + classes[index_2]
        classes.remove(classes[index_2])
        plot_clustering(step, classes)


if __name__ == "__main__":
    main()
