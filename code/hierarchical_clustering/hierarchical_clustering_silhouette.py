"""
    Exercise illustrating hierarchical clustering.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from hierarchical_clustering import find_closest_classes
from sklearn import metrics

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

metric = "euclidean"

# build distance matrix between points
condensed_distance = scipy.spatial.distance.pdist(data, metric=metric)
distance_matrix = scipy.spatial.distance.squareform(condensed_distance)

# initializes classes
# each data point is its own class.
nb_datapoints = x_coordinates.shape[0]
classes = [[i] for i in range(nb_datapoints)]


def get_label(sample: int, classes: list[list]) -> int:
    for (class_id, class_content) in enumerate(classes):
        if sample in class_content:
            return class_id


def main():
    """
    main loop
    """
    silhouette_scores = list()
    nb_clusters = list()
    step = 0
    while len(classes) > 2:
        print(f"step: {step}")
        step += 1
        index_1, index_2 = find_closest_classes(classes)
        classes[index_1] = classes[index_1] + classes[index_2]
        classes.remove(classes[index_2])
        labels = [get_label(i, classes) for i in range(nb_datapoints)]


if __name__ == "__main__":
    main()
