"""
    Perform the k-means algorithm (unsupervised learning)
    using array operations
"""
import os
from joblib.externals.cloudpickle import dis

import matplotlib.pyplot as plt
import numpy as np


def main(nbs_of_iterations: int) -> None:
    """
    perform a k-means algorithm
    with 3 clusters on a simple 2D dataset.
    """

    # load the data
    datapath = os.path.join("data", "data.npy")
    data = np.load(datapath)
    x = data[:, 0]
    y = data[:, 1]

    nb_samples = len(x)

    # clean images
    clean("images")

    # we dont initialize the centroids completely randomly
    x_min = min(x)
    x_max = max(x)

    # initialize the centroid positions
    centroids = np.random.uniform(x_min, x_max, size=(3, 2))
    x_centroids = centroids[:, 0]
    y_centroids = centroids[:, 1]
    print("initial centroid positions")

    for iteration in range(nbs_of_iterations):
        print(f"\niteration: {iteration}")

        """
        add lines here
        """

        plot_clustering(
            iteration,
            x,
            y,
            cluster_0,
            cluster_1,
            cluster_2,
            x_centroids,
            y_centroids,
            "assign_samples_to_centroid",
        )

        # Update centroids positions
        """
        add lines here
        """
        print(f"x0: {x_centroids[0]:.2f}  y0: {y_centroids[0]:.2f}")
        print(f"x1: {x_centroids[1]:.2f}  y1: {y_centroids[1]:.2f}")
        print(f"x2: {x_centroids[2]:.2f}  y2: {y_centroids[2]:.2f}")

        plot_clustering(
            iteration,
            x,
            y,
            cluster_0,
            cluster_1,
            cluster_2,
            x_centroids,
            y_centroids,
            "move_centroids",
        )


def plot_clustering(
    iteration: int,
    x: np.ndarray,
    y: np.ndarray,
    cluster_0: np.ndarray,
    cluster_1: np.ndarray,
    cluster_2: np.ndarray,
    x_centroids,
    y_centroids,
    step: str,
) -> None:
    """
    Plot the current state of the clustering

    """
    plt.plot(
        x[cluster_0], y[cluster_0], "o", color="darkorange", markersize=3, alpha=0.8
    )
    plt.plot(
        x[cluster_1], y[cluster_1], "o", color="firebrick", markersize=3, alpha=0.8
    )
    plt.plot(
        x[cluster_2], y[cluster_2], "o", color="cornflowerblue", markersize=3, alpha=0.8
    )
    plt.plot(x_centroids, y_centroids, "o", color="lime")
    title = f"update centroids : iteration {iteration} (centroids in green)"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    figpath = os.path.join("images", f"it_{iteration}_{step}.pdf")
    plt.savefig(figpath)
    plt.close("all")


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


if __name__ == "__main__":
    nbs_of_iterations = 10
    main(nbs_of_iterations)
