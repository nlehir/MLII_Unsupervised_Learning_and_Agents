"""
    compute the inertia
    of a dataset related to a given axis
"""
import numpy as np
import matplotlib.pyplot as plt
from orthogonal_projection import orthogonal_projection


def test_axis(axis: np.ndarray, data: np.ndarray):
    """
    Evaluates the inertia
    of the dataset related to an axis.
    we assume the data are centered.

    The axis is first encoded as a vector M=(u, v),
    such that the axis corresponds to
    the straight the line (OM).

    u must be nonzero.
    """
    x_data = data[:, 0]
    y_data = data[:, 1]
    # normalize the axis
    axis = 1 / np.linalg.norm(axis) * axis

    # plot the data
    plt.plot(x_data, y_data, "o", color="olivedrab", markersize="3")

    # plot the axis we want to project on
    coefficient_directeur = axis[1] / axis[0]
    x_axis = np.linspace(-8, 8, 100)
    plt.plot(
        x_axis,
        coefficient_directeur * x_axis,
        alpha=0.5,
        color="darkblue",
        label="axis",
    )

    nb_datapoints = len(data)

    """
    project each datapoint to the chosen axis
    and compute the inertia due to this point.
    """

    """
    vectorized method
    add lines here
    """

    """
    iterative method
    (to complete)
    """
    inertia = 0
    for datapoint_index in range(nb_datapoints):
        vector = data[datapoint_index, :]
        projected_vector = orthogonal_projection(vector, axis)
        # compute the inertia due to this sample
        """
        add lines here
        """


        # plot the projection
        plt.plot(
            [vector[0], projected_vector[0]],
            [vector[1], projected_vector[1]],
            color="mediumturquoise",
            alpha=0.5,
            label="projected vector",
        )

    inertia /= nb_datapoints


    plt.title(f"axis=({axis[0]:.2f}, {axis[1]:.2f}) \ninertia = {inertia:.2f}")
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(f"images/projection axis=({axis[0]:.2f}, {axis[1]:.2f}).pdf")
    plt.close()


def main() -> None:
    # load and center the data
    data = np.load("data.npy")
    data = data - data.mean(axis=0)
    plt.plot(data[0], data[1], "o", color="olivedrab", markersize="3")
    plt.title("centered data")
    plt.savefig("images/centered data.pdf")

    # choose axes and compute the inertia
    axes = [
            np.array([3, 1]),
            np.array([1, -2]),
            np.array([0.1, -2]),
            np.array([-2.5, 8]),
            ]
    for axis in axes:
        test_axis(
            axis=axis,
            data=data,
        )

if __name__ == "__main__":
    main()
