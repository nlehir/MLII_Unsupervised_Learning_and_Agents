"""
    compute the inertia
    of a dataset related to a given axis
"""
import numpy as np
import matplotlib.pyplot as plt


def orthogonal_projection(vector, axis):
    inner_product = np.dot(vector, axis)
    projected_vector = inner_product * axis
    return projected_vector


# load and center the data
data = np.load("data.npy")
x_data = data[:, 0]
y_data = data[:, 1]
x_mean = np.mean(x_data)
y_mean = np.mean(y_data)

# center the data
x_data = x_data - np.mean(x_data)
y_data = y_data - np.mean(y_data)
data = np.column_stack((x_data, y_data))

# plot the centered data
plt.plot(x_data, y_data, "o", color="olivedrab", markersize="3")
plt.title("centered data")
plt.savefig("images/centered data.pdf")


def test_axis(axis: np.ndarray, x_data: np.ndarray, y_data: np.ndarray):
    """
    Evaluates the inertia
    of the dataset related to an axis.
    we assume the data are centered.

    The axis is first encoded as a vector M=(u, v),
    such that the axis corresponds to
    the straight the line (OM).

    u must be nonzero.
    """
    # normalize the axis
    axis = 1 / np.linalg.norm(axis) * axis
    # check if vector is normed
    # print(np.linalg.norm(axis))

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

    # project each datapoint to the chosen axis
    # and compute the inertia due to this point.
    nb_datapoints = x_data.shape[0]
    inertia = 0
    for datapoint_index in range(nb_datapoints):
        vector = data[datapoint_index, :]
        projected_vector = orthogonal_projection(vector, axis)
        # check orthogonality
        # print(np.dot(axis, vector-projected_vector))

        # compute the inertia due to this sample
        difference = vector - projected_vector
        difference_norm = np.linalg.norm(difference)
        inertia += difference_norm**2

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


if __name__ == "__main__":
    # choose axes and compute the inertia
    axis = np.array([3, 1])
    test_axis(
        axis,
        x_data,
        y_data,
    )

    axis = np.array([1, -2])
    test_axis(
        axis,
        x_data,
        y_data,
    )

    axis = np.array([0.1, -2])
    test_axis(
        axis,
        x_data,
        y_data,
    )

    axis = np.array([-2.5, 8])
    test_axis(
        axis,
        x_data,
        y_data,
    )
