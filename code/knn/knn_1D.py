import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

u = 10
n_test = 50
sigma = 0.4
sigma = 1


def elementary_function(w, a, x):
    return np.sin(w * x) + a * x


def generate_function(u):
    """
    generate an example function
    """
    np.random.seed(13)
    params = list()
    for i in range(u):
        w_i = math.pi / 2 * np.random.randint(1, 50)
        a_i = 0.1 * np.random.rand()
        params.append((w_i, a_i))

    def oracle(params, x):
        result = 0
        for i in range(len(params)):
            w = params[i][0]
            a = params[i][1]
            result += elementary_function(w, a, x)
        return result

    return oracle, params


def generate_dataset(oracle, params, n_samples, sigma):
    x_data = np.random.uniform(0, 1, n_samples)
    y_data = oracle(params, x_data)
    noise = np.random.normal(0, sigma, n_samples)
    y_data += noise
    return x_data, y_data


def predict(
    x_data: np.ndarray, y_data: np.ndarray, x_test: np.ndarray, k: int
) -> np.ndarray:
    """
    predict by local averaging
    """
    # compute pairwise distances
    dist_x_data = cdist(x_test, x_data, "euclidean")
    # sort the distances
    sorted_indexes = np.argsort(dist_x_data, axis=1)
    # get the k nearest neighbors for each sample in x_test
    k_neighbors = sorted_indexes[:, :k]
    y_predictions = list()
    for i in range(len(x_test)):
        x_neighbors = k_neighbors[i]
        y_neighbors = y_data[x_neighbors]
        # average the predictions on the dataset
        y_prediction = np.mean(y_neighbors)
        y_predictions.append(y_prediction)
    return np.asarray(y_predictions).reshape(n_test, 1)


def knn(n_samples: int, k: int) -> None:
    """
    perform knn perdiction with
    n_samples in the dataset
    and
    k nearest neighbors
    """
    oracle, params = generate_function(u)

    # plot dataset
    alpha = 0.8 * min(1, max(0.1, 1.1 - math.log10(n_samples) / 4))
    x_data, y_data = generate_dataset(oracle, params, n_samples, sigma)
    if n_samples > 1e3:
        selection = range(int(n_samples / 30))
        alpha = 0.8 * min(1, max(0.1, 1.1 - math.log10(n_samples / 30) / 4))
        plt.plot(
            x_data[selection],
            y_data[selection],
            "o",
            label="data",
            alpha=alpha,
            markersize=3,
            color="slateblue",
        )
    else:
        plt.plot(
            x_data,
            y_data,
            "o",
            label="data",
            alpha=alpha,
            markersize=3,
            color="slateblue",
        )

    # show oracle
    x_oracle = np.linspace(0, 1, 200)
    y_oracle = oracle(params, x_oracle)
    plt.plot(x_oracle, y_oracle, label="Bayes estimator", color="aqua")
    plt.xlabel("x")
    plt.ylabel("y")

    # predict using knn
    x_test = np.random.rand(n_test).reshape(n_test, 1)
    y_predictions = predict(x_data.reshape(n_samples, 1), y_data, x_test, k)
    # plot
    plt.plot(x_test, y_predictions, "x", label="prediction", color="coral")

    # get the correct label for x_test
    y_truth = oracle(params, x_test)

    # error
    loss = np.linalg.norm(y_predictions - y_truth) / n_samples

    # save
    plt.legend(loc="best")
    title = (
        f"kNN regression\n"
        f"{k} neighbors, {n_samples} samples\n"
        r"$\sigma=$"
        f"{sigma:.1f}\n"
        f"test error={loss:.2E}"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"images_1D/knn_k={k}_samples={n_samples}_sigma={sigma}.pdf")
    plt.close()

    print(f"\n{k} neighbors, {n_samples} samples")
    print(f"loss {loss:.2E}")


def main():
    for k in [1, 10, 20]:
        for n_samples in [int(10**n) for n in [1, 2, 4, 5]]:
            knn(n_samples, k)


if __name__ == "__main__":
    main()
