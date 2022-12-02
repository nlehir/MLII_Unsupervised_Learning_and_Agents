"""
    adapted from
    https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

    # TODO: improve <01-12-22, nlehir> #
"""
import os
from sklearn.datasets import load_digits
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def plot_pca_components(
    x: np.ndarray,
    coefficients=None,
    mean=0,
    components=None,
    imshape=(8, 8),
    n_components=8,
    fontsize=12,
    show_mean=True,
) -> None:
    if coefficients is None:
        coefficients = x

    if components is None:
        components = np.eye(len(coefficients), len(x))

    mean = np.zeros_like(x) + mean

    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(2, 4 + bool(show_mean) + n_components, hspace=0.3)

    def show(i, j, x, title=None):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation="nearest")
        if title:
            ax.set_title(title, fontsize=fontsize)

    show(slice(2), slice(2), x, "Sample")

    approx = mean.copy()

    counter = 2
    if show_mean:
        show(0, 2, np.zeros_like(x) + mean, r"$\mu$")
        show(1, 2, approx, r"$1 \cdot \mu$")
        counter += 1

    for i in range(n_components):
        approx = approx + coefficients[i] * components[i]
        show(0, i + counter, components[i], r"$c_{0}$".format(i + 1))
        show(
            1,
            i + counter,
            approx,
            r"${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1),
        )
        if show_mean or i > 0:
            plt.gca().text(
                0,
                1.05,
                "$+$",
                ha="right",
                va="bottom",
                transform=plt.gca().transAxes,
                fontsize=fontsize,
            )

    show(slice(2), slice(-2, None), approx, "Reconstruction")
    return fig


if __name__ == "__main__":
    # load data
    digits = load_digits()
    # perform PCA
    # and project the data
    # on the principal components
    nb_components = 8
    pca = PCA(n_components=nb_components)
    Xproj = pca.fit_transform(digits.data)
    sns.set_style("white")
    # plot the projected components and the reconstruction
    data_index = 12
    fig = plot_pca_components(
        digits.data[data_index],
        Xproj[data_index],
        pca.mean_,
        pca.components_,
        n_components=nb_components,
    )
    figpath = os.path.join("images", f"reconstruction_{data_index}_{nb_components}.pdf")
    fig.savefig(figpath)
