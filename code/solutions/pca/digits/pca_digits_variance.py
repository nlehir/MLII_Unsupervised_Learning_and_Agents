from sklearn.datasets import load_digits
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main() -> None:
    digits = load_digits()

    # the data consists in 1797 samples
    # of 8*8 pixels images

    pca = PCA().fit(digits.data)

    variance_ratio = pca.explained_variance_ratio_

    cumulated_variance = np.cumsum(variance_ratio)

    plt.plot(cumulated_variance, "o")
    plt.xlabel("number of components")
    plt.ylabel("explained variance")
    fig_name = "explained_variance_ratio.pdf"
    fig_path = os.path.join("images", fig_name)
    plt.savefig(fig_path)

if __name__ == "__main__":
    main()
