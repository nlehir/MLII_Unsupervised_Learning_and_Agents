from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

digits = load_digits()

# the data consists in 1797 samples
# of 8*8 pixels images

pca = PCA().fit(digits.data)

variance_ratio = pca.explained_variance_ratio_
