from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

redudant_data = np.load("redundant_data.npy")

pca = PCA().fit(redudant_data)

variance_ratio = pca.explained_variance_ratio_
