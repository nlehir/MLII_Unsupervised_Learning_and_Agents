import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


n_samples = 100
mu = 5
std = 10
data = np.random.normal(loc=mu, scale=std, size=(n_samples,))
nbs_samples = range(1, 50)
for nb_samples in nbs_samples:
    subset = data[:nb_samples]
    sns.kdeplot(data=subset)
    title=f"{nb_samples} samples"
    # plt.legend(loc="best")
    plt.title(title)
    plt.savefig(f"kde_{nb_samples}_samples.pdf")
