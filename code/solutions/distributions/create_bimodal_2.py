import numpy as np
import matplotlib.pyplot as plt

mean_1 = 4
std_dev_1 = 1
nb_point_1 = 1000

mean_2 = 15
std_dev_2 = 3
nb_point_2 = 1000

nb_point = nb_point_1 + nb_point_2

mode_1 = np.random.normal(loc=mean_1, scale=std_dev_1, size=(nb_point_1, ))
mode_2 = np.random.normal(loc=mean_2, scale=std_dev_2, size=(nb_point_1, ))

array = np.concatenate((mode_1, mode_2))
plt.hist(array, bins=500)
plt.savefig("hist.pdf")
