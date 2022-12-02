"""
Using the KL divergenve, choose the best model to fit the data
"""

import csv
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np

# open file
file_name = 'empirical_distribution.csv'

index = list()
data = list()

# load the csv file to python lists
with open(file_name, 'r') as f:
    reader = csv.reader(f)
    row_index = 0
    for row in reader:
        if row_index >= 1:
            index.append(int(row[0]))
            data.append(float(row[1]))
        row_index = row_index + 1


nb_points = len(data)

"""
plot the data
"""
# plt.plot(index, data)
# title = 'Empirical Distribution'
# plt.xlabel('index')
# plt.ylabel('age (years)')
# plt.savefig('images/empirical_distribution.pdf')
# plt.close()


"""
build a histogram of the empirical data
"""
nbins = 100
empirical_data, bins, _ = plt.hist(data, bins=nbins)
title = f"empirical distribution : histogram, {nbins} bins"
plt.title(title)
plt.xlabel('age (years)')
plt.ylabel('number of occurrences')
plt.savefig(f"images/empirical_hist_{nbins}_bins.pdf")
plt.close()


"""
try models and compare them to the empirical distribution
"""


def try_model(mean_model, std_model, empirical_data, bins):
    """
    we will use normal models in this example
    """
    # sample new data from our model
    # we will compare them to the empirical set
    model_sample = np.random.normal(mean_model, std_model, nb_points)

    # Compute the histogram of data sampled from your model
    # we must use the same bins as the empirical distribution
    # in order to compare the distributions
    model_hist, _, _ = plt.hist(model_sample, bins=bins, alpha=0.4, label="model")

    # plot the histogram of the empirical data
    # to visually compare the two sample sets (empirical vs model)
    empirical_data_hist, bins, _ = plt.hist(data, bins=nbins, alpha=0.4, label="data")

    # we will compute the KL divergence, but before we will
    # cheat and add artificial samples to each value.
    # if we don't do so, the divergence might be infinite
    # because some probabilities can take a value of 0
    model_hist = model_hist + 1
    empirical_data_hist = empirical_data_hist + 1

    # compute the kl divergence between the two distributions
    # the entropy function of scipy computes the KL
    # divergence if two distributions are provided
    # (see the scipy doc)
    kl_divergence = entropy(model_hist, empirical_data_hist)

    # print information
    # print('\n---')
    # print(kl_divergence)
    # print(model_hist)
    # print(empirical_data_hist)

    # annotate the plot with the KL divergence
    # we don't need all the decimals here so we can
    # use round to keep only two decimals
    plt.annotate(f"KL={kl_divergence:.2f}",
                 (50, 50),
                 fontsize=(14))
    title = f"model mean : {mean_model}, model std : {std_model}"
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel('age (years)')
    plt.ylabel('nb of occurrences')
    plt.savefig(f"images/model_hist_mean_{mean_model}_std_{std_model}.pdf")
    plt.close()


try_model(35, 5, empirical_data, bins)
try_model(35, 3, empirical_data, bins)
try_model(30, 2, empirical_data, bins)
try_model(32, 4, empirical_data, bins)
