"""
Read data from a csv
"""

import csv

import matplotlib.pyplot as plt

# open file
file_name = "multimodal_distribution.csv"

index = list()
data = list()

with open(file_name, "r") as f:
    reader = csv.reader(f)
    row_index = 0
    for row in reader:
        if row_index >= 1:
            index.append(int(row[0]))
            data.append(float(row[1]))
        row_index = row_index + 1

"""
plot the raw data
"""
plt.plot(index, data)
title = "data 1"
plt.title("raw data")
plt.xlabel("index")
plt.ylabel("value")
plt.savefig("images/multimodal_distribution.pdf")
plt.close()

"""
plot a histogram
"""
# use a relevant number of bins !
nbins = 100
plt.hist(data, bins=nbins)
plt.xlabel("value")
plt.ylabel("nb of occurrences")
plt.title("multimodal distribution")
plt.savefig(f"images/multimodal_distribution_{nbins}_bins.pdf")
plt.close()
