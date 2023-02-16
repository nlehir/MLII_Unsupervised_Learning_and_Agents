"""
Create data to study overfitting
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
from oracle import oracle, sigma

file_name = "noisy_data.csv"

inputs = np.random.uniform(0, 1, 100)

# create data with random noise
outputs = oracle(inputs) + np.random.normal(0, sigma, inputs.shape)


with open(file_name, "w") as csvfile:
    filewriter = csv.writer(csvfile, delimiter=",")
    for point in range(len(inputs)):
        filewriter.writerow([inputs[point], outputs[point]])


title = "Noisy data"
plt.plot(inputs, outputs, "o", markersize=3)
plt.xlabel("input")
plt.ylabel("output")
plt.title(title)
plt.savefig("data_to_fit.pdf")
