import numpy as np
import math
import matplotlib.pyplot as plt

angle_division = 100
number_of_rounds = 5
N_max = angle_division*number_of_rounds

n = np.arange(0, N_max)
theta = n*2*math.pi/angle_division
radius = np.exp(-n/angle_division)


def polar(radius, theta):
    return radius*np.cos(theta), radius*np.sin(theta)


positions = polar(radius, theta)

plt.plot(positions[0], positions[1])
plt.title("plongement droite réelle")
plt.xlim([-1, 1.2])
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.ylim([-1, 1])
plt.savefig("plongement.pdf")
