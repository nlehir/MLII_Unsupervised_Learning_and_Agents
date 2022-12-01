"""
Perform gradient descent on a simple function
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def function_to_minimize(x: float, y: float) -> float:
    """
    simple convex function that is
    to be minimized by gradient descent
    """
    return x**3 + x**4 + y**4 + 4 * y**2 + 5 - x


def xgradient(x: float, y: float) -> float:
    """
    compute the x coordinate of the gradient
    """
    return 3 * x**2 + 4 * x**3 - 1


def ygradient(x: float, y: float) -> float:
    """
    compute the y coordinate of the gradient
    """
    return 4 * y**3 + 8 * y


for image in os.listdir("function_1/"):
    os.remove(os.path.join("function_1", image))

# plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X = np.linspace(-20, 20)
Y = np.linspace(-20, 20)
X, Y = np.meshgrid(X, Y)
S = X + Y

for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        S[i][j] = function_to_minimize(X[i][j], Y[i][j])

ax.plot_wireframe(X, Y, S, rstride=5, cstride=5, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("function_to_minimize_1.pdf")

# initialize the optimization
scope = 25
x_star = np.random.uniform(-scope, scope)
y_star = np.random.uniform(-scope, scope)

# iterate the gradient algorithm
N_iterations = 100
gamma = 0.01
for iteration in range(N_iterations):
    _xgradient = xgradient(x_star, y_star)
    _ygradient = ygradient(x_star, y_star)
    x_star = x_star - gamma * _xgradient
    y_star = y_star - gamma * _ygradient
    z = function_to_minimize(x_star, y_star)
    if iteration % 5 == 0:
        ax.scatter(x_star, y_star, z, marker="x", color="red")
        plt.savefig(f"function_1/{iteration}.pdf")
    print(f"x* : {x_star:.2f}, y* : {y_star:.2f}, z : {z:.2f}")
