import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_position(
    agent_position: tuple[int, int],
    world: np.ndarray,
    step: int,
    image_folder: str,
) -> None:
    """
    plot the agent in its environment
    """
    figpath = os.path.join(image_folder, f"agent_position_step_{step}.pdf")
    plt.savefig(figpath)
    plt.close()


def plot_value_function(
    ax,
    title,
    array: np.ndarray,
) -> None:
    """
    plot the value function while we compute it
    """

def plot_all(
    agent_position: np.ndarray,
    value_function: np.ndarray,
    world: np.ndarray,
    step: int,
    image_folder,
) -> None:
    fig, (ax_pos, ax_val) = plt.subplots(1, 2)
    """
    plot the value function while we compute it
    """

    title = f"position of agent at step {step}"
    world_copy = np.copy(world)
    world_copy[agent_position[0], agent_position[1]] = 3
    ax_pos.imshow(world_copy)
    ax_pos.set_title("Agent position")

    # ax_val.colorbar()
    divider = make_axes_locatable(ax_val)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax_val.imshow(value_function, cmap='bone')
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax_val.set_title("Value function")

    title = f"value iteration, step {step}"
    fig.suptitle(title)
    figpath = os.path.join(image_folder, f"value_iteration_step_{step}.pdf")
    plt.savefig(figpath)
    plt.close()
