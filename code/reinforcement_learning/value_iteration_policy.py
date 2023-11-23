"""
Compute and use the optimal policy after computing the
value by Value iteration
"""

import os
from pydoc import doc

import numpy as np
import matplotlib.pyplot as plt
from value_iteration import clean, pick_random_position

image_folder = os.path.join("images", "value_iteration_policy")
clean(image_folder)

# load world
world_path = os.path.join("data", "world.npy")
world = np.load(world_path)
available_positions = np.where(world)

# load reward
reward_path = os.path.join("data", "reward.npy")
reward = np.load(reward_path)

# load value function
value_function_path = os.path.join("data", "value_function.npy")
value_function = np.load(value_function_path)


def pick_new_position(
    agent_position: tuple[int, int], value_function: np.ndarray
) -> tuple[int, int]:
    """
    policy: choose the bet action based on the value function

    Add lines here
    """
    return agent_position

def plot_position(step, travel, agent_position, world):
    title = f"position of agent at step {step}"
    # we need to make a copy otherwise it will not work
    world_copy = np.copy(world)
    world_copy[agent_position[0], agent_position[1]] = 3
    plt.imshow(world_copy)
    plt.title(title)
    figpath = os.path.join(
        image_folder, f"agent_position_travel_{travel}_step_{step}.pdf"
    )
    plt.savefig(figpath)
    plt.close()


if __name__ == "__main__":
    for travel in range(1, 10):
        print("---\npick new random position for agent")
        agent_position = pick_random_position(available_positions)
        for step in range(15):
            plot_position(step, travel, agent_position, world)
            agent_position = pick_new_position(agent_position, value_function)
            print(f"step {step} : agent position {agent_position}")
