"""
Perform the value iteration algorithm in a simple world
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt

image_folder = os.path.join("images", "value_iteration")

# load world
world_path = os.path.join("data", "world.npy")
world = np.load(world_path)
available_positions = np.where(world)

# load reward
reward_path = os.path.join("data", "reward.npy")
reward = np.load(reward_path)

# set discount factor
gamma = 0.8

# initialize value function and reward
value_function = np.zeros(world.shape)
# the reward from the point of view of the agent
known_reward = np.zeros(world.shape)


def clean(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


def pick_random_position(available_positions: np.ndarray) -> tuple[int, int]:
    """
    pick a random position in order to initialize the position of our agent
    """
    number_of_available_positions = available_positions[0].shape[0]
    random_index = random.randint(0, number_of_available_positions - 1)
    i_coordinate = available_positions[0][random_index]
    j_coordinate = available_positions[1][random_index]
    return i_coordinate, j_coordinate


def plot_position(
    agent_position: tuple[int, int],
    world: np.ndarray,
    step: int,
) -> None:
    """
    plot the agent in its environment
    """
    title = f"position of agent at step {step}"
    world_copy = np.copy(world)
    world_copy[agent_position[0], agent_position[1]] = 3
    plt.imshow(world_copy)
    plt.title(title)
    figpath = os.path.join(image_folder, f"agent_position_step_{step}.pdf")
    plt.savefig(figpath)
    plt.close()


def plot_value_function(value_function: np.ndarray, step: int) -> None:
    """
    plot the value function while we compute it
    """
    title = f"value function at step {step}"
    plt.imshow(value_function)
    plt.colorbar()
    plt.title(title)
    figpath = os.path.join(image_folder, f"value_function_step_{step}.pdf")
    plt.savefig(figpath)
    plt.close()


def new_position_available(new_position: tuple[int, int], world: np.ndarray) -> bool:
    """
    check if the position new_position is compatible with
    the world we created
    """
    # in python, this can be interpreted as a boolean
    return world[new_position[0], new_position[1]]


def move_agent(agent_position: tuple[int, int], world: np.ndarray):
    """
    determine a new position for the agent (randomly),
    in order to continue the exploration of the environment,
    possibly to find rewards at new positions.
    """
    # boolean representing if we moved the agent
    moved_agent = False
    # try to move the agent until it moves
    while not moved_agent:
        direction = random.randint(0, 3)
        if direction == 0:
            # try to go left
            new_position = (agent_position[0], agent_position[1] - 1)
        elif direction == 1:
            # try to go top
            new_position = (agent_position[0] - 1, agent_position[1])
        elif direction == 2:
            # try to go right
            new_position = (agent_position[0], agent_position[1] + 1)
        elif direction == 3:
            # try to go bottom
            new_position = (agent_position[0] + 1, agent_position[1])
        # check if position is available
        if new_position_available(new_position, world):
            # go out of the loop
            moved_agent = True
    return new_position


def update_value_function(
    value_function: np.ndarray, known_reward: np.ndarray
) -> np.ndarray:
    """
    Update the value function according to the Bellman equation
    """
    for i in range(1, world.shape[0] - 1):
        for j in range(1, world.shape[0] - 1):
            # still check that the position is available
            # otherwise, the value should stay at 0
            if world[i, j]:
                value_function[i, j] = known_reward[i, j] + max(
                    gamma * value_function[i - 1, j],
                    gamma * value_function[i, j - 1],
                    gamma * value_function[i, j + 1],
                    gamma * value_function[i + 1, j],
                )
    return value_function


if __name__ == "__main__":
    agent_position = pick_random_position(available_positions)
    clean(image_folder)
    for step in range(200):
        print(f"step {step} : agent position {agent_position}")
        plot_position(agent_position, world, step)

        # move the agent randomly
        agent_position = move_agent(agent_position, world)

        # cherck if there is a reward at that position
        obtained_reward = reward[agent_position[0], agent_position[1]]
        if obtained_reward:
            if not known_reward[agent_position[0], agent_position[1]]:
                print(
                    f"----- found reward in position {agent_position}: {obtained_reward}"
                )
                known_reward[agent_position[0], agent_position[1]] = obtained_reward

        # update the value function with the Bellmann equation
        value_function = update_value_function(value_function, known_reward)
        plot_value_function(value_function, step)

        # periodically reinitialize the position of the agent.
        if step % 15 == 0:
            print("----- re initialize agent position")
            agent_position = pick_random_position(available_positions)

    # safe our evaluation for usage later
    value_function_path = os.path.join("data", "value_function.npy")
    np.save(value_function_path, value_function)
