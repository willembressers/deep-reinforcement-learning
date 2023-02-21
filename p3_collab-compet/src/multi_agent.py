"""This file contains the MultiAgent class that will control multiple agents."""
import configparser
import pathlib

import torch
from numpy import ndarray, zeros
from src.agent import Agent


class MultiAgent:
    """A multi agent will control multiple agents."""

    def __init__(self, state_size: int, action_size: int, num_agents: int) -> None:
        """Initialize the agent.

        Args:
            state_size (int): Describes the size of the state space.
            action_size (int): Describes the size of the action space.
            num_agents (int): Describes how many agents will participate.
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.num_agents: int = num_agents

        # load the configuration from the config.ini file
        self.config = configparser.ConfigParser()
        self.config.read(pathlib.Path(__file__).parents[1] / "assets" / "config.ini")

        # Choose the fastest processor (in the hardware)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
            device = torch.device("mps")

        # initialize the agents (add the to a tuple so the order is immutable)
        self.agents: tuple = ()
        for _ in range(self.num_agents):
            self.agents += (
                Agent(state_size=state_size, action_size=action_size, device=device),
            )

    def act(self, state: ndarray, add_noise=True) -> ndarray:
        """Let the agents act on the given state.

        Args:
            state (ndarray): The current state of the environment.
            add_noise (bool, optional): The option to add noise. Defaults to True.

        Returns:
            ndarray: a matrix of (self.num_agents) rows and (self.action_size) columns.
        """
        # select an action (for each agent)
        actions: ndarray = zeros((self.num_agents, self.action_size))
        for index, agent in enumerate(self.agents):
            actions[index] = agent.act(state, add_noise)

        return actions

    def reset(self) -> None:
        """Reset all the agents."""
        for agent in self.agents:
            agent.reset()

    def save(self, episode: int) -> None:
        """Save the models.

        Args:
            episode (str): The episode will be used as subdirectory.
        """
        for index, agent in enumerate(self.agents):
            agent.save(str(episode), index)
            agent.save("latest", index)

    def step(
        self,
        state: ndarray,
        actions: ndarray,
        rewards: list,
        next_state: ndarray,
        dones: list,
    ) -> None:
        """Save experience in replay memory.

        Args:
            state (ndarray): _description_
            actions (ndarray): _description_
            rewards (list): _description_
            next_state (ndarray): _description_
            dones (list): _description_
        """
        for agent in self.agents:
            agent.step(state, actions, rewards, next_state, dones)

    def learn(self):
        """Learn from memory."""
        for agent in self.agents:
            agent.learn()
