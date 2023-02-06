"""Interacts with and learns from the environment."""
import configparser
import pathlib

import numpy as np


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(
            self, state_size: int, action_size: int, num_agents: int) -> None:
        """Initialize the agent.

        Args:
            state_size (int): _description_
            action_size (int): _description_
            num_agents (int): _description_
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.num_agents: int = num_agents

        # load the configuration from the config.ini file
        self.config = configparser.ConfigParser()
        self.config.read(pathlib.Path('.') / 'assets' / 'config.ini')

        print(self.config)

    def act(self, states, add_noise=True):
        """Act on the given states.

        Args:
            states (_type_): _description_
            add_noise (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # select an action (for each agent)
        actions = np.random.randn(self.num_agents, self.action_size)
        actions = np.clip(actions, -1, 1)

        return actions
