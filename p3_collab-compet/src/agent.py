"""This file contains the Agent class, which will interact with its environment."""
import pathlib
from types import SimpleNamespace

import torch
from numpy import clip, ndarray
from numpy.random import randn
from src.actor import Actor
from src.critic import Critic
from src.ou_noise import OUNoise
from torch import device


class Agent:
    """An Agent will learn and interact with the environment."""

    local: SimpleNamespace = SimpleNamespace()
    target: SimpleNamespace = SimpleNamespace()

    def __init__(self, state_size: int, action_size: int, device: device) -> None:
        """Initialize the agent.

        Args:
            state_size (int): Describes the size of the state space.
            action_size (int): Describes the size of the action space.
            device (device): Describes the processor to run on.
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.device: device = device

        # Initialize the local networks
        self.local.actor: Actor = Actor(state_size, action_size).to(self.device)
        self.local.critic: Critic = Critic(state_size, action_size).to(self.device)

        # Make some noise
        self.noise: OUNoise = OUNoise(action_size)

    def act(self, state: ndarray, add_noise=True) -> ndarray:
        """Act on the given state.

        Args:
            state (ndarray): The current state of the environment.
            add_noise (bool, optional): _description_. Defaults to True.

        Returns:
            ndarray: a 2d (movement, jumping) array describing the action of the agent.
        """
        # select a random action
        action: ndarray = randn(self.action_size)

        # clip the actions are between -1 and 1
        return clip(action, -1.0, 1.0)

    def reset(self) -> None:
        """Reset the noise to mean (mu)."""
        self.noise.reset()

    def save(self, subdir: str, index: int) -> None:
        """Save the actor and critic networks.

        Args:
            index (int): An identifier which differentiates between the agents.
            subdir (str): Which subdirectory to write to.
        """
        # ensure the directory exists
        path = pathlib.Path(__file__).parents[1] / "checkpoints" / subdir
        path.mkdir(parents=True, exist_ok=True)

        # save the actor and the critic.
        torch.save(self.local.actor.state_dict(), path / f"agent_{index}_actor.pth")
        torch.save(self.local.critic.state_dict(), path / f"agent_{index}_critic.pth")
