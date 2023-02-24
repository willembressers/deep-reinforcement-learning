"""Fixed-size buffer to store experience tuples."""
import configparser
import pathlib
import random
from collections import deque, namedtuple

import torch
from numpy import array, hstack, uint8, vstack
from torch import Tensor, from_numpy


class ReplayBuffer:
    """Buffer to store experience."""

    def __init__(
        self,
        action_size: int,
        num_agents: int,
        state_size: int,
        device,
    ) -> None:
        """Initialize a ReplayBuffer object."""
        self.num_agents: int = num_agents
        self.state_size: int = state_size
        self.action_size: int = action_size

        # load the configuration from the config.ini file
        config = configparser.ConfigParser()
        config.read(pathlib.Path(".") / "assets" / "config.ini")
        buffer_size: float = config.getint(
            "replay_buffer", "buffer_size", fallback=100000
        )
        self.batch_size: int = config.getint(
            "replay_buffer", "batch_size", fallback=128
        )
        self.seed = random.seed(config.getint("replay_buffer", "seed", fallback=1234))

        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.device = device

    def add(self, states, actions, rewards, next_states, dones) -> None:
        """Add experience to memory.

        Args:
            states (_type_): _description_
            actions (_type_): _description_
            rewards (_type_): _description_
            next_states (_type_): _description_
            dones (_type_): _description_
        """
        self.memory.append(
            self.experience(
                hstack(states),
                hstack(actions),
                array(rewards),
                hstack(next_states),
                array(dones),
            )
        )

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states: Tensor = (
            from_numpy(vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )

        actions: Tensor = (
            from_numpy(vstack([e.action for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )

        rewards: Tensor = (
            from_numpy(vstack([e.reward for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )

        next_states: Tensor = (
            from_numpy(vstack([e.next_state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )

        dones: Tensor = (
            from_numpy(
                vstack([e.done for e in experiences if e is not None]).astype(uint8)
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def sufficient(self) -> bool:
        """Check if the current memory is large enough.

        Returns:
            bool: _description_
        """
        return len(self.memory) > self.batch_size
