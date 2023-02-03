"""Fixed-size buffer to store experience tuples."""

import random
from collections import deque, namedtuple

from numpy import uint8, vstack
from torch import Tensor, from_numpy


class ReplayBuffer:
    """Buffer to store experience."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int,
                 device) -> None:
        """Initialize a ReplayBuffer object.

        Args:
            action_size (int): _description_
            buffer_size (int): _description_
            batch_size (int): _description_
            device (_type_): _description_
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
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
        for i, _ in enumerate(states):
            self.memory.append(self.experience(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            ))

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states: Tensor = from_numpy(vstack(
            [e.state for e in experiences if e is not None]
        )).float().to(self.device)

        actions: Tensor = from_numpy(vstack(
            [e.action for e in experiences if e is not None]
        )).float().to(self.device)

        rewards: Tensor = from_numpy(vstack(
            [e.reward for e in experiences if e is not None]
        )).float().to(self.device)

        next_states: Tensor = from_numpy(vstack(
            [e.next_state for e in experiences if e is not None]
        )).float().to(self.device)

        dones: Tensor = from_numpy(vstack(
            [e.done for e in experiences if e is not None]
        ).astype(uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)
