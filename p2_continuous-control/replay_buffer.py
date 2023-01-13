"""Fixed-size buffer to store experience tuples."""

import random
from collections import deque, namedtuple

from numpy import uint8, vstack
from torch import Tensor, from_numpy


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        for i in range(len(states)):
            e = self.experience(states[i], actions[i], rewards[i],
                                next_states[i], dones[i])
            self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
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

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
