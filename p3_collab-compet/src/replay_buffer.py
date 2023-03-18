# python core modules
import random
from collections import deque, namedtuple

# 3rd party modules
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, config):
        """Initialize a ReplayBuffer object.

        Args:
            config (_type_): _description_
        """
        # get the parameters
        buffer_size = config.getint("buffer", "size", fallback=100000)
        seed = config.getint("default", "seed", fallback=1234)
        self.batch_size = config.getint("default", "batch_size", fallback=256)

        # set the class variables
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "full_state",
                "action",
                "reward",
                "next_state",
                "next_full_state",
                "done",
            ],
        )
        self.seed = random.seed(seed)

    def add(self, state, full_state, action, reward, next_state, next_full_state, done):
        """Add some experience.

        Args:
            state (_type_): _description_
            full_state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            next_full_state (_type_): _description_
            done (function): _description_
        """
        e = self.experience(
            state, full_state, action, reward, next_state, next_full_state, done
        )
        self.memory.append(e)

    def sample(self):
        """Get a batch of experiences.

        Returns:
            _type_: _description_
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.array([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        full_states = (
            torch.from_numpy(
                np.array([e.full_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.array([e.action for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.array([e.reward for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.array([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_full_states = (
            torch.from_numpy(
                np.array([e.next_full_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.array([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (
            states,
            full_states,
            actions,
            rewards,
            next_states,
            next_full_states,
            dones,
        )

    def __len__(self):
        """Get the count of experiences.

        Returns:
            _type_: _description_
        """
        return len(self.memory)
