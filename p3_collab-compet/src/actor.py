"""This file contains the Actor class that is a NN that predicts actions based on the given state."""
import configparser
import pathlib

from torch import manual_seed
from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.init import calculate_gain, xavier_uniform_


class Actor(Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int) -> None:
        """Initialize the actor.

        Args:
            state_size (int): Describes the size of the state space.
            action_size (int): Describes the size of the action space.
        """
        super().__init__()

        self.state_size: int = state_size
        self.action_size: int = action_size

        # load the configuration from the config.ini file
        config = configparser.ConfigParser()
        config.read(pathlib.Path(__file__).parents[1] / "assets" / "config.ini")
        fc1_units: int = config.getint("actor", "fc1_units", fallback=256)
        fc2_units: int = config.getint("actor", "fc2_units", fallback=128)
        self.seed = manual_seed(config.getint("actor", "seed", fallback=1234))

        # initialize weight gains
        self.relu_gain = calculate_gain("relu")
        self.tanh_gain = calculate_gain("tanh")

        # layers
        self.fc1: Linear = Linear(state_size, fc1_units)
        self.fc2: Linear = Linear(fc1_units, fc2_units)
        self.fc3: Linear = Linear(fc2_units, action_size)
        self.bn1: BatchNorm1d = BatchNorm1d(fc1_units)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the weight paramaters by uniformal distribution."""
        xavier_uniform_(self.fc1.weight.data, self.relu_gain)
        xavier_uniform_(self.fc2.weight.data, self.relu_gain)
        xavier_uniform_(self.fc3.weight.data, self.tanh_gain)

    def forward(self, state):
        """Forward pass that maps states -> actions."""
        x = self.bn1(self.fc1(state)).relu()
        x = self.fc2(x).relu()
        return self.fc3(x).tanh()
