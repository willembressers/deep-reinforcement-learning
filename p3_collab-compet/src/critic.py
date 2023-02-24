"""This file contains the Critic class that is a NN that predicts actions based on the given state."""
import configparser
import pathlib

from torch import cat, manual_seed
from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.init import calculate_gain, xavier_uniform_


class Critic(Module):
    """Critic (Value) Model."""

    def __init__(self, state_size: int, action_size: int, num_agents: int) -> None:
        """Initialize the critic.

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
        fc1_units: int = config.getint("critic", "fc1_units", fallback=512)
        fc2_units: int = config.getint("critic", "fc2_units", fallback=256)
        fc3_units: int = config.getint("critic", "fc3_units", fallback=128)
        self.seed = manual_seed(config.getint("critic", "seed", fallback=1234))

        # initialize weight gains
        self.relu_gain = calculate_gain("relu")
        self.linear_gain = calculate_gain("linear")

        # layers
        self.fc1 = Linear(num_agents * state_size, fc1_units)
        self.fc2 = Linear(fc1_units + (num_agents * action_size), fc2_units)
        self.fc3 = Linear(fc2_units, fc3_units)
        self.fc4 = Linear(fc3_units, 1)
        self.bn1 = BatchNorm1d(fc1_units)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the weight paramaters by uniformal distribution."""
        xavier_uniform_(self.fc1.weight.data, self.relu_gain)
        xavier_uniform_(self.fc2.weight.data, self.relu_gain)
        xavier_uniform_(self.fc3.weight.data, self.relu_gain)
        xavier_uniform_(self.fc4.weight.data, self.linear_gain)

    def forward(self, state, action):
        """Forward pass that maps (state, action) pairs -> Q-values."""
        xs = self.bn1(self.fc1(state)).relu()
        x = cat((xs, action), dim=1)
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        return self.fc4(x)
