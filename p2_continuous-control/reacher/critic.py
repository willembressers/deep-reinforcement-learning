"""Critic (Value) Model."""
from configparser import ConfigParser
from pathlib import Path

from torch import cat
from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.init import calculate_gain, xavier_uniform_


class Critic(Module):
    """Critic (Value) Model."""

    dir_assets: Path = Path('.') / 'assets'

    def __init__(self, state_size: int, action_size: int) -> None:
        """Initialize parameters and build model.

        Args:
            state_size (_type_): _description_
            action_size (_type_): _description_
        """
        super().__init__()
        # load the configuration
        self.__load_configuration()
        print(f"--> Critic: fc1_units: {self.fc1_units}")
        print(f"--> Critic: fc2_units: {self.fc2_units}")

        # initialize weight gains
        self.relu_gain = calculate_gain('relu')
        self.linear_gain = calculate_gain('linear')

        # layers
        self.fc1: Linear = Linear(state_size, self.fc1_units)
        self.fc2: Linear = Linear(self.fc1_units + action_size, self.fc2_units)
        self.fc3: Linear = Linear(self.fc2_units, 1)
        self.bn1: BatchNorm1d = BatchNorm1d(self.fc1_units)

        self.reset_parameters()

    def __load_configuration(self) -> None:
        """Load the configuration from the config.ini file."""
        config: ConfigParser = ConfigParser()
        config.read(self.dir_assets / 'config.ini')

        self.fc1_units: int = config.getint(
            'critic', 'fc1_units', fallback=256
        )
        self.fc2_units: int = config.getint(
            'critic', 'fc2_units', fallback=128
        )

    def reset_parameters(self) -> None:
        """Reset the weight paramaters by uniformal distribution."""
        xavier_uniform_(self.fc1.weight.data, self.relu_gain)
        xavier_uniform_(self.fc2.weight.data, self.relu_gain)
        xavier_uniform_(self.fc3.weight.data, self.linear_gain)

    def forward(self, state, action):
        """forward pass that maps (state, action) pairs -> Q-values."""
        xs = self.bn1(self.fc1(state)).relu()
        x = cat((xs, action), dim=1)
        x = self.fc2(x).relu()
        return self.fc3(x)
