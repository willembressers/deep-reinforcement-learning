"""Actor (Policy) Model."""
from configparser import ConfigParser
from pathlib import Path

from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.init import calculate_gain, xavier_uniform_


class Actor(Module):
    """Actor (Policy) Model."""

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
        print(f"--> Actor: fc1_units: {self.fc1_units}")
        print(f"--> Actor: fc2_units: {self.fc2_units}")

        # initialize weight gains
        self.relu_gain = calculate_gain('relu')
        self.tanh_gain = calculate_gain('tanh')

        # layers
        self.fc1: Linear = Linear(state_size, self.fc1_units)
        self.fc2: Linear = Linear(self.fc1_units, self.fc2_units)
        self.fc3: Linear = Linear(self.fc2_units, action_size)
        self.bn1: BatchNorm1d = BatchNorm1d(self.fc1_units)

        self.reset_parameters()

    def __load_configuration(self) -> None:
        """Load the configuration from the config.ini file."""
        config: ConfigParser = ConfigParser()
        config.read(self.dir_assets / 'config.ini')

        self.fc1_units: int = config.getint(
            'actor', 'fc1_units', fallback=256
        )
        self.fc2_units: int = config.getint(
            'actor', 'fc2_units', fallback=128
        )

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
