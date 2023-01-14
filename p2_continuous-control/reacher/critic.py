"""Critic (Value) Model."""

from torch import cat
from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.init import calculate_gain, xavier_uniform_


class Critic(Module):
    """Critic (Value) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int,
                 fc1_units=400, fc2_units=300) -> None:
        """Initialize parameters and build model."""
        super().__init__()
        self.seed: int = seed

        # TODO
        self.fc1_units: int = fc1_units
        self.fc2_units: int = fc2_units

        # weight initialization gains
        self.relu_gain = calculate_gain('relu')
        self.linear_gain = calculate_gain('linear')

        # layers
        self.fcs1: Linear = Linear(state_size, fc1_units)
        self.fc2: Linear = Linear(fc1_units + action_size, fc2_units)
        self.fc3: Linear = Linear(fc2_units, 1)
        self.bn1: BatchNorm1d = BatchNorm1d(fc1_units)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the weight paramaters by uniformal distribution."""
        xavier_uniform_(self.fcs1.weight.data, self.relu_gain)
        xavier_uniform_(self.fc2.weight.data, self.relu_gain)
        xavier_uniform_(self.fc3.weight.data, self.linear_gain)

    def forward(self, state, action):
        """forward pass that maps (state, action) pairs -> Q-values."""
        xs = self.bn1(self.fcs1(state)).relu()
        x = cat((xs, action), dim=1)
        x = self.fc2(x).relu()
        return self.fc3(x)
