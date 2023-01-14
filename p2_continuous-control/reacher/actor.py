"""Actor (Policy) Model."""

from torch import manual_seed
from torch.nn import BatchNorm1d, Linear, Module
from torch.nn.init import calculate_gain, xavier_uniform_


class Actor(Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int,
                 fc1_units=400, fc2_units=300) -> None:
        """Initialize parameters and build model.

        Args:
            state_size (_type_): _description_
            action_size (_type_): _description_
            seed (_type_): _description_
            fc1_units (int, optional): _description_. Defaults to 400.
            fc2_units (int, optional): _description_. Defaults to 300.
        """
        super().__init__()
        self.seed = manual_seed(seed)

        # out_features on the fully connected layers
        self.fc1_units: int = fc1_units
        self.fc2_units: int = fc2_units

        # initialize weight gains
        self.relu_gain = calculate_gain('relu')
        self.tanh_gain = calculate_gain('tanh')

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
