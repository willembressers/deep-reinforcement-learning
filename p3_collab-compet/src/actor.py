# 3rd party modules
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, config, state_size, action_size):
        """Initialize the Actor network.

        Args:
            config (_type_): _description_
            state_size (_type_): _description_
            action_size (_type_): _description_
        """
        super(Actor, self).__init__()

        # get the parameters
        seed = config.getint("default", "seed", fallback=1234)
        fc1_units = config.getint("actor", "fc1_units", fallback=256)
        fc2_units = config.getint("actor", "fc2_units", fallback=128)

        # set the class variables
        self.seed = torch.manual_seed(seed)

        # fully connected layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # normalization layers
        self.bn1 = nn.BatchNorm1d(fc1_units)

        # initialize the weights
        self._init_weights()

    def _init_weights(self):
        """Reset the network weights."""
        # initialize the weights of the fully connected layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

        # initialize the biases of the fully connected layers to 0
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        """Do a forward pass throught the network.

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """

        x = self.fc1(state)
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.tanh(x)

        return x
