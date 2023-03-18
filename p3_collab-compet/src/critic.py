# 3rd party modules
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, config, full_state_size, actions_size):
        """Initialize the Critic network.

        Args:
            config (_type_): _description_
            full_state_size (_type_): _description_
            actions_size (_type_): _description_
        """
        super(Critic, self).__init__()

        # get the parameters
        seed = config.getint("default", "seed", fallback=1234)
        fc1_units = config.getint("critic", "fc1_units", fallback=256)
        fc2_units = config.getint("critic", "fc2_units", fallback=128)

        # set the class variables
        self.seed = torch.manual_seed(seed)

        # fully connected layers
        self.fc1 = nn.Linear(full_state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + actions_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

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

    def forward(self, state, action):
        """Do a forward pass throught the network.

        Args:
            state (_type_): _description_
            action (_type_): _description_

        Returns:
            _type_: _description_
        """

        x = self.fc1(state)
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x
