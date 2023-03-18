# python core modules
import copy
import random

# 3rd party modules
import numpy as np


class OUNoise:
    def __init__(self, config, action_size):
        """Initialize the noise.

        Args:
            config (_type_): _description_
            action_size (_type_): _description_
        """
        # get the parameters
        seed = config.getint("default", "seed", fallback=1234)
        mu = config.getfloat("agent", "mu", fallback=0.0)
        theta = config.getfloat("agent", "theta", fallback=0.15)
        sigma = config.getfloat("agent", "sigma", fallback=0.1)

        # set the class variables
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.action_size = action_size

    def reset(self):
        """Re-set the internal state."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update the internal state.

        Returns:
            _type_: _description_
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(
            self.action_size
        )
        self.state = x + dx
        return self.state
