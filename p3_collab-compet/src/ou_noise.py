"""Ornstein-Uhlenbeck process."""
import configparser
import copy
import pathlib
import random

from numpy import array, ndarray, ones


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    state = None

    def __init__(self, action_size: int) -> None:
        """Initialize parameters and noise process.

        Args:
            action_size (int): Describes the size of the action space.
        """
        # load the configuration from the config.ini file
        config = configparser.ConfigParser()
        config.read(pathlib.Path(".") / "assets" / "config.ini")
        self.mu: ndarray = config.getfloat("noise", "mu", fallback=0.0) * ones(
            action_size
        )
        self.theta: float = config.getfloat("noise", "theta", fallback=0.15)
        self.sigma: float = config.getfloat("noise", "sigma", fallback=0.2)
        self.seed = random.seed(config.getint("noise", "seed", fallback=1234))
        self.reset()

    def reset(self) -> None:
        """Reset the noise to mu."""
        self.state = copy.copy(self.mu)

    def sample(self) -> ndarray:
        """Add a delta (based on theta, mu, sigma) and add it to the noise.

        Returns:
            ndarray: _description_
        """
        x: ndarray = self.state
        dx: ndarray = self.theta * (self.mu - x) + self.sigma * array(
            [random.random() for i in range(len(x))]
        )
        self.state: ndarray = x + dx
        return self.state
