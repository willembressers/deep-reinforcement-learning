"""Ornstein-Uhlenbeck process."""

import copy
import random

from numpy import array, ndarray, ones


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2) -> None:
        """Initialize parameters and noise process.

        Args:
            size (_type_): _description_
            seed (_type_): _description_
            mu (_type_, optional): _description_. Defaults to 0..
            theta (float, optional): _description_. Defaults to 0.15.
            sigma (float, optional): _description_. Defaults to 0.2.
        """
        self.mu: ndarray = mu * ones(size)
        self.theta: float = theta
        self.sigma: float = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self) -> None:
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> ndarray:
        """Update internal state and return it as a noise sample.

        Returns:
            ndarray: _description_
        """
        x: ndarray = self.state
        dx: ndarray = self.theta * (self.mu - x) + self.sigma * array([random.random() for i in range(len(x))])
        self.state: ndarray = x + dx
        return self.state
