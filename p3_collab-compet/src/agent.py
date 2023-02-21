"""This file contains the Agent class, which will interact with its environment."""
import configparser
import pathlib
from types import SimpleNamespace

import torch
import torch.optim as optim
from numpy import clip, ndarray
from numpy.random import randn
from src.actor import Actor
from src.critic import Critic
from src.ou_noise import OUNoise
from src.replay_buffer import ReplayBuffer
from torch import Tensor, device, from_numpy, load, manual_seed, no_grad
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_


class Agent:
    """An Agent will learn and interact with the environment."""

    local: SimpleNamespace = SimpleNamespace()
    target: SimpleNamespace = SimpleNamespace()

    def __init__(self, state_size: int, action_size: int, device: device) -> None:
        """Initialize the agent.

        Args:
            state_size (int): Describes the size of the state space.
            action_size (int): Describes the size of the action space.
            device (device): Describes the processor to run on.
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.device: device = device

        # load the configuration from the config.ini file
        config = configparser.ConfigParser()
        config.read(pathlib.Path(".") / "assets" / "config.ini")

        self.batch_size: float = config.getint("agent", "batch_size", fallback=128)
        self.gamma: float = config.getfloat("agent", "gamma", fallback=0.99)
        self.tau: float = config.getfloat("agent", "tau", fallback=0.001)
        self.lr_actor: float = config.getfloat("agent", "lr_actor", fallback=0.001)
        self.lr_critic: float = config.getfloat("agent", "lr_critic", fallback=0.001)
        self.weight_decay: int = config.getint("agent", "weight_decay", fallback=0)
        self.sigma: float = config.getfloat("agent", "sigma", fallback=0.1)

        # Initialize the local networks
        self.local.actor: Actor = Actor(state_size, action_size).to(self.device)
        self.local.critic: Critic = Critic(state_size, action_size).to(self.device)

        # Initialize the target networks
        self.target.actor: Actor = Actor(state_size, action_size).to(self.device)
        self.target.critic: Critic = Critic(state_size, action_size).to(self.device)

        # define the optimizers (actor / critic)
        self.local.actor_optimizer = optim.Adam(
            self.local.actor.parameters(), lr=self.lr_actor
        )
        self.local.critic_optimizer = optim.Adam(
            self.local.critic.parameters(),
            lr=self.lr_critic,
            weight_decay=self.weight_decay,
        )

        # Make some noise
        self.noise: OUNoise = OUNoise(action_size)

        # define the replay buffer
        self.memory: ReplayBuffer = ReplayBuffer(
            action_size, self.batch_size, self.device
        )

    def act(self, state: ndarray, add_noise=True) -> ndarray:
        """Act on the given state.

        Args:
            state (ndarray): The current state of the environment.
            add_noise (bool, optional): _description_. Defaults to True.

        Returns:
            ndarray: a 2d (movement, jumping) array describing the action of the agent.
        """
        # select a random action
        action: ndarray = randn(self.action_size)

        # clip the actions are between -1 and 1
        return clip(action, -1.0, 1.0)

    def reset(self) -> None:
        """Reset the noise to mean (mu)."""
        self.noise.reset()

    def save(self, subdir: str, index: int) -> None:
        """Save the actor and critic networks.

        Args:
            index (int): An identifier which differentiates between the agents.
            subdir (str): Which subdirectory to write to.
        """
        # ensure the directory exists
        path = pathlib.Path(__file__).parents[1] / "checkpoints" / subdir
        path.mkdir(parents=True, exist_ok=True)

        # save the actor and the critic.
        torch.save(self.local.actor.state_dict(), path / f"agent_{index}_actor.pth")
        torch.save(self.local.critic.state_dict(), path / f"agent_{index}_critic.pth")

    def step(
        self,
        state: ndarray,
        actions: ndarray,
        rewards: list,
        next_state: ndarray,
        dones: list,
    ):
        """Save experience in replay memory.

        Args:
            state (ndarray): _description_
            actions (ndarray): _description_
            rewards (list): _description_
            next_state (ndarray): _description_
            dones (list): _description_
        """
        self.memory.add(state, actions, rewards, next_state, dones)

    def learn(self):
        """Learn, if enough samples are available in memory."""
        # ensure we've sufficient in memory
        if len(self.memory) > self.batch_size:

            # loop 10 times (learn from 10 samples)
            for _ in range(10):
                states, actions, rewards, next_states, dones = self.memory.sample()

                # Get predicted next-state actions and Q values from target models
                actions_next = self.target.actor(next_states)
                q_targets_next = self.target.critic(next_states, actions_next)

                # Compute Q targets for current states (y_i)
                q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

                # Compute critic loss
                q_expected = self.local.critic(states, actions)
                critic_loss: Tensor = mse_loss(q_expected, q_targets)

                # Minimize the loss
                self.local.critic_optimizer.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.local.critic.parameters(), 1)
                self.local.critic_optimizer.step()

                # Compute actor loss
                actions_pred = self.local.actor(states)
                actor_loss = -self.local.critic(states, actions_pred).mean()

                # Minimize the loss
                self.local.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.local.actor_optimizer.step()

                # update target networks
                self.soft_update(self.local.critic, self.target.critic, self.tau)
                self.soft_update(self.local.actor, self.target.actor, self.tau)

    def soft_update(self, local_model, target_model, tau) -> None:
        """Soft update model parameters.

        Args:
            local_model (_type_): _description_
            target_model (_type_): _description_
            tau (_type_): _description_
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
