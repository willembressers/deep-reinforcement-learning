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
from torch import Tensor, device, from_numpy, load, manual_seed, no_grad
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_


class Agent:
    """An Agent will learn and interact with the environment."""

    local: SimpleNamespace = SimpleNamespace()
    target: SimpleNamespace = SimpleNamespace()

    def __init__(
        self,
        index: int,
        state_size: int,
        action_size: int,
        num_agents: int,
        device: device,
    ) -> None:
        """Initialize the agent.

        Args:
            index (int): _description_
            state_size (int): _description_
            action_size (int): _description_
            num_agents (int): _description_
            device (device): _description_
        """
        self.index: int = index
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.device: device = device

        # load the configuration from the config.ini file
        config = configparser.ConfigParser()
        config.read(pathlib.Path(".") / "assets" / "config.ini")
        self.gamma: float = config.getfloat("agent", "gamma", fallback=0.99)
        self.tau: float = config.getfloat("agent", "tau", fallback=0.001)
        self.lr_actor: float = config.getfloat("agent", "lr_actor", fallback=0.001)
        self.lr_critic: float = config.getfloat("agent", "lr_critic", fallback=0.001)
        self.weight_decay: int = config.getint("agent", "weight_decay", fallback=0)

        # Initialize the local networks
        self.local.actor: Actor = Actor(state_size, action_size).to(self.device)
        self.local.critic: Critic = Critic(state_size, action_size, num_agents).to(
            self.device
        )

        # Initialize the target networks
        self.target.actor: Actor = Actor(state_size, action_size).to(self.device)
        self.target.critic: Critic = Critic(state_size, action_size, num_agents).to(
            self.device
        )

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

    def learn(self, agents, experiences):
        """Learn, if enough samples are available in memory."""
        # unstack the experience
        (
            states_flat,
            actions_flat,
            rewards_flat,
            next_states_flat,
            dones_flat,
        ) = experiences

        # get the number of agents
        num_agents = len(agents)

        # reshape the tensors
        states = states_flat.reshape(-1, num_agents, self.state_size)
        actions = actions_flat.reshape(-1, num_agents, self.action_size)
        rewards = rewards_flat.reshape(-1, num_agents)
        next_states = next_states_flat.reshape(-1, num_agents, self.state_size)
        dones = torch.max(dones_flat, dim=1).values.reshape(-1, 1)

        # Get predicted next-state actions and Q values from target models
        next_actions: tuple = ()
        for index, agent in enumerate(agents):
            next_state = next_states[:, index]
            next_actions += (agent.target.actor(next_state),)
        next_actions_flat = torch.cat(next_actions, dim=1)

        # get the critic's opinion
        Q_targets_next = self.target.critic(next_states_flat, next_actions_flat)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:, self.index].reshape(-1, 1) + (
            self.gamma * Q_targets_next * (1 - dones)
        )

        # Compute critic loss
        Q_expected = self.local.critic(states_flat, actions_flat)
        critic_loss = mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.local.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.local.critic.parameters(), 1)
        self.local.critic_optimizer.step()

        # Compute actor loss
        pred_actions: tuple = ()
        for index, agent in enumerate(agents):
            state = states[:, index]
            pred_actions += (agent.local.actor(state),)
        pred_actions_flat = torch.cat(pred_actions, dim=1)
        actor_loss = -self.local.critic(states_flat, pred_actions_flat).mean()

        # Minimize the loss
        self.local.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.local.actor_optimizer.step()

        # update target networks
        self.soft_update(self.local.critic, self.target.critic)
        self.soft_update(self.local.actor, self.target.actor)

    def soft_update(self, local_model, target_model):
        tau = self.tau
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
