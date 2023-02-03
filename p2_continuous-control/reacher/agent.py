"""Interacts with and learns from the environment."""

import torch.optim as optim
from numpy import clip
from reacher.actor import Actor
from reacher.critic import Critic
from reacher.ou_noise import OUNoise
from reacher.replay_buffer import ReplayBuffer
from torch import Tensor, from_numpy, load, no_grad
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_


class Agent():
    """Interacts with and learns from the environment."""

    # define the constants
    buffer_size: int = int(1e5)
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 1e-3
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    weight_decay: int = 0
    sigma: float = 0.1

    def __init__(
            self, state_size: int, action_size: int, seed: int, device) -> None:
        """Initialize an Agent object.

        Args:
            state_size (_type_): _description_
            action_size (_type_): _description_
            seed (_type_): _description_
            device (_type_): _description_
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.seed: int = seed
        self.device = device

        # Initialize the actor networks (local / target)
        self.local_actor: Actor = Actor(
            state_size, action_size, seed).to(
            self.device)
        self.target_actor: Actor = Actor(
            state_size, action_size, seed).to(
            self.device)

        # Initialize the critic networks (local / target)
        self.local_critic: Critic = Critic(
            state_size, action_size, seed).to(
            self.device)
        self.target_critic: Critic = Critic(
            state_size, action_size, seed).to(
            self.device)

        # define the optimizers (actor / critic)
        self.local_actor_optimizer = optim.Adam(
            self.local_actor.parameters(),
            lr=self.lr_actor
        )
        self.local_critic_optimizer = optim.Adam(
            self.local_critic.parameters(),
            lr=self.lr_critic,
            weight_decay=self.weight_decay
        )

        # Make some noise
        self.noise: OUNoise = OUNoise(action_size, seed, sigma=self.sigma)

        # define the replay buffer
        self.memory: ReplayBuffer = ReplayBuffer(
            action_size, self.buffer_size, self.batch_size, seed, self.device)

    def step(self, states, actions, rewards, next_states, dones) -> None:
        """Save experience in replay memory.

        Args:
            states (_type_): _description_
            actions (_type_): _description_
            rewards (_type_): _description_
            next_states (_type_): _description_
            dones (_type_): _description_
        """
        self.memory.add(states, actions, rewards, next_states, dones)

    def sample_and_learn(self) -> None:
        """Learn, if enough samples are available in memory."""
        if len(self.memory) > self.batch_size:
            for _ in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Return actions for given state as per current policy.

        Args:
            state (_type_): _description_
            add_noise (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        state: Tensor = from_numpy(state).float().to(self.device)
        self.local_actor.eval()

        with no_grad():
            action = self.local_actor(state).cpu().data.numpy()

        self.local_actor.train()

        if add_noise:
            action += self.noise.sample()

        return clip(action, -1.0, 1.0)

    def reset(self) -> None:
        """Reset the noise to mean (mu)."""
        self.noise.reset()

    def learn(self, experiences, gamma) -> None:
        """Update policy and parameters, given the experience.

        Args:
            experiences (_type_): _description_
            gamma (_type_): _description_
        """
        states, actions, rewards, next_states, dones = experiences

        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states)
        q_targets_next = self.target_critic(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Compute critic loss
        q_expected = self.local_critic(states, actions)
        critic_loss: Tensor = mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.local_critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.local_critic.parameters(), 1)
        self.local_critic_optimizer.step()

        # Compute actor loss
        actions_pred = self.local_actor(states)
        actor_loss = -self.local_critic(states, actions_pred).mean()

        # Minimize the loss
        self.local_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.local_actor_optimizer.step()

        # update target networks
        self.soft_update(self.local_critic, self.target_critic, self.tau)
        self.soft_update(self.local_actor, self.target_actor, self.tau)

    def soft_update(self, local_model, target_model, tau) -> None:
        """Soft update model parameters.

        Args:
            local_model (_type_): _description_
            target_model (_type_): _description_
            tau (_type_): _description_
        """
        for target_param, local_param in zip(
                target_model.parameters(),
                local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def load_model(self, filepath: str) -> None:
        """Load the filepath as the actor.

        Args:
            filepath (str): _description_
        """
        # load the pytorch model
        saved_model = load(filepath)

        # create a new local actor
        self.local_actor: Actor = Actor(
            state_size=saved_model['state_size'],
            action_size=saved_model['action_size'],
            seed=2,
            fc1_units=saved_model['fc1_units'],
            fc2_units=saved_model['fc2_units']
        ).to(self.device)

        # assign the weights
        self.actor_local.load_state_dict(saved_model['state_dict'])
