"""Interacts with and learns from the environment."""

import random

import torch.optim as optim
from numpy import clip
from torch import Tensor, from_numpy, load, no_grad
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

from actor import Actor
from critic import Critic
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

# replay buffer size
BUFFER_SIZE: int = int(1e5)

# minibatch size
BATCH_SIZE: int = 128

# discount factor
GAMMA: float = 0.99

# for soft update of target parameters
TAU: float = 1e-3

# # learning rate
# # LR = 5e-4

# learning rate of the actor
LR_ACTOR: float = 1e-4

# l earning rate of the critic
LR_CRITIC: float = 1e-3

# # how often to update the network
# # UPDATE_EVERY = 4

# L2 weight decay
WEIGHT_DECAY = 0

# standard deviation for noise
SIGMA: float = 0.1


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, device) -> None:
        """Initialize an Agent object.

        Args:
            state_size (_type_): _description_
            action_size (_type_): _description_
            random_seed (_type_): _description_
            device (_type_): _description_
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device = device

        # Initialize the actor networks (local / target)
        self.local_actor: Actor = Actor(
            state_size,
            action_size,
            random_seed
        ).to(self.device)
        self.target_actor: Actor = Actor(
            state_size,
            action_size,
            random_seed
        ).to(self.device)

        # Initialize the critic networks (local / target)
        self.local_critic: Critic = Critic(
            state_size,
            action_size,
            random_seed
        ).to(self.device)
        self.target_critic: Critic = Critic(
            state_size,
            action_size,
            random_seed
        ).to(self.device)

        # define the optimizers (actor / critic)
        self.local_actor_optimizer = optim.Adam(
            self.local_actor.parameters(),
            lr=LR_ACTOR
        )
        self.local_critic_optimizer = optim.Adam(
            self.local_critic.parameters(),
            lr=LR_CRITIC,
            weight_decay=WEIGHT_DECAY
        )

        # Make some noise
        self.noise: OUNoise = OUNoise(
            action_size,
            random_seed,
            sigma=SIGMA
        )

        # define the replay buffer
        self.memory: ReplayBuffer = ReplayBuffer(
            action_size,
            BUFFER_SIZE,
            BATCH_SIZE,
            random_seed,
            self.device
        )

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
        if len(self.memory) > BATCH_SIZE:
            for _ in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

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
        Q_targets_next = self.target_critic(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.local_critic(states, actions)
        critic_loss: Tensor = mse_loss(Q_expected, Q_targets)

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
        self.soft_update(self.local_critic, self.target_critic, TAU)
        self.soft_update(self.local_actor, self.target_actor, TAU)

    def soft_update(self, local_model, target_model, tau) -> None:
        """Soft update model parameters.

        Args:
            local_model (_type_): _description_
            target_model (_type_): _description_
            tau (_type_): _description_
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def load_model(self, filepath: str) -> None:
        """_summary_

        Args:
            filepath (str): _description_
        """
        # load the pytorch model
        saved_model = load(filepath)

        # create a new local actor
        self.actor_local: Actor = Actor(
            state_size=saved_model['state_size'],
            action_size=saved_model['action_size'],
            seed=2,
            fc1_units=saved_model['fc1_units'],
            fc2_units=saved_model['fc2_units']
        ).to(self.device)

        # assign the weights
        self.actor_local.load_state_dict(saved_model['state_dict'])
