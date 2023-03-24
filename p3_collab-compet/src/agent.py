# python core modules
import random
import types

# 3rd party modules
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# custom modules
from src.actor import Actor
from src.critic import Critic
from src.ou_noise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Choose the fastest processor (in the hardware)
# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# elif torch.backends.mps.is_available() & torch.backends.mps.is_built():
#     device = torch.device("mps")


class Agent:

    actor = types.SimpleNamespace()
    critic = types.SimpleNamespace()

    def __init__(self, config, state_size, action_size, num_agents):
        """Initialize the agent.

        Args:
            config (_type_): _description_
            state_size (_type_): _description_
            action_size (_type_): _description_
            num_agents (_type_): _description_
        """

        # get the parameters
        self.config = config
        seed = config.getint("default", "seed", fallback=1234)
        self.learn_episode = config.getint("default", "learn_episode", fallback=100)

        self.gamma = config.getfloat("agent", "gamma", fallback=0.99)
        lr_actor = config.getfloat("agent", "lr_actor", fallback=0.0001)
        lr_critic = config.getfloat("agent", "lr_critic", fallback=0.001)
        weight_decay = config.getint("agent", "weight_decay", fallback=0)
        self.reduction_rate = config.getfloat("agent", "reduction_rate", fallback=0.99)
        self.reduction_ratio = config.getfloat("agent", "reduction_ratio", fallback=1.0)
        self.reduction_end = config.getfloat("agent", "reduction_end", fallback=0.1)

        # set the class variables
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor.local = Actor(config, state_size, action_size).to(device)
        self.actor.target = Actor(config, state_size, action_size).to(device)
        self.actor.optimizer = optim.Adam(self.actor.local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic.local = Critic(
            config, state_size * num_agents, action_size * num_agents
        ).to(device)
        self.critic.target = Critic(
            config, state_size * num_agents, action_size * num_agents
        ).to(device)
        self.critic.optimizer = optim.Adam(
            self.critic.local.parameters(), lr=lr_critic, weight_decay=weight_decay
        )

        # set the states of the network
        self.soft_update(self.critic.local, self.critic.target, 1)
        self.soft_update(self.actor.local, self.actor.target, 1)

        # Noise process
        self.noise = OUNoise(config, action_size)

    def act(self, state, episode=0, add_noise=True):
        """Act on the given state.

        Args:
            state (_type_): _description_
            episode (_type_): _description_
            add_noise (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor.local.eval()
        with torch.no_grad():
            action = self.actor.local(torch.unsqueeze(state, 0)).cpu().data.numpy()
        self.actor.local.train()

        if add_noise:
            if (
                episode > self.learn_episode
                and self.reduction_ratio > self.reduction_end
            ):
                self.reduction_ratio = self.reduction_rate ** (
                    episode - self.learn_episode
                )
            action += self.reduction_ratio * (
                0.5 * np.random.standard_normal(self.action_size)
            )

        if add_noise:
            action += self.noise.sample()

        # clip the actions are between -1 and 1
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        """Reset the noise to mean (mu)."""
        self.noise.reset()

    def learn(self, experiences):
        """learn from the given batch of experiences.

        Args:
            experiences (_type_): _description_
        """
        (
            full_states,
            actions,
            actor_local_actions,
            actor_target_actions,
            agent_state,
            agent_action,
            agent_reward,
            agent_done,
            next_states,
            next_full_states,
        ) = experiences

        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic.target(next_full_states, actor_target_actions)

        # Compute Q targets for current states (y_i)
        Q_targets = agent_reward + (self.gamma * Q_targets_next * (1 - agent_done))

        # Compute critic loss
        Q_expected = self.critic.local(full_states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic.local(full_states, actor_local_actions).mean()

        # Minimize the loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Update the network parameters.

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
