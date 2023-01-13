"""Train the agent."""
from collections import deque
from pathlib import Path

import numpy as np
from agent import Agent
from torch import save
from tqdm import tqdm


class DDPG:
    """Train an agent on an environment."""

    episodes: int = 500
    timesteps: int = 1000
    scores: list = []
    scores_window: deque = deque(maxlen=100)  # last 100 scores
    checkpoint_dir: Path = Path('.') / 'checkpoints'
    model_dir: Path = Path('.') / 'models'

    def __init__(self, env, brain_name, agent) -> None:
        """Initialize the Deep Determenistic Policy Gradient class.

        Args:
            env (_type_): _description_
            brain_name (_type_): _description_
            agent (_type_): _description_
            episodes (int, optional): _description_. Defaults to 500.
            timesteps (int, optional): _description_. Defaults to 1000.
        """
        self.env = env
        self.brain_name = brain_name
        self.agent: Agent = agent

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents: int = len(env_info.agents)

    def train(self) -> None:
        """Train the agents."""
        progress_bar = tqdm(
            range(1, self.episodes + 1),
            desc="Training agents"
        )

        for episode in progress_bar:
            # reset the environment
            env_info = self.env.reset(train_mode=True)[self.brain_name]

            # get the current state (for each agent)
            states = env_info.vector_observations

            # reset agent
            self.agent.reset()

            # initialize the score (for each agent)
            scores: np.ndarray = np.zeros(self.num_agents)

            # loop over all time steps
            for timestep in range(self.timesteps):

                # let the agent decide which actions to take
                actions = self.agent.act(states, add_noise=True)

                # send all actions to tne environment
                env_info = self.env.step(actions)[self.brain_name]

                # get next states, rewards and done flags (for each agent)
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # Save experience in replay memory
                self.agent.step(states, actions, rewards, next_states, dones)
                if timestep % 20 == 0:
                    self.agent.sample_and_learn()

                # set the next states as the current state
                states = next_states

                # accumulate rewards
                scores += rewards

                # exit loop if episode finished
                if np.any(dones):
                    break

            self.save_checkpoints(episode)

            # save most recent scores
            self.scores.append(scores)
            self.scores_window.append(scores)

            # get the mean
            avg_scores: float = np.mean(self.scores)
            avg_scores_window: float = np.mean(self.scores_window)
            progress_bar.set_description(
                f"Training progress: {avg_scores:.2f} / {avg_scores_window:.2f}")

    def save_checkpoints(self, episode) -> None:
        """Save the agents."""
        if episode % 10 == 0:
            save(
                self.agent.local_actor.state_dict(),
                self.checkpoint_dir / f'actor-{episode}.pth'
            )
            save(
                self.agent.local_critic.state_dict(),
                self.checkpoint_dir / f'critic-{episode}.pth'
            )

    def save_models(self) -> None:
        """Save the agents."""
        # save the actor
        save(
            {
                'state_size': self.agent.state_size,
                'action_size': self.agent.action_size,
                'fc1_units': self.agent.actor_local.fc1_units,
                'fc2_units': self.agent.actor_local.fc2_units,
                'state_dict': self.agent.actor_local.state_dict()
            },
            self.model_dir / 'saved_model_actor.pth'
        )

        # save the critic
        save(
            {
                'state_size': self.agent.state_size,
                'action_size': self.agent.action_size,
                'fc1_units': self.agent.local_critic.fc1_units,
                'fc2_units': self.agent.local_critic.fc2_units,
                'state_dict': self.agent.local_critic.state_dict()
            },
            self.model_dir / 'saved_model_critic.pth'
        )
