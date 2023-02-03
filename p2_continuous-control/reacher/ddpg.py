"""Train the agent."""
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import any, mean, ndarray, zeros
from reacher.agent import Agent
from torch import save
from tqdm import tqdm


class DDPG:
    """Train an agent using an Deep Determenistic Policy Gradient."""

    episodes: int = 200
    timesteps: int = 1000
    scores: list = []
    scores_window: deque = deque(maxlen=100)  # last 100 scores
    checkpoint_dir: Path = Path('.') / 'checkpoints'
    model_dir: Path = Path('.') / 'models'

    def __init__(self, env, brain_name, agent: Agent) -> None:
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
        print(f"-> Epsiodes:\t\t\t{self.episodes}")
        print(f"-> Timesteps (per episode):\t{self.timesteps}")

        self.scores: list = []

        progress_bar = tqdm(
            range(1, self.episodes + 1),
            desc="Training agents")

        for episode in progress_bar:
            # reset the environment
            env_info = self.env.reset(train_mode=True)[self.brain_name]

            # get the current state (for each agent)
            states = env_info.vector_observations

            # reset agent
            self.agent.reset()

            # initialize the score (for each agent)
            scores: ndarray = zeros(self.num_agents)

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
                if any(dones):
                    break

            self.save_checkpoints(episode)

            # save most recent scores
            self.scores.append(scores)

            # update the progress bar
            progress_bar.set_description(
                f"Training progress: ({mean(scores):.4f})")

        return self.scores

    def save_checkpoints(self, episode) -> None:
        """Save a checkpoint of the actor and critic."""
        if episode % 10 == 0:

            save(
                self.agent.local_actor.state_dict(),
                self.checkpoint_dir / 'actors' / f'{episode}.pth'
            )
            save(
                self.agent.local_critic.state_dict(),
                self.checkpoint_dir / 'critics' / f'{episode}.pth'
            )

    def save_models(self) -> None:
        """Save the actor and critic."""
        # save the actor
        save(
            {
                'state_size': self.agent.state_size,
                'action_size': self.agent.action_size,
                'fc1_units': self.agent.local_actor.fc1_units,
                'fc2_units': self.agent.local_actor.fc2_units,
                'state_dict': self.agent.local_actor.state_dict()
            },
            self.model_dir / 'actor.pth'
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
            self.model_dir / 'critic.pth'
        )

        # for convenient method chaining
        return self

    def plot_training_history(self) -> None:
        """Plot the training history."""
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.title('Training history')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.legend(
            [f"Reacher: {i}" for i in range(1, self.num_agents + 1)],
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        plt.show()

    def plot_average_scores(self, window_size: int = 100) -> None:
        """Plot the average scores, at a certain window.

        Args:
            window_size (int, optional): _description_. Defaults to 100.
        """
        # calculate the average scores
        avg_scores = np.array(self.scores).mean(axis=1)

        # calculate the average scores on a certain window
        avg_scores_window: list = [
            np.mean(avg_scores[max(0, i - window_size): i + 1])
            for i in range(len(avg_scores))
        ]

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(
            np.arange(1, len(self.scores) + 1),
            avg_scores,
            label="window size 1"
        )
        plt.plot(
            np.arange(1, len(self.scores) + 1),
            avg_scores_window,
            label=f"window size {window_size}"
        )
        plt.plot(
            np.arange(1, len(self.scores) + 1),
            [30] * len(self.scores),
            linestyle='--',
            color='red'
        )
        plt.title('Training history | average scores')
        plt.ylabel('Average Score')
        plt.xlabel('Episode')
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        plt.show()
