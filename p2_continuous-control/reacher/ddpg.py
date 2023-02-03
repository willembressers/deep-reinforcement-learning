"""Train the agent."""
from collections import deque
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import any, mean, ndarray, zeros
from reacher.agent import Agent
from torch import save
from tqdm import tqdm


class DDPG:
    """Train an agent using an Deep Determenistic Policy Gradient."""

    scores: list = []
    scores_window: deque = deque(maxlen=100)  # last 100 scores
    dir_checkpoint: Path = Path('.') / 'checkpoints'
    dir_models: Path = Path('.') / 'models'
    dir_assets: Path = Path('.') / 'assets'
    timestamp = ""

    def __init__(self, env, brain_name, agent: Agent) -> None:
        """Initialize the Deep Determenistic Policy Gradient class.

        Args:
            env (_type_): _description_
            brain_name (_type_): _description_
            agent (_type_): _description_
            episodes (int, optional): _description_. Defaults to 500.
            timesteps (int, optional): _description_. Defaults to 1000.
        """
        # load the configuration
        self.__load_configuration()

        self.env = env
        self.brain_name = brain_name
        self.agent: Agent = agent

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents: int = len(env_info.agents)

    def __load_configuration(self) -> None:
        """Load the configuration from the config.ini file."""
        config: ConfigParser = ConfigParser()
        config.read(self.dir_assets / 'config.ini')

        self.episodes: int = config.getint(
            'ddpg', 'episodes', fallback=100
        )
        self.timesteps: int = config.getint(
            'ddpg', 'timesteps', fallback=100
        )

    def train(self) -> None:
        """Train the agents."""
        print(f"-> Epsiodes:\t\t\t{self.episodes}")
        print(f"-> Timesteps (per episode):\t{self.timesteps}")

        self.scores: list = []
        self.timestamp: str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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

                # save experience in replay memory
                self.agent.step(states, actions, rewards, next_states, dones)

                # update after 20 steps.
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

        # generate the plots
        self.plot_scores(self.scores)
        self.plot_scores_avg(self.scores)

        return self.scores

    def save_checkpoints(self, episode) -> None:
        """Save a checkpoint of the actor and critic."""
        if episode % 10 == 0:

            save(
                self.agent.local_actor.state_dict(),
                self.dir_checkpoint / 'actors' / f'{episode}.pth'
            )
            save(
                self.agent.local_critic.state_dict(),
                self.dir_checkpoint / 'critics' / f'{episode}.pth'
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
            self.dir_models / 'actor.pth'
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
            self.dir_models / 'critic.pth'
        )

        # for convenient method chaining
        return self

    def plot_scores(self, scores) -> None:
        """Plot the training history."""
        plt.figure()
        plt.plot(np.arange(len(scores)), scores)
        plt.title(f'{self.timestamp} | Training history')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.legend(
            [f"Reacher: {i}" for i in range(1, self.num_agents + 1)],
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        plt.savefig(
            self.dir_assets / f"scores_{self.timestamp}.png",
            format="png",
            bbox_inches='tight'
        )
        plt.savefig(
            self.dir_assets / "scores_latest.png",
            format="png",
            bbox_inches='tight'
        )

    def plot_scores_avg(self, scores, window_size: int = 100) -> None:
        """Plot the average training history.

        Args:
            scores (_type_): _description_
            window_size (int, optional): _description_. Defaults to 100.
        """
        # get the average scores
        avg_scores = np.array(scores).mean(axis=1)
        avg_scores_window = [
            np.mean(avg_scores[max(0, i - window_size): i + 1])
            for i in range(len(avg_scores))
        ]

        # plot the scores
        plt.figure()
        plt.plot(
            np.arange(1, len(scores) + 1),
            avg_scores,
            label="window = 1"
        )
        plt.plot(
            np.arange(1, len(scores) + 1),
            avg_scores_window,
            label=f"window = {window_size}"
        )
        plt.hlines(
            y=30,
            xmin=0,
            xmax=self.episodes,
            colors='red',
            linestyles='dotted',
            label='goal'
        )
        plt.title(f'{self.timestamp} | Training history | Average Score')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.legend()
        plt.savefig(
            self.dir_assets / f"scores_avg_{self.timestamp}.png",
            format="png",
            bbox_inches='tight'
        )
        plt.savefig(
            self.dir_assets / "scores_avg_latest.png",
            format="png",
            bbox_inches='tight'
        )
        plt.show()
