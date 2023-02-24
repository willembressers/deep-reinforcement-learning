"""Train the multi agent."""
import configparser
import datetime
import pathlib

from matplotlib.pyplot import (
    figure,
    hlines,
    legend,
    plot,
    savefig,
    title,
    xlabel,
    ylabel,
)
from numpy import arange, array, mean, ndarray, zeros
from src.multi_agent import MultiAgent
from tqdm import tqdm


class DDPG:
    """Train the multi agent using an Deep Determenistic Policy Gradient."""

    scores: list = []

    def __init__(self, env, brain_name, multi_agent: MultiAgent) -> None:
        """Initialize the Deep Determenistic Policy Gradient class.

        Args:
            env (_type_): _description_
            brain_name (_type_): _description_
            multi_agent (MultiAgent): _description_
        """
        self.env = env
        self.brain_name = brain_name
        self.multi_agent: MultiAgent = multi_agent
        self.num_agents: int = multi_agent.num_agents

        # load the configuration from the config.ini file
        self.dir_assets = pathlib.Path(__file__).parents[1] / "assets"
        config = configparser.ConfigParser()
        config.read(self.dir_assets / "config.ini")

        # get the training configuration.
        self.episodes: int = config.getint("ddpg", "episodes", fallback=100)
        self.timesteps: int = config.getint("ddpg", "timesteps", fallback=100)

    def train(self) -> None:
        """Train the agents."""
        # (re)set the variables
        self.scores: list = []
        self.timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # setup a progress bar (with tqdm)
        progress_bar = tqdm(range(1, self.episodes + 1), desc="Training multi agent")

        # loop over the episodes
        for episode in progress_bar:

            # reset the environment
            env_info = self.env.reset(train_mode=True)[self.brain_name]

            # get the current states
            states = env_info.vector_observations

            # reset agent
            self.multi_agent.reset()

            # initialize the score (for each agent)
            scores: ndarray = zeros(self.num_agents)

            # loop over all time steps
            for timestep in range(self.timesteps):

                # select an action (for each agent)
                actions: ndarray = self.multi_agent.act(states, add_noise=True)

                # send all actions to the environment
                env_info = self.env.step(actions)[self.brain_name]

                # get next states, rewards and done flags (for each agent)
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # save experience in replay memory
                self.multi_agent.step(states, actions, rewards, next_states, dones)

                # set the next states as the current states
                states = next_states

                # accumulate rewards
                scores += rewards

                # exit loop if episode finished
                if any(dones):
                    break

            # learn the agents
            self.multi_agent.learn()

            # save most recent scores
            self.scores.append(scores)

            # save (every 20th episode)
            if episode % 20 == 0:
                self.multi_agent.save(episode)

        # generate the plots
        self.plot_scores(self.scores)
        self.plot_scores_avg(self.scores)

    def plot_scores(self, scores) -> None:
        """Plot the training history."""
        figure()
        plot(arange(len(scores)), scores)
        title(f"{self.timestamp} | Training history")
        ylabel("Score")
        xlabel("Episode")
        legend(
            [f"Agent: {i}" for i in range(1, self.num_agents + 1)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        savefig(
            self.dir_assets / "interim" / f"scores_{self.timestamp}.png",
            format="png",
            bbox_inches="tight",
        )
        savefig(
            self.dir_assets / "scores_latest.png", format="png", bbox_inches="tight"
        )

    def plot_scores_avg(self, scores, window_size: int = 100) -> None:
        """Plot the average training history.

        Args:
            scores (_type_): _description_
            window_size (int, optional): _description_. Defaults to 100.
        """
        # get the average scores
        avg_scores = array(scores).mean(axis=1)
        avg_scores_window = [
            mean(avg_scores[max(0, i - window_size) : i + 1])
            for i in range(len(avg_scores))
        ]

        # plot the scores
        figure()
        plot(arange(1, len(scores) + 1), avg_scores, label="window = 1")
        plot(
            arange(1, len(scores) + 1),
            avg_scores_window,
            label=f"window = {window_size}",
        )
        hlines(
            y=0.5,
            xmin=0,
            xmax=self.episodes,
            colors="red",
            linestyles="dotted",
            label="goal",
        )
        title(f"{self.timestamp} | Training history | Average Score")
        ylabel("Score")
        xlabel("Episode")
        legend()
        savefig(
            self.dir_assets / "interim" / f"scores_avg_{self.timestamp}.png",
            format="png",
            bbox_inches="tight",
        )
        savefig(
            self.dir_assets / "scores_avg_latest.png", format="png", bbox_inches="tight"
        )
