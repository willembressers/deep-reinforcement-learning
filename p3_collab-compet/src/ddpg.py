# python core modules
import datetime
import pathlib
from collections import deque

# 3rd party modules
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# custom modules
from src.multi_agent import MultiAgent
from tqdm import tqdm


class DDPG:
    def __init__(self, config, multi_agent, num_agents, dir_root):
        """Initialize the Deep Deterministic Policy Gradient.

        Args:
            config (_type_): _description_
            state_size (_type_): _description_
            action_size (_type_): _description_
            num_agents (_type_): _description_
        """

        # get the parameters
        self.episodes = config.getint("ddpg", "episodes", fallback=1000)
        self.target_score = config.getfloat("ddpg", "target_score", fallback=0.5)
        self.target_window = config.getint("ddpg", "target_window", fallback=100)

        self.multi_agent = multi_agent

        # set the class variables
        self.num_agents = num_agents
        self.dir_root = dir_root

    def train(self, env, brain_name):
        """Train the multi-agent on the given environment.

        Args:
            env (_type_): _description_
            brain_name (_type_): _description_
        """

        # get the current timestamp
        self.timestamp = datetime.datetime.now()

        # get the agent indexes (just for plotting)
        index_agents = [f"agent-{i}" for i in range(self.num_agents)]

        target_episode = None
        scores_history = []

        # setup a progress bar (with tqdm)
        progress_bar = tqdm(range(1, self.episodes + 1), desc="Training multi agent")

        # loop over the episodes
        for episode in progress_bar:

            # reset the environment
            env_info = env.reset(train_mode=True)[brain_name]

            # get the current state (for each agent)
            state = env_info.vector_observations

            # reset the multi_agent
            self.multi_agent.reset()

            # initialize the score (for each agent)
            scores = np.zeros(self.num_agents)

            # loop until done
            while True:

                # let the multi_agent decide which actions to take
                action = self.multi_agent.act(state, episode, add_noise=True)

                # send all actions to tne environment
                env_info = env.step(action)[brain_name]

                # get next states, rewards and done flags (for each agent)
                next_state = env_info.vector_observations
                reward = env_info.rewards
                done = env_info.local_done

                # save experience in replay memory
                self.multi_agent.step(episode, state, action, reward, next_state, done)

                # set the next states as the current state
                state = next_state

                # accumulate rewards
                scores += reward

                # exit loop if episode finished
                if np.any(done):
                    break

            # append the scores as a dict to the history
            scores_history.append(
                {index: score for index, score in zip(index_agents, scores.tolist())}
            )

            # describe the avg score in the progress bar
            progress_bar.set_description(f"Avg score: {np.mean(scores):.5f}")

            # when target adchieved >>> secure the model and generate a plot
            if target_episode == None and np.mean(scores) >= self.target_score:
                target_episode = episode
                self.save_models(episode)
                df = self.save_scores(scores_history)
                self.plot(df, episode)

            # every 100th episode >>>  secure the model and generate a plot
            if episode % 100 == 0:
                self.save_models(episode)
                df = self.save_scores(scores_history)
                self.plot(df, episode)

    def plot(self, df, episode):
        """Generate a plotly (interactive) html file.

        Args:
            df (_type_): _description_
            episode (_type_): _description_
            target_episode (_type_): _description_
        """
        # generate the plot
        fig = px.line(
            df,
            x="episode",
            y="score",
            color="name",
            title=f"Training history | {self.timestamp.strftime('%Y-%m-%d | %H:%M:%S')} | Episode: {episode}",
        )

        # add the target (horizontal) line
        fig.add_hline(
            y=self.target_score, line_width=1, line_dash="dash", line_color="red"
        )

        # add the target (vertical) episode line
        target_episode = df.loc[
            (df["name"] == "rolling-mean 100") & (df["score"] >= 0.5), "episode"
        ].min()
        if target_episode > 0:
            fig.add_vline(
                x=target_episode, line_width=2, line_dash="dash", line_color="blue"
            )

        # save plots (html)
        fig.write_html(
            self.dir_root
            / "assets"
            / "interim"
            / f"scores_{self.timestamp.strftime('%Y-%m-%d-%H-%M-%S')}.html"
        )
        fig.write_html(self.dir_root / "assets" / "scores.html")
        fig.write_image(self.dir_root / "assets" / "scores.png")

    def save_models(self, episode=0):
        """Save the multi-agent (agents).

        Args:
            episode (int, optional): _description_. Defaults to 0.
        """
        # ensure the directory exists
        interim_path = self.dir_root / "checkpoints" / "interim" / f"episode_{episode}"
        interim_path.mkdir(parents=True, exist_ok=True)

        # save the interim models
        self.multi_agent.save(interim_path)

        # save the latest models
        self.multi_agent.save(self.dir_root / "checkpoints")

    def save_scores(self, scores_history):
        """Create and save a scores_history dataframe.

        Args:
            scores_history (_type_): _description_

        Returns:
            _type_: _description_
        """
        # construct a dataframe of all scores
        df = pd.DataFrame(scores_history)

        # add the mean & rolling_mean columns
        df["mean"] = df.mean(axis=1)
        df[f"rolling-mean {self.target_window}"] = (
            df["mean"].rolling(self.target_window).mean()
        )

        # reshape the dataframe
        df = (
            df.stack()
            .reset_index()
            .rename(columns={"level_0": "episode", "level_1": "name", 0: "score"})
        )

        # backup the data
        df.to_csv(
            self.dir_root
            / "data"
            / "interim"
            / f"scores_{self.timestamp.strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            index=False,
        )
        df.to_csv(self.dir_root / "data" / "scores.csv", index=False)

        return df
