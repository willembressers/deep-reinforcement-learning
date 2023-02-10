"""Train the multi agent."""
import configparser
import pathlib

from numpy import ndarray, zeros
from src.multi_agent import MultiAgent
from tqdm import tqdm


class DDPG:
    """Train the multi agent using an Deep Determenistic Policy Gradient."""

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

        # load the configuration from the config.ini file
        self.config = configparser.ConfigParser()
        self.config.read(pathlib.Path(__file__).parents[1] / "assets" / "config.ini")

    def train(self) -> None:
        """Train the agents."""
        # get the training configuration.
        episodes: int = self.config.getint("ddpg", "episodes", fallback=100)
        timesteps: int = self.config.getint("ddpg", "timesteps", fallback=100)

        # setup a progress bar (with tqdm)
        progress_bar = tqdm(range(1, episodes + 1), desc="Training multi agent")

        # loop over the episodes
        for episode in progress_bar:

            # reset the environment
            env_info = self.env.reset(train_mode=True)[self.brain_name]

            # get the current state
            state = env_info.vector_observations

            # reset agent
            self.multi_agent.reset()

            # loop over all time steps
            for timestep in range(timesteps):

                # select an action (for each agent)
                actions: ndarray = self.multi_agent.act(state, add_noise=True)

                # send all actions to the environment
                env_info = self.env.step(actions)[self.brain_name]

                # get next states, rewards and done flags (for each agent)
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

            # update and save
            if episode % 20 == 0:
                self.multi_agent.save(episode)
