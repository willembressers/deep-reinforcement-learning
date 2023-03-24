import configparser
import os
import pathlib

import numpy as np
from src.multi_agent import MultiAgent
from unityagents import UnityEnvironment

# load the environment
env: UnityEnvironment = UnityEnvironment(file_name="p3_collab-compet/Tennis.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# get the path to the current directory
dir_root = pathlib.Path(os.path.abspath(".")) / "p3_collab-compet"

# load the configuration from the config.ini file
config = configparser.ConfigParser()
config.read(dir_root / "assets" / "config.ini")

# initialize the agent
multi_agent: MultiAgent = MultiAgent(
    config=config, state_size=state_size, action_size=action_size, num_agents=num_agents
)

# Load the trained actor netwerk into it.
multi_agent.load(dir_root)

# play the game
for i in range(1, 6):  # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        actions = multi_agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
    print("Score (max over agents) from episode {}: {}".format(i, np.max(scores)))

# DONE
env.close()
