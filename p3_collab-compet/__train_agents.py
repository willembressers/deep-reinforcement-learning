import configparser
import pathlib

from src.ddpg import DDPG
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

# load the configuration from the config.ini file
config = configparser.ConfigParser()
config.read(
    pathlib.Path(__file__).parents[1] / "p3_collab-compet" / "assets" / "config.ini"
)

# initialize the multi agent
multi_agent = MultiAgent(config, state_size, action_size, num_agents)

# initialize the trainer
ddpg = DDPG(config, num_agents, multi_agent)

# start training
ddpg.train(env, brain_name)

# DONE
env.close()
