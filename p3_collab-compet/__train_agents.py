import numpy as np
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

# initialize the agent
multi_agent: MultiAgent = MultiAgent(
    state_size=state_size, action_size=action_size, num_agents=num_agents
)

# initialize the DDPG
ddpg: DDPG = DDPG(env, brain_name, multi_agent)

# train the agent
scores = ddpg.train()

# DONE
env.close()
