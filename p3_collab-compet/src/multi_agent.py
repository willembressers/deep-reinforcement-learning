# 3rd party modules
import numpy as np
import torch
# custom modules
from src.agent import Agent
from src.replay_buffer import ReplayBuffer

UPDATE_FREQ = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent:
    def __init__(self, config, state_size, action_size, num_agents):
        """Initialize the multi-agent (and create agents).

        Args:
            config (_type_): _description_
            state_size (_type_): _description_
            action_size (_type_): _description_
            num_agents (_type_): _description_
        """

        # get the parameters
        self.batch_size = config.getint("default", "batch_size", fallback=256)
        self.learn_episode = config.getint("default", "learn_episode", fallback=100)
        self.tau = config.getfloat("multi_agent", "tau", fallback=0.001)
        self.repeat = config.getint("multi_agent", "repeat", fallback=100)

        # creating agents and store them into agents list
        self.agents = [
            Agent(config, state_size, action_size, num_agents)
            for i in range(num_agents)
        ]

        # Replay memory
        self.memory = ReplayBuffer(config)

    def reset(self):
        """Reset the agents."""
        for agent in self.agents:
            agent.reset()

    def act(self, state, episode=0, add_noise=True):
        """Let the agents decide which action to take.

        Args:
            state (_type_): _description_
            episode (_type_): _description_
            add_noise (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # collect the chosen actions here
        actions = []

        # loop over the agents
        for agent_state, agent in zip(state, self.agents):

            # let the agent pick an action
            action = agent.act(agent_state, episode, add_noise)
            action = np.reshape(action, newshape=(-1))

            # append it to the list
            actions.append(action)

        return np.stack(actions)

    def step(self, episode, state, action, reward, next_state, done):
        """Update the agents with the new information / situation.

        Args:
            episode (_type_): _description_
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        # reshape the states
        full_state = np.reshape(state, newshape=(-1))
        next_full_state = np.reshape(next_state, newshape=(-1))

        # store the new information / situation in memory
        self.memory.add(
            state, full_state, action, reward, next_state, next_full_state, done
        )

        # if there is sufficient memory
        if len(self.memory) > self.batch_size and episode > self.learn_episode:
            for _ in range(self.repeat):

                # let the agents learn
                for agent in self.agents:
                    self.learn(agent)

                # update the agents
                for agent in self.agents:
                    agent.soft_update(agent.actor.local, agent.actor.target, self.tau)
                    agent.soft_update(agent.critic.local, agent.critic.target, self.tau)

    def learn(self, agent):
        """Let the given agent some time to learn.

        Args:
            agent (_type_): _description_
        """
        # grab a random batch of experiences
        (
            states,
            full_states,
            actions,
            rewards,
            next_states,
            next_full_states,
            dones,
        ) = self.memory.sample()

        # initialize (to zero) the actions of the target network
        actions_target = torch.zeros(actions.shape, dtype=torch.float, device=device)

        # let all target networks choose an action (given the current state)
        for index, agent_i in enumerate(self.agents):

            # get the id of the current agent
            if agent == agent_i:
                id = index

            # forward pass to take some action
            actions_target[:, index, :] = agent_i.actor.target(states[:, index])

        # get the (state, action, reward, done) of the current agent
        state = states[:, id, :]
        action = actions[:, id, :]
        reward = rewards[:, id].view(-1, 1)
        done = dones[:, id].view(-1, 1)

        # replace action of the specific agent with actor_local actions
        actions_local = actions.clone()
        actions_local[:, id, :] = agent.actor.local(state)

        # flatt actions
        actions = actions.view(self.batch_size, -1)
        actions_target = actions_target.view(self.batch_size, -1)
        actions_local = actions_local.view(self.batch_size, -1)

        agent_experience = (
            full_states,
            actions,
            actions_local,
            actions_target,
            state,
            action,
            reward,
            done,
            next_states,
            next_full_states,
        )

        # pass it onn
        agent.learn(agent_experience)

    def save(self, path):
        """Save the actor (and critic) networks on a specific location.

        Args:
            path (_type_): _description_
        """

        # loop over the agents
        for idx, agent in enumerate(self.agents):

            # save the actor
            torch.save(agent.actor.local.state_dict(), path / f"actor_{idx}.pth")

            # save the critic (no for, need when playing)
            torch.save(agent.critic.local.state_dict(), path / f"critic_{idx}.pth")

    def load(self, dir_root):
        """Load the actor networks for all agents.

        Args:
            dir_root (_type_): _description_
        """

        # loop over the agents
        for idx, agent in enumerate(self.agents):

            # load the pytorch model
            state_dict = torch.load(dir_root / "checkpoints" / f"actor_{idx}.pth")

            # assign the weights
            agent.actor.local.load_state_dict(state_dict)
