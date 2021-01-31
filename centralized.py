import time
import math
import random
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# local imports
from models import DDQNAgent, get_eps_threshold, VCG_DDQNAgent, VCG
from common_utils import obs_to_tens, plot_ipd_rewards, plot_rewards
from envs import get_env

print(f'done imports')



### Set up IPD vars ###########################################################
"""
### Run IQL with VCG redistribution 

Requires minor code change: 
each agent returns $Q(\ ., \bf{a})$ instead of immediately returning the action.

This is needed for the VCG mechanism that sort of acts like a Mixer. 

Then the `pick_action_from_Q(.)` function returns the action, just as `pick_action(.)` did before. 

Notes: 
- asdf
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is: {device}')

env = get_env('ipd')
n_actions = 2
# n_local_actions = 2
# n_actions = np.prod([space.n for space in env.action_space])
obs_size = 1

BATCH_SIZE=128
PLT_PERIOD = 100  # faster than 1
MAX_EPISODES = 13000
MEM_SIZE=256 # 300
OPTIM_PERIOD=1
GAMMA = 0.999
TARGET_UPDATE = 10 # in epochs

EPS_START = 0.9
EPS_END = 0.01
DECAY = 1/6
LR = 0.01

HS=50

agents = [VCG_DDQNAgent(agent_index,
                        obs_size,
                        n_actions,
                        device,
                        batch_size=BATCH_SIZE,
                        gamma=GAMMA,
                        target_update=TARGET_UPDATE,
                        max_episodes=MAX_EPISODES,
                        lr=LR,
                        hidden_size=HS,
                        mem_size=MEM_SIZE,
                        eps_start=EPS_START,
                        eps_end=EPS_END,
                        eps_decay=DECAY)
          for agent_index in range(2)]

# Commented out IPython magic to ensure Python compatibility.
# %pdb on

from matplotlib import rc
rc('animation', html='jshtml')

steps_done = 0
all_rewards = []
all_eps_thresholds = []
all_losses = []

vcg_mech = VCG(agents)

for i_episode in tqdm(range(MAX_EPISODES)):

    obs = obs_to_tens(env.reset(), device)

    [agent.reset() for agent in agents]
    for agent, init_obs in zip(agents, obs):  # first state
        agent.last_obs = init_obs

    ep_losses = []
    ep_rewards = []
    ep_q_values = []

    for t in count():

        q_values_per_agent = []
        actions = []

        # how do we do centralized Q's without approximate policies / communication?
        # last_actions = [agent.last_act if agent.last_act is not None else 0.0 for agent in agents]
        for agent, a_obs in zip(agents, obs):

            q_values = agent.q_values(a_obs)
            q_values_per_agent.append(q_values)

            act = agent.pick_action_from_Q(q_values, i_episode)
            agent.last_act = act
            actions.append(act)

        # currently not used
        q_values_per_agent = torch.stack(q_values_per_agent)
        payments = vcg_mech(q_values_per_agent)

        # TODO: what invariants should I expect??
        # sum(q_values_per_agent) = -1 * payments
        # assert q_values_per_agent.sum().isclose(-1 * payments.sum())
        # just plain IQL

        obs, rewards, done, _ = env.step(actions)
        obs = obs_to_tens(obs, device)
        ep_rewards.append(rewards)

        tens_rewards = [torch.tensor([reward], device=device) for reward in rewards]

        # Store the transition in memory
        for agent, action, ob, reward, payment in zip(agents, actions, obs, tens_rewards, payments):
            agent.remember(action, ob, reward, payment)

        # Perform one step of the optimization (on the target network)
        if t % OPTIM_PERIOD == 0:
            losses, infos = [], []
            for agent in agents:
                result = agent.optimize_model()
                losses.append(result[0])
                infos.append(result[1])
            ep_losses.append(losses)
            ep_q_values.append([torch.mean(info.get('next_state_values', torch.zeros(1)))
                                for info in infos])

        if done:
            # logging
            all_eps_thresholds.append(get_eps_threshold(i_episode, MAX_EPISODES, EPS_START, EPS_END, DECAY))

            all_rewards.append(np.array(ep_rewards).mean(0))  # mean reward from episode per agent
            all_losses.append(np.array(ep_losses).mean(0))

            if i_episode % PLT_PERIOD == 0:
                pass
                plot_ipd_rewards(all_rewards, all_eps_thresholds, all_losses, scale=1, hide_r=False, win=100,
                                 q_values=ep_q_values)
            break

    # Update the target network, copying all weights and biases in DQN
    agent.new_episode(i_episode)

    # if i_episode % 200 == 0:
    #    torch.save(policy_net.state_dict(), f'data/actions/model_{i_episode}.chk')

print('Training done')

plt.savefig('centralized.pdf',bbox_inches='tight') 






if __name__ == '__main__':
    x='debug'
