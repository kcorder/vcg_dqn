from collections import namedtuple
import random
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from common_utils import get_eps_threshold, obs_to_tens


Transition    = namedtuple('Transition',
                           ('state', 'action', 'next_state', 'reward'))

VCGTransition = namedtuple('VCGTransition',
                           ('state', 'action', 'next_state', 'reward', 'payment'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 20):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(obs_size, hidden_size)
        self.n1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.l3 = nn.Linear(hidden_size, num_actions)
        self.activ = torch.nn.LeakyReLU()

    # Called with either one element to determine next action, or a batch
    def forward(self, x):
        # print(f'action shape {x.shape}')
        hidden = self.activ(self.n1(self.l1(x)))
        #hidden2 = self.activ(self.n2(self.l2(hidden)))
        #output = self.activ(self.l3(hidden2))
        #output = self.l3(hidden2)
        output = self.l3(hidden)

        return output


class DDQNAgent:

    def __init__(self,
                 obs_size: int,
                 n_actions: int,
                 device: str,
                 gamma: float,
                 target_update: int,
                 max_episodes: int,
                 batch_size: int,
                 lr: float,
                 hidden_size: int,
                 mem_size: int,
                 eps_start: float,
                 eps_end: float,
                 eps_decay: float):

        # TODO: refactor "policy_net" to "critic" or "Q"
        self.policy_net = DQN(obs_size, n_actions, hidden_size=hidden_size).to(device)
        self.target_net = DQN(obs_size, n_actions, hidden_size=hidden_size).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(mem_size)

        self.gamma = gamma
        self.target_update = target_update
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.device = device

        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay

        self.last_obs = None
        self.last_act = None

        # self.reset()

    def pick_action(self, obs, episode_id: int):

        sample = random.random()
        eps_threshold = get_eps_threshold(episode_id, self.max_episodes,
                                          self._eps_start, self._eps_end, self._eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                # print(f'state: {obs} {obs.shape}')
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device, dtype=torch.long)

    def reset(self):
        self.last_obs = None
        self.last_act = None

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0, {}
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 )
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
        #                               device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done = 1

        with torch.no_grad():
            target_max = self.target_net(state_batch).max(dim=1)[0]
            td_target = reward_batch + self.gamma * target_max * (1 - done)
        old_val = self.policy_net(state_batch).gather(1, action_batch.view(-1, 1)).squeeze()
        loss = F.mse_loss(td_target, old_val)
        return_loss = loss.clone().to('cpu').item()

        # # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        #
        # # Compute V(s_{t+1}) for all next states.
        # next_state_values = torch.zeros(self.batch_size, device=device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # # Compute the expected Q values
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # return_loss = loss.clone().to('cpu').item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        nn.utils.clip_grad_norm_(list(self.policy_net.parameters()), 0.5)
        self.optimizer.step()

        # return return_loss, {'next_state_values': next_state_values}
        return return_loss, {'next_state_values': target_max}

    def remember(self, action, obs, reward):
        # Store the transition in memory
        if self.last_obs is not None:
            self.memory.push(self.last_obs, action, obs, reward)

        self.last_obs = obs

    def new_episode(self, episode_id: int):
        # tracking network
        if episode_id % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # track_network_weights(self.policy_net, self.target_net, 0.001)


class VDNMixer(nn.Module):
    """
    Source: https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/vdn.py

    Paper: https://arxiv.org/pdf/1706.05296.pdf
    """

    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=2, keepdim=True)


class QMixer(nn.Module):
    """
    Source of inspiration: https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py

    Paper: https://arxiv.org/pdf/1803.11485.pdf
    """

    def __init__(self,
                 n_agents: int,
                 obs_size: int,
                 embed_dim: int = 10):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = obs_size * n_agents  # expects every agent has the same obs size

        # hyper layers
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """Do the following:
        -output of QNets and their states
        -produce weights and biases for the mixing network
        -mix Q values into one global Q value
        -output global Q value
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        # TODO not finished!!


class MixedAgent:
    """
    QMIX or VDN mixing used (compromises between IQL and global QL)
    """

    def __init__(self,
                 obs_size: int,
                 num_actions: int,
                 n_agents: int,
                 embed_dim: int = 10,
                 hidden_size: int = 30,
                 use_qmix: bool = False,
                 device = 'cpu'):

        self.n_agents = n_agents
        self.device = device

        if use_qmix:
            # mixing network that takes state and agent q_value outputs and produces..
            self.mixer = QMixer(n_agents, obs_size, embed_dim)

            # see https://github.com/oxwhirl/pymarl/blob/73960e11c5a72e7f9c492d36dbfde02016fde05a/src/learners/q_learner.py#L22
            self.target_mixer = copy.deepcopy(self.mixer)

            print(f'WARNING: not finished')
        else:
            self.mixer = VDNMixer()

        # make the QNetwork per agent
        self.q_nets = [DQN(obs_size, num_actions, hidden_size) for _ in range(self.num_agents)]

    def forward(self, obs):
        """

        TODO optimize model and forward

        Get list of arrays with observation
        """
        assert len(obs) == self.n_agents
        obs_t = obs_to_tens(obs, self.device)

        q_values = []
        for agent_obs, q_net in zip(obs, self.q_nets):
            q_vals = q_net(agent_obs)
            q_values.append(q_vals)

        # should produce [BS=1, obs_size, n_agents]
        observations = torch.stack(obs_t)
        q_values_stacked = torch.stack(q_values)

        mixed_values = self.mixer.forward(q_values_stacked, observations)
        # TODO not tested!!
        return mixed_values

    # TODO add optimize model etc..

    def _pick_agent_action(self, sample, threshold, agent_q_vals):
        if sample > threshold:
            with torch.no_grad():
                return agent_q_vals.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]],
                                device=self.device, dtype=torch.long)

    def pick_action(self, obs, episode_id: int):

        # get list of tensors per q_net
        q_vals = []
        obs_tens = obs_to_tens(obs)
        for q_net, agent_obs in zip(self.q_nets, obs_tens):
            q_agent = q_net.forward(agent_obs)
            q_vals.append(q_agent)

        # sample random values per agent
        samples = [random.random() for _ in range(self.n_agents)]
        eps_threshold = get_eps_threshold(episode_id, self.max_episodes)

        # either arg_max or random uniform per agent
        actions = []
        for sample, agent_q_vals in zip(samples, q_vals):
            actions.append(self._pick_agent_action(sample, eps_threshold, agent_q_vals))

        return actions


class VCG(nn.Module):
    """VCG mechanism by Kevin"""

    def __init__(self, agents: list):
        nn.Module.__init__(self)
        self.agents = agents

        ### use MLP for learned h function? Currently using Clarke pivot rule
        # self.MLP = nn.Sequential(OrderedDict([
        #     ('linear1', nn.Linear(10,10)),
        #     ('relu1', nn.ReLU()),
        #     ('linear2', nn.Linear(10,10)),
        #     ('relu2', nn.ReLU()),
        #     ('linear3', nn.Linear(10, 10)),
        # ]))

    def forward(self, q_values_per_agent: torch.tensor):
        """
        Args:
            q_values_per_agent: (num_agents x num_actions) tensor of q_values for current state

        second price auction -> give to highest bidder, charge 2nd highest bid
        The "payments" here correspond to
        """
        q_values_per_agent = q_values_per_agent.squeeze()
        num_agents = q_values_per_agent.shape[0]
        assert num_agents == len(self.agents)

        report_vector = torch.argmax(q_values_per_agent, dim=0)  # x^opt
        externality_payments = self.externality(q_values_per_agent)

        payments = []
        for i, agent in enumerate(self.agents):
            pay_i = (report_vector.sum() - report_vector[i]).float()
            pay_i += externality_payments[i]
            payments.append(pay_i)

        # goes through the argmax per agent

        # print(f"q_values_per_agent = {q_values_per_agent}")
        # print(f"payments = {payments}")

        return torch.stack(payments)  # (num_agents)

    def externality(self, q_values_per_agent: torch.tensor):
        """Clarke pivot rule implementation for the heuristic payment function h

        See: https://en.wikipedia.org/wiki/Vickrey%E2%80%93Clarke%E2%80%93Groves_mechanism#The_Clarke_pivot_rule
        """
        total_action_qvalues = q_values_per_agent.squeeze().sum(dim=0)
        payments = []
        for i, agent in enumerate(self.agents):
            total_other_vals = torch.cat((total_action_qvalues[:i],
                                          total_action_qvalues[i + 1:]))
            payments.append(-1 * total_other_vals.max())

        return payments


class VCGReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = VCGTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class VCG_DDQNAgent:
    """must include the redistributed rewards or Q values in the
    replay buffer for the optimization step
    """

    def __init__(self,
                 agent_index: int,
                 obs_size: int,
                 n_actions: int,
                 device: str,
                 gamma: float,
                 target_update: int,
                 max_episodes: int,
                 batch_size: int,
                 lr: float,
                 hidden_size: int,
                 mem_size: int,
                 eps_start: float,
                 eps_end: float,
                 eps_decay: float):

        self.agent_index = agent_index

        # TODO: refactor "policy_net" to "critic" or "Q"
        self.policy_net = DQN(obs_size, n_actions, hidden_size=hidden_size).to(device)
        self.target_net = DQN(obs_size, n_actions, hidden_size=hidden_size).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = VCGReplayMemory(mem_size)

        self.gamma = gamma
        self.target_update = target_update
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.device = device

        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay

        # TODO: estimates payments from state and q_values
        self.payment_critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.reset()

    def pick_action(self, obs, episode_id: int):

        sample = random.random()
        eps_threshold = get_eps_threshold(episode_id, self.max_episodes,
                                          self._eps_start, self._eps_end, self._eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                # print(f'state: {obs} {obs.shape}')
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device, dtype=torch.long)

    def q_values(self, obs) -> torch.tensor:
        with torch.no_grad():
            return self.policy_net(obs)

    def pick_action_from_Q(self, q_values: torch.tensor, episode_id: int):
        """same as above `pick_action` but broken into choosing from Q(.,vect{a})"""
        sample = random.random()
        eps_threshold = get_eps_threshold(episode_id, self.max_episodes,
                                          self._eps_start, self._eps_end, self._eps_decay)

        if sample > eps_threshold:
            return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device, dtype=torch.long)

    def reset(self):
        self.last_obs = None

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0, {}
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 )
        batch = VCGTransition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        payment_batch = torch.stack(batch.payment, dim=0)
        done = 1

        with torch.no_grad():
            target_max = self.target_net(state_batch).max(dim=1)[0]
            target_with_payment = target_max + payment_batch
            td_target = reward_batch + self.gamma * target_with_payment * (1 - done)
            # td_target = reward_batch + self.gamma * target_max * (1 - done)
        old_val = self.policy_net(state_batch).gather(1, action_batch.view(-1, 1)).squeeze()
        loss = F.mse_loss(td_target, old_val)
        return_loss = loss.clone().to('cpu').item()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.policy_net.parameters()), 0.5)
        self.optimizer.step()

        return return_loss, {'next_state_values': target_max}

    def remember(self, action, obs, reward, payment):
        # Store the transition in memory
        if self.last_obs is not None:
            self.memory.push(self.last_obs, action, obs, reward, payment)

        self.last_obs = obs

    def new_episode(self, episode_id: int):
        # tracking network
        if episode_id % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # track_network_weights(self.policy_net, self.target_net, 0.001)
