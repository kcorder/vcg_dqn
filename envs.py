import numpy as np
import gym

def get_env(env_name):
    if env_name == 'ipd':
        return IteratedPrisonersDilemmaEnv()
    elif env_name == 'global_ipd':
        return GlobalIteratedPrisonersDilemmaEnv()

    return gym.make(env_name).unwrapped


class IteratedPrisonersDilemmaEnv(gym.Env):
    """
    action_space: The Space object corresponding to valid actions
    observation_space: The Space object corresponding to valid observations
    reward_range: A tuple corresponding to the min and max possible rewards
    """
    def __init__(self):
        super(IteratedPrisonersDilemmaEnv, self).__init__()
        self.num_agents = 2
        self.action_space = [gym.spaces.Discrete(2),
                             gym.spaces.Discrete(2)]
        self.observation_space = [gym.spaces.Box(shape=(1,), low=0.0, high=1.0),
                                  gym.spaces.Box(shape=(1,) ,low=0.0, high=1.0)]
        # action 0: cooperate, action 1: defect

    def step(self, actions):
        """get agent rewards for cooperate/defect, always returns done"""
        p1_action = actions[0]
        p2_action = actions[1]

        if p1_action == p2_action == 0:
            reward1 = -1
            reward2 = -1
        elif p1_action == p2_action == 1:
            reward1 = -2
            reward2 = -2
        elif p1_action == 0 and p2_action == 1:
            reward1 = -3
            reward2 = 0
        elif p1_action == 1 and p2_action == 0:
            reward1 = 0
            reward2 = -3
        else:
            raise ValueError('something bad happened')

        return [0.0, 0.0], [reward1, reward2], True, {}

    def reset(self):
        return [0.0, 0.0]

class GlobalIteratedPrisonersDilemmaEnv(IteratedPrisonersDilemmaEnv):
    """IPD solved by a global policy"""

    def __init__(self):
        super(GlobalIteratedPrisonersDilemmaEnv, self).__init__()
        self.num_agents = 2
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(shape=(2,), low=0.0, high=1.0)

    def global_to_local(self, action):
        if action == 0:
            return [0, 0] # cooperate, cooperate
        elif action == 1:
            return [1, 0] # defect, cooperate
        elif action == 2:
            return [0, 1] # cooperate, defect
        elif action == 3:
            return [1, 1] # defect, defect
        else:
            raise ValueError('invalid action')

    def step(self, actions):
        local_actions = self.global_to_local(actions)

        observations, rewards, done, info = super().step(local_actions)
        return np.array(observations), sum(rewards), done, info

    def reset(self):
        return np.array(super().reset())



class LeversEnv(gym.Env):

    def __init__(self, n_levers, n_agents, distinct_lever=None):
        super(LeversEnv, self).__init__()

        self.n_agents = n_agents
        self.n_levers = n_levers

        if distinct_lever is None:
            self.distinct_lever = np.random.randint(0, n_levers)
        else:
            self.distinct_lever = distinct_lever

        self._levers = [0] * n_levers  # _levers[i] := num agents that chose lever i
        self._lever_rewards = [1.] * n_levers  # _lever_rewards[i] := reward if all choose lever i
        self._lever_rewards[self.distinct_lever] = 0.9


    def step(self, joint_action):
        for action in joint_action:
            self._levers[action] += 1
        if np.max(self._levers) == np.sum(self._levers): # all picked same lever
            chosen_lever = int(np.argmax(self._levers))
            reward = self._lever_rewards[chosen_lever]
        else: # chose different levers
            reward = 0.

        return [1]*self.n_agents, [reward]*self.n_agents, [True]*self.n_agents, [{}]*self.n_agents

    def reset(self):
        self._levers = [0] * self.n_levers
        return 0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError

        out_string = ''
        for i, num in enumerate(self._levers):
            out_string = out_string + f"[{num}] "

        print(out_string)
        return out_string

