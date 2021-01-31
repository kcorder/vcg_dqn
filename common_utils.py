import math
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display


def get_eps_threshold(episode: int, max_episodes: int,
                      EPS_START, EPS_END, DECAY) -> float:
    """episode -1 means minimal exploration!"""
    assert EPS_START > EPS_END

    if episode < 0:
        eps_threshold = EPS_END
    else:
        position = (episode + 1) / max_episodes

        eps = math.exp(-1. * position / DECAY)
        eps_threshold = eps * (EPS_START - EPS_END) + EPS_END

    return eps_threshold


def obs_to_tens(obs, device):
    """Get list of observations, return list of tensors"""
    results = []

    for ob in obs:
        if not isinstance(ob, (np.ndarray, np.generic)):
            ob = np.array(ob)
        o = torch.from_numpy(ob).unsqueeze(0).float().to(device)
        results.append(o.view(1, -1))

    return results

def track_network_weights(source, target, tau: float):
    if tau == 0:
        return
    assert 0 < tau <= 1

    for target_param, local_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def ma(data, N: int):
    """Expects data of shape [time, num_agnets], returns moving average over the time"""

    if data.shape[0] <= N:
        return None

    result = []
    for dim in range(data.shape[1]):
        rr = data[:, dim]
        convolved = np.convolve(rr, np.ones((N,)) / N, mode='valid')
        res = np.append(np.zeros(N - 1), convolved)
        result.append(res)

    total_res = np.array(result)
    return total_res.T


def plot_ma(data, name_base: str, win: int, hide_data: bool = False, scale: float = 1):
    """Expects data of shape [time, num_agents] or [time] and plots lines and their averages"""
    names = []
    ALP = 0.15

    # to 2D numpy
    data = np.asarray(data)
    if len(data.shape) == 1:
        data = data.reshape(data.size, 1)

    data = data * scale
    p = None

    # original line of requested
    if not hide_data:
        p = plt.plot(data, alpha=ALP)
        names = [f'{name_base}_{aid}' for aid in list(range(data.shape[1]))]
        if not isinstance(p, List):
            p = [p]

    # moving average if possible
    avg_data = ma(data, win)
    if avg_data is not None:
        for series_id in range(avg_data.shape[1]):
            # preserve color if possible
            if p is not None:
                plt.plot(avg_data[:, series_id], color=p[series_id].get_color())
            else:
                plt.plot(avg_data[:, series_id])
        names = names + [f'avg_{name_base}_{aid}' for aid in list(range(data.shape[1]))]

    return names


def plot_rewards(episode_durations,
                 rewards,
                 eps_thresholds,
                 losses,
                 scale=100,
                 l_scale=10,
                 ylim=None,
                 win=100):
    ALP = 0.15
    plt.figure(2)
    plt.clf()
    names = []
    # if is_ipython:
    display.clear_output(wait=True)
    # display.display(plt.gcf())

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('durations/rewards/eps')

    # episode durations
    durations = np.array(episode_durations).reshape(len(episode_durations), 1)
    plt.plot(durations, alpha=ALP)
    names.append('ep. duration')

    # rewards
    names = names + plot_ma(rewards, 'reward', win)

    # exploration
    plt.plot(np.asarray([ee * scale for ee in eps_thresholds]))
    names.append(f'eps. threshold * {scale}')

    # losses
    names = names + plot_ma(losses, f'loss*{l_scale}', win, scale=l_scale)

    if ylim is not None:
        plt.ylim(top=ylim)
    ax = plt.gca()
    ax.legend(names)

    plt.pause(0.0001)  # pause a bit so that plots are updated


def plot_ipd_rewards(rewards,
                     eps_thresholds,
                     losses,
                     scale=10,
                     ylim=None,
                     win=20,
                     hide_r=False,
                     q_values=None):
    """Plot reards, exploration and losses per agent for IPD and IQL"""
    plt.figure(2)
    plt.clf()
    names = []
    display.clear_output(wait=True)

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('durations/rewards/eps')

    # rewards
    names = names + plot_ma(rewards, 'reward', win, hide_r)

    # exploration
    plt.plot(np.asarray([ee * scale for ee in eps_thresholds]))
    names.append(f'eps * {scale}')

    # losses
    names = names + plot_ma(losses, 'loss', win)

    # q_values
    if q_values:
        names = names + plot_ma(q_values, 'q_values', win)

    if ylim is not None:
        plt.ylim(top=ylim)
    ax = plt.gca()
    ax.legend(names)

    plt.pause(0.0001)  # pause a bit so that plots are updated