# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from collections import defaultdict
from irlc.ex10.q_agent import QAgent
from irlc.gridworld.gridworld_environments import GridworldEnvironment, grid_bridge_grid
from irlc import train
from irlc.ex08.rl_agent import TabularQ

# A simple UCB action-selection problem (basic problem)
very_basic_grid = [['#',1, '#'],
                    [1, 'S', 2],
                    ['#',1, '#']]


class UCBQAgent(QAgent):
    def __init__(self, env, gamma=1.0, alpha=0.5, c=1.0):
        self.c = c
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=0)
        self.N = defaultdict(int)
        self.t = 0

    def pi(self, s, k, info=None):
        actions, Qs = self.Q.get_Qs(s, info)
        n_total = sum(self.N[(s, a)] for a in actions)
        if n_total == 0:
            ucb = list(Qs)
        else:
            ucb = [float('inf') if self.N[(s, a)] == 0
                   else Q + self.c * np.sqrt(np.log(n_total) / self.N[(s, a)])
                   for a, Q in zip(actions, Qs)]
        return actions[int(np.argmax(ucb))]

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        self.N[(s, a)] += 1
        self.t += 1
        max_q_sp = 0 if done else self.Q[sp, self.Q.get_optimal_action(sp, info_sp)]
        self.Q[s, a] += self.alpha * (r + self.gamma * max_q_sp - self.Q[s, a])


def get_ucb_actions(layout : list, alpha : float, c : float, episodes : int, plot=False) -> list:
    """ Return the sequence of actions the agent tries in the environment with the given layout-string when trained over 'episodes' episodes.
    To create an environment, you can use the line:

    > env = GridworldEnvironment(layout)

    See also the demo-file.

    The 'plot'-parameter is optional; you can use it to add visualization using a line such as:

    if plot:
        env = GridworldEnvironment(layout, render_mode='human')

    Or you can just ignore it. Make sure to return the truncated action list (see the rebels_demo.py-file or project description).
    In other words, the return value should be a long list of integers corresponding to actions:
    actions = [0, 1, 2, ..., 1, 3, 2, 1, 0, ...]
    """
    if plot:
        env = GridworldEnvironment(layout, render_mode='human')
    else:
        env = GridworldEnvironment(layout)
    agent = UCBQAgent(env, alpha=alpha, c=c)
    stats, trajectories = train(env, agent, num_episodes=episodes, return_trajectory=True, verbose=False)
    actions = [a for traj in trajectories for a in traj.action[:-1]]
    return actions

if __name__ == "__main__":
    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=4, plot=False)
    print("Number of actions taken", len(actions))
    print("List of actions taken over 4 episodes", actions)

    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=8, plot=False)
    print("Number of actions taken", len(actions))
    print("Actions taken over 8 episodes", actions)

    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=9, plot=False)
    print("Number of actions taken", len(actions))
    print("Actions taken over 9 episodes", actions) # In this particular case, you can also predict the 9th action. Why?

    # Simulate 100 episodes. This should solve the problem.
    actions = get_ucb_actions(very_basic_grid, alpha=0.1, c=5, episodes=100, plot=False)
    print("Basic: Actions taken over 100 episodes", actions)

    # Simulate 100 episodes for the bridge-environment. The UCB-based method should solve the environment without being overly sensitive to c.
    # You can compare your result with the Q-learning agent in the demo, which performs horribly.
    actions = get_ucb_actions(grid_bridge_grid, alpha=0.1, c=5, episodes=300, plot=False)
    print("Bridge: Actions taken over 300 episodes. The agent should solve the environment:", actions)
