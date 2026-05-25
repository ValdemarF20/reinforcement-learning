# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
import matplotlib.pyplot as plt
from irlc.ex10.q_agent import QAgent
from irlc import main_plot, savepdf
from irlc.ex01.agent import train
from irlc.ex10.q_agent import cliffwalk, alpha, epsilon

class SarsaAgent(QAgent):
    r""" Implement the Sarsa control method from (SB18, Section 6.4). It is recommended you complete
    the Q-agent first because the two methods are very similar and the Q-agent is easier to implement. """
    def __init__(self, env, gamma=1, alpha=0.5, epsilon=0.1):
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi(self, s, k, info=None):
        if k == 0:
            # Beginning of episode: generate first action epsilon-greedily
            self.a = self.pi_eps(s, info)
        return self.a

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        """
        generate A' as self.a by being epsilon-greedy. Re-use code from the Agent class.
        """
        self.a = self.pi_eps(sp, info_sp)  # generate A' from S' (on-policy)
        # Sarsa update: use Q(S', A') not max_a Q(S', a)
        q_sp_ap = 0 if done else self.Q[sp, self.a]
        self.Q[s, a] += self.alpha * (r + self.gamma * q_sp_ap - self.Q[s, a])

    def __str__(self):
        return f"Sarsa{self.gamma}_{self.epsilon}_{self.alpha}"

sarsa_exp = f"experiments/cliffwalk_Sarsa"
if __name__ == "__main__":
    env, q_experiments = cliffwalk()  # get results from Q-learning
    agent = SarsaAgent(env, epsilon=epsilon, alpha=alpha)
    for _ in range(10):
        train(env, agent, sarsa_exp, num_episodes=200, max_runs=10)
    main_plot(q_experiments + [sarsa_exp], smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q and Sarsa learning on " + env.spec.name)
    savepdf("QSarsa_learning_cliff")
    plt.show()
