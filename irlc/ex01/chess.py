# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""This file contains code for the Chess Tournament problem."""
import numpy as np
from gymnasium.spaces.discrete import Discrete
from gymnasium import Env

class ChessTournament(Env):
    """The ChessTournament gymnasium-environment which simulate a chess tournament.

    In the problem, a chess tournament ends when a player wins two games in a row. The results
    of each game are -1, 0, 1 corresponding to a loss, draw and win for player 1. See:
    https://www.youtube.com/watch?v=5UQU1oBpAic

    To implement this, we define the step-function such that one episode of the environment corresponds to playing
    a chess tournament to completion. Once the environment completes, it returns a reward of +1 if the player won
    the tournament, and otherwise 0.

    Each step therefore corresponds to playing a single game in the tournament.
    To implement this, we use a state corresponding to the sequence of games in the tournament:

    >>> self.s = [0, -1, 1, 0, 0, 1]

    In the self.step(action)-function, we ignore the action, simulate the outcome of a single game,
    and append the outcome to self.s. We then compute whether the tournament has completed, and if so
    a reward of 1 if we won.
    """

    def __init__(self, p_draw=3 / 4, p_win=2 / 3):
        self.action_space = Discrete(1)
        self.p_draw = p_draw
        self.p_win = p_win
        self.s = []  # A chess tournament is a sequence of won/lost games s = [0, -1, 1, 0, ...]

    def reset(self):
        """Reset the tournament environment to begin to simulate a new tournament."""
        self.s = []
        return self.s, {}

    def step(self, action):
        """Play a single game in the current tournament"""
        game_outcome = None  # should be -1, 0, or 1 depending on outcome of single game.
        if np.random.rand() < self.p_draw:
            game_outcome = 0
        else:
            if np.random.rand() < self.p_win:
                game_outcome = 1
            else:
                game_outcome = -1
        self.s.append(game_outcome)

        done = len(self.s) >= 2 and self.s[-1] == self.s[-2] and self.s[-1] != 0
        r = self.s[-1] == 1 if done else 0
        return self.s, r, done, False, {}

def main():
    """The main method of the chess-game problem."""
    T = 5000
    from irlc import train, Agent
    env = ChessTournament()
    stats, _ = train(env, Agent(env), num_episodes=T)
    p_win = np.mean([st['Accumulated Reward'] for st in stats])
    avg_length = np.mean([st['Length'] for st in stats])

    print("Agent: Estimated chance I won the tournament: ", p_win)
    print("Agent: Average tournament length", avg_length)


if __name__ == "__main__":
    main()
