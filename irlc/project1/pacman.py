# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.pacman.gamestate import GameState

east = """ 
%%%%%%%%
% P   .%
%%%%%%%% """ 

east2 = """
%%%%%%%%
%    P.%
%%%%%%%% """

SS2tiny = """
%%%%%%
%.P  %
% GG.%
%%%%%%
"""

SS0tiny = """
%%%%%%
%.P  %
%   .%
%%%%%%
"""

SS1tiny = """
%%%%%%
%.P  %
%  G.%
%%%%%%
"""

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

class PacmanDPModel(DPModel):
    """A DPModel for the Pacman environment — used for win probability computation."""
    def __init__(self, x0, N):
        super().__init__(N)
        self.x0 = x0
        self._state_spaces = get_future_states(x0, N)

    def S(self, k):
        return self._state_spaces[k]

    def A(self, x, k):
        return x.A()

    def f(self, x, u, w, k):
        # w encodes the next state directly
        return w

    def g(self, x, u, w, k):
        return 0.0

    def gN(self, x):
        # -1 if won, so -J_pi(x0) = win probability
        return -1.0 if x.is_won() else 0.0

    def Pw(self, x, u, k):
        return p_next(x, u)


class PacmanShortestPathDPModel(DPModel):
    """DPModel for shortest path: each step costs 1, terminal cost = 0 if won, large otherwise."""
    def __init__(self, x0, N):
        super().__init__(N)
        self.x0 = x0
        self._state_spaces = get_future_states(x0, N)

    def S(self, k):
        return self._state_spaces[k]

    def A(self, x, k):
        return x.A()

    def f(self, x, u, w, k):
        return w

    def g(self, x, u, w, k):
        return 0.0 if (x.is_won() or x.is_lost()) else 1.0

    def gN(self, x):
        return 0.0 if x.is_won() else 1e9

    def Pw(self, x, u, k):
        return p_next(x, u)


def p_next(x : GameState, u: str): 
    """ Given the agent is in GameState x and takes action u, the game will transition to a new state xp.
    The state xp will be random when there are ghosts. This function should return a dictionary of the form

    {..., xp: p, ...}

    of all possible next states xp and their probability -- you need to compute this probability.

    Hints:
        * In the above, xp should be a GameState, and p will be a float. These are generated using the functions in the GameState x.
        * Start simple (zero ghosts). Then make it work with one ghosts, and then finally with any number of ghosts.
        * Remember the ghosts move at random. I.e. if a ghost has 3 available actions, it will choose one with probability 1/3
        * The slightly tricky part is that when there are multiple ghosts, different actions by the individual ghosts may lead to the same final state
        * Check the probabilities sum to 1. This will be your main way of debugging your code and catching issues relating to the previous point.
    """
    # Start with pacman's deterministic move
    current = {x.f(u): 1.0}

    # Let each ghost take its random turn
    num_players = x.players()
    for ghost_idx in range(1, num_players):
        next_states = defaultdict(float)
        for state, prob in current.items():
            if state.is_won() or state.is_lost():
                next_states[state] += prob
            else:
                ghost_actions = state.A()
                p_ghost = 1.0 / len(ghost_actions)
                for ga in ghost_actions:
                    xp = state.f(ga)
                    next_states[xp] += prob * p_ghost
        current = dict(next_states)

    return current


def go_east(map): 
    """ Given a map-string map (see examples in the top of this file) that can be solved by only going east, this will return
    a list of states Pacman will traverse. The list it returns should therefore be of the form:

    [s0, s1, s2, ..., sn]

    where each sk is a GameState object, the first element s0 is the start-configuration (corresponding to that in the Map),
    and the last configuration sn is a won GameState obtained by going east.

    Note this function should work independently of the number of required east-actions.

    Hints:
        * Use the GymPacmanEnvironment class. The report description will contain information about how to set it up, as will pacman_demo.py
        * Use this environment to get the first GameState, then use the recommended functions to go east
    """
    env = PacmanEnvironment(layout_str=map)
    x, _ = env.reset()
    states = [x]
    while not x.is_won():
        x = x.f('East')
        states.append(x)
    return states

def get_future_states(x, N): 
    state_spaces = [[x]]
    for k in range(N):
        next_set = {}
        for state in state_spaces[k]:
            for action in state.A():
                for xp in p_next(state, action).keys():
                    if xp not in next_set:
                        next_set[xp] = xp
        state_spaces.append(list(next_set.values()))
    return state_spaces

def win_probability(map, N=10): 
    """ Assuming you get a reward of -1 on wining (and otherwise zero), the win probability is -J_pi(x_0). """
    x0, _ = PacmanEnvironment(layout_str=map).reset()
    model = PacmanDPModel(x0, N)
    J, pi = DP_stochastic(model)
    return -J[0][x0]

def shortest_path(map, N=10): 
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    x0, _ = PacmanEnvironment(layout_str=map).reset()
    model = PacmanShortestPathDPModel(x0, N)
    J, pi = DP_stochastic(model)
    # Reconstruct path by following policy
    actions = []
    states = [x0]
    x = x0
    for k in range(N):
        if x.is_won() or x.is_lost():
            break
        u = pi[k][x]
        actions.append(u)
        # Deterministic: take the action and pick the most likely next state (no ghost case),
        # or just follow the policy state transitions
        xp_dict = p_next(x, u)
        # Pick next state according to the policy (in deterministic case, there is only one)
        # Follow by picking the argmax probability state
        x = max(xp_dict, key=xp_dict.get)
        states.append(x)
    return actions, states


def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    # This function contains examples of calling your functions. However, you should use unitgrade to verify correctness.

    ## Problem 7: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    ## Problem 8: try the p_next function for a few empty environments. Does the result look sensible?
    x, _ = PacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map: 'east'")
    print(x)
    print(p_next(x, action))  # use str(state) to get a nicer representation.

    print(f"Transitions when taking action {action} in map: 'east2'")
    x, _ = PacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Problem 9
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x, _ = PacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]: # Print all elements in S_1.
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Problem 10
    N = 20  # Planning horizon
    action, states = shortest_path(east, N)
    print("east: Optimal action sequence:", action)

    action, states = shortest_path(datadiscs, N)
    print("datadiscs: Optimal action sequence:", action)

    action, states = shortest_path(SS0tiny, N)
    print("SS0tiny: Optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this tends to increase with planning depth
    wp = []
    for n in range(10):
        wp.append(win_probability(SS1tiny, N=n))
    print(wp)
    print("One ghost:", win_probability(SS1tiny, N=12))


def two_ghosts():
    # Win probability when planning using two ghosts
    print("Two ghosts:", win_probability(SS2tiny, N=12))

if __name__ == "__main__":
    no_ghosts()
    one_ghost()
    two_ghosts()
