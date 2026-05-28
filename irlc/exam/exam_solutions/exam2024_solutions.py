"""
Exam 2024 Spring — Programming question solutions (Q13, Q14, Q15)

Q13: Bridal gown inventory DP       (question_inventory.py)
Q14: Bill's apartment MDP           (question_bill_mdp.py)
Q15: Continuous control / RK4       (question_control.py)
"""
import numpy as np
import math

# ---------------------------------------------------------------------------
# Q13 — Bridal gown inventory DP
# ---------------------------------------------------------------------------
# Base model: m actions {0,...,m-1}, uniform demand w ~ Uniform{0,...,m-1},
# same cost/dynamics as standard inventory (see exam2024spring folder).

from irlc.exam.exam2024spring.inventory import InventoryDPModel
from irlc.exam.exam2024spring.dp import DP_stochastic


class BridalGownModel(InventoryDPModel):
    """Bridal gown store variant.
    - m actions: A_k(x) = {0, 1, ..., m-1}  (plus optional 'sale' action)
    - Demand: p_W(w=i) = 1/m for i=0,...,m-1
    - Optional 'sale' action: clears inventory (x'=0), cost = 3/4*(m-w)
    """
    action_sale = "sale"

    def __init__(self, N=3, m=3, allow_sale=False):
        self.m = m
        self.allow_sale = allow_sale
        super().__init__(N=N)

    def A(self, x, k):
        space = list(range(self.m))
        if self.allow_sale:
            space = space + [self.action_sale]
        return space

    def g(self, x, u, w, k):
        if u == self.action_sale:
            return 3/4 * (self.m - w)  # reduced-price sale revenue
        return InventoryDPModel.g(self, x, u, w, k)

    def f(self, x, u, w, k):
        if u == self.action_sale:
            return 0  # sale clears inventory
        return InventoryDPModel.f(self, x, u, w, k)

    def Pw(self, x, u, k):
        # Uniform demand on {0, ..., m-1}
        return {w: 1/self.m for w in range(self.m)}


def a_get_cost(N: int, m: int, x0: int) -> float:
    """Optimal expected cost J*(x0) for the bridal gown problem (no sale option)."""
    model = BridalGownModel(N=N, m=m, allow_sale=False)
    J, pi = DP_stochastic(model)
    return J[0][x0]


def b_sale(N: int, m: int, x0: int) -> float:
    """Optimal expected cost J*(x0) when the sale action is also available."""
    model = BridalGownModel(N=N, m=m, allow_sale=True)
    J, pi = DP_stochastic(model)
    return J[0][x0]


# ---------------------------------------------------------------------------
# Q14 — Bill's apartment MDP
# ---------------------------------------------------------------------------
from irlc.exam.exam2024spring.mdp import MDP
from irlc.exam.exam2024spring.policy_evaluation import policy_evaluation
from irlc.exam.exam2024spring.value_iteration import value_iteration


class BillMDP(MDP):
    """
    States: 0 (no apartment), 1 (has apartment).
    Actions at s=1: 0 = AirBnB (safe), 1 = gamble (risky).
    Actions at s=0: 0 only (no choices).
    Never terminates (is_terminal always False).
    """
    def __init__(self, r_airbnb=0.01):
        self.p_win = 0.45
        self.r_airbnb = r_airbnb
        super().__init__(initial_state=1)

    def is_terminal(self, state):
        return False

    def A(self, s):
        if s == 0:
            return [0]
        return [0, 1]  # 0=AirBnB, 1=gamble

    def Psr(self, s, a):
        if s == 0:
            return {(0, 0): 1.0}
        if s == 1 and a == 0:  # AirBnB: stay, earn r_airbnb
            return {(1, self.r_airbnb): 1.0}
        if s == 1 and a == 1:  # Gamble: win (p_win, R=2) or lose (1-p_win, R=0 + apartment gone)
            return {(1, 2): self.p_win,
                    (0, 0): 1 - self.p_win}


def a_always_airbnb(r_airbnb: float, gamma: float) -> float:
    """Value v_pi(s0) when Bill always AirBnBs.
    Closed-form: r_airbnb / (1 - gamma)  (geometric series, state never changes)."""
    mdp = BillMDP(r_airbnb=r_airbnb)
    pi = {0: {0: 1.0},
          1: {0: 1.0, 1: 0.0}}  # always action 0 (AirBnB)
    J = policy_evaluation(pi=pi, mdp=mdp, gamma=gamma)
    return J[1]


def b_random_decisions(r_airbnb: float, gamma: float) -> float:
    """Value v_pi(s0) when Bill uses 50/50 random policy at s=1."""
    mdp = BillMDP(r_airbnb=r_airbnb)
    pi = {0: {0: 1.0},
          1: {0: 0.5, 1: 0.5}}
    J = policy_evaluation(pi=pi, mdp=mdp, gamma=gamma)
    return J[1]


def c_is_it_better_to_gamble(r_airbnb: float, gamma: float) -> bool:
    """Returns True if the optimal policy chooses to gamble (a=1) at s=1."""
    mdp = BillMDP(r_airbnb=r_airbnb)
    pi, V = value_iteration(mdp, gamma)
    return pi[1] == 1


# ---------------------------------------------------------------------------
# Q15 — Continuous control / RK4 simulation
# ---------------------------------------------------------------------------
# Dynamics: x_dot = f(x, u) = -exp(u - x^2).  One-dimensional x, u in R.

import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost


class ExponentialControl(ControlModel):
    """1-D system: dx/dt = -exp(u - x^2)."""
    def sym_f(self, x, u, t=None):
        return [-sym.exp(u[0] - x[0]**2)]

    def get_cost(self):
        return SymbolicQRCost(Q=np.eye(1), R=np.eye(1))


def a_xdot(x: float, a: float) -> float:
    """Evaluate x_dot = f(x, u) for closed-loop policy u(t) = a*x^2."""
    u = a * x**2
    m = ExponentialControl()
    return m.f((x,), (u,), 0)[0]


def b_rk4_simulate(u0: float, tF: float) -> float:
    """Simulate the system from x(0)=0 with constant policy u(t)=u0
    until time tF using RK4.  Return the final state x(tF)."""
    m = ExponentialControl()
    xs, us, ts, _ = m.simulate((0.0,), u_fun=(u0,), t0=0, tF=tF)
    return float(xs[-1][0])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Q13
    print("Q13a) expected cost (N=6, m=4, x0=0):",
          round(a_get_cost(6, 4, 0), 2), "→ expect 13.75")
    print("Q13b) expected cost with sale (N=6, m=4, x0=0):",
          round(b_sale(6, 4, 0), 2), "→ expect ≈11.25")

    # Q14
    print("Q14a) always_airbnb (r=0.01, gamma=0.99):",
          round(a_always_airbnb(0.01, 0.99), 1), "→ expect ≈1")
    print("Q14b) random_decisions (r=0.01, gamma=0.99):",
          round(b_random_decisions(0.01, 0.99), 3), "→ expect ≈1.612")
    print("Q14c) better to gamble? (r=0.02, gamma=0.99):",
          c_is_it_better_to_gamble(0.02, 0.99), "→ expect False")
    print("Q14c) better to gamble? (r=0.01, gamma=0.99):",
          c_is_it_better_to_gamble(0.01, 0.99), "→ expect True")

    # Q15
    print("Q15a) xdot(x=2, a=1):", round(a_xdot(2, 1), 2), "→ expect -1")
    print("Q15b) x(tF=3) with u0=2:", round(b_rk4_simulate(2, 3), 2), "→ expect ≈-2.09")
