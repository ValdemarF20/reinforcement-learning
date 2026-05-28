"""
Exam 2023 Spring — Programming question solutions (Q13, Q14, Q15)

Q13: Flower-store inventory DP  (question_inventory.py)
Q14: Pendulum LQR               (question_lqr.py)
Q15: Bandit algorithms          (question_bandit.py)
"""
import numpy as np

# ---------------------------------------------------------------------------
# Q13 — Flower-store inventory DP
# ---------------------------------------------------------------------------
# The model inherits from the standard InventoryDPModel in the irlc toolbox.
# Two changes: new cost g_k(x,u,w) = c*u + |x+u-w|, and new distribution P_W.

from irlc.exam.exam2023spring.inventory import InventoryDPModel
from irlc.exam.exam2023spring.dp import DP_stochastic


class FlowerStoreModel(InventoryDPModel):
    """Flower-store variant: cost includes ordering cost c per unit and
    absolute mismatch |x + u - w|.  Demand distribution changed to
    P(w=0)=0.1, P(w=1)=0.3, P(w=2)=0.6."""

    def __init__(self, N=3, c=0.0, prob_empty=False):
        self.c = c
        self.prob_empty = prob_empty
        super().__init__(N=N)

    def g(self, x, u, w, k):
        # Stage cost: ordering cost + absolute mismatch
        if self.prob_empty:
            return 0.0  # no stage cost when maximising terminal probability
        return u * self.c + abs(x + u - w)

    def gN(self, x):
        if self.prob_empty:
            # Terminal cost trick: −1 when x_N=1, else 0  → J* = −p(x_N=1)
            return -1.0 if x == 1 else 0.0
        return 0.0

    def Pw(self, x, u, k):
        return {0: 0.1, 1: 0.3, 2: 0.6}


def a_get_policy(N: int, c: float, x0: int) -> int:
    """Return the optimal action mu*_0(x0) for the flower-store problem
    with ordering cost c per unit, planning horizon N."""
    model = FlowerStoreModel(N=N, c=c, prob_empty=False)
    J, pi = DP_stochastic(model)
    return pi[0][x0]


def b_prob_one(N: int, x0: int) -> float:
    """Return p(x_N = 1 | x_0) under the policy that maximises
    the probability of ending with exactly one item in stock."""
    model = FlowerStoreModel(N=N, prob_empty=True)
    J, pi = DP_stochastic(model)
    # J[0][x0] = -p(x_N=1) because we used g_N = -1[x_N=1]
    return -J[0][x0]


# ---------------------------------------------------------------------------
# Q14 — Pendulum LQR
# ---------------------------------------------------------------------------
from irlc.exam.exam2023spring.dlqr import LQR
from irlc.ex04.model_pendulum import PendulumModel
from irlc.ex04.discrete_control_model import DiscreteControlModel


def _get_AB_linear(a: float):
    """Return (A, B, d) for the parametric double-integrator problem:
       x_{k+1} = [[1,a],[0,1]] x_k + [[0],[1]] u_k + [[1],[0]]
    """
    A = np.array([[1, a], [0, 1]])
    B = np.array([[0], [1]])
    d = np.array([1, 0])
    return A, B, d


def a_LQR_solve(a: float, x0: np.ndarray) -> float:
    """Apply LQR (Q=R=I, N=100 steps) to the parametric linear problem
    and return the first action u_0 the controller takes at x_0."""
    A, B, d = _get_AB_linear(a)
    Q = np.eye(2)
    R = np.eye(1)
    N = 100
    (L, l), _ = LQR(A=[A]*N, B=[B]*N, d=[d]*N, Q=[Q]*N, R=[R]*N)
    u = float((L[0] @ x0 + l[0])[0])
    return u


def b_linearize(theta: float):
    """Linearize the discretized pendulum model (Delta=0.5) around
    x_bar = [theta, 0], u_bar = 0.  Return (A, B, d) as numpy arrays."""
    model = PendulumModel()
    dmodel = DiscreteControlModel(model=model, dt=0.5)
    xbar = np.array([theta, 0.0])
    ubar = np.array([0.0])
    xp = dmodel.f(xbar, ubar, k=0)
    A, B = dmodel.f_jacobian(xbar, ubar, k=0)
    d = xp - A @ xbar - B @ ubar
    return A, B, d


def c_get_optimal_linear_policy(x0: np.ndarray) -> float:
    """Linearize the pendulum around theta=0, then solve LQR (Q=R=I, N=100)
    and return the first control action as a float."""
    x0 = np.asarray(x0)
    Q = np.eye(2)
    R = np.eye(1)
    A, B, d = b_linearize(theta=0.0)
    N = 100
    (L, l), _ = LQR([A]*N, [B]*N, [d]*N, Q=[Q]*N, R=[R]*N)
    return float((L[0] @ x0 + l[0])[0])


# ---------------------------------------------------------------------------
# Q15 — Bandit algorithms
# ---------------------------------------------------------------------------

def a_select_next_action_epsilon0(k: int, actions: list, rewards: list) -> int:
    """Greedy (epsilon=0) sample-average bandit.
    Returns argmax_a Q(a) where Q(a) = mean reward for arm a so far."""
    N = {a: 0 for a in range(k)}
    S = {a: 0.0 for a in range(k)}
    for a, r in zip(actions, rewards):
        N[a] += 1
        S[a] += r
    Q = {a: S[a] / N[a] if N[a] > 0 else 0.0 for a in range(k)}
    return max(Q, key=Q.get)


def b_select_next_action(k: int, actions: list, rewards: list,
                         epsilon: float) -> int:
    """Epsilon-greedy sample-average bandit.
    With prob epsilon: random arm; else greedy arm."""
    N = {a: 0 for a in range(k)}
    S = {a: 0.0 for a in range(k)}
    for a, r in zip(actions, rewards):
        N[a] += 1
        S[a] += r
    Q = {a: S[a] / N[a] if N[a] > 0 else 0.0 for a in range(k)}
    if np.random.rand() < epsilon:
        return np.random.randint(k)
    return max(Q, key=Q.get)


def c_nonstationary_Qs(k: int, actions: list, rewards: list,
                       alpha: float) -> dict:
    """Non-stationary bandit: exponential recency-weighted average.
    Q(a) <- Q(a) + alpha * (r - Q(a))  for each (action, reward) pair.
    Returns final Q-values as a dict {arm: Q(arm)}."""
    Q = {a: 0.0 for a in range(k)}
    for a, r in zip(actions, rewards):
        Q[a] = Q[a] + alpha * (r - Q[a])
    return Q


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Q13
    print("Q13a) optimal action (c=0.5, N=3, x0=0):", a_get_policy(3, 0.5, 0), "→ expect 1")
    print("Q13b) prob(x_N=1) (N=3, x0=0):", round(b_prob_one(3, 0), 3), "→ expect 0.492")

    # Q14
    x0 = np.array([1.0, 0.0])
    print("Q14a) LQR action (a=1, x0=[1,0]):", round(a_LQR_solve(1, x0), 3), "→ expect ≈-1.666")
    A, B, d = b_linearize(np.pi/2)
    print("Q14b) d[1]:", round(float(d[1]), 2), "→ expect ≈4.91")
    print("Q14c) policy at [0.1,0]:", round(c_get_optimal_linear_policy(np.array([0.1, 0.0])), 3), "→ expect ≈-1.07")

    # Q15
    actions = [1, 0, 2, 1, 2, 4, 5, 4, 3, 2, 1, 1]
    rewards = [1, 1, 1, 0, 1, 3, 2, 0, 4, 1, 1, 2]
    k = 10
    print("Q15a) greedy action:", a_select_next_action_epsilon0(k, actions, rewards), "→ expect 3")
    Q = c_nonstationary_Qs(k, actions, rewards, alpha=0.1)
    print("Q15c) Q[2]:", round(Q[2], 3), "→ expect 0.271")
