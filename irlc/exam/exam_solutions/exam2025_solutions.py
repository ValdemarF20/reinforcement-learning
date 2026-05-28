"""
Exam 2025 Spring — Programming question solutions (Q13, Q14, Q15)

Q13: Pendulum dynamics + Euler simulation   (question_control_pendulum.py)
Q14: Cake-store inventory DP                (question_inventory_cake.py)
Q15: Q-learning implementation              (question_simple_q.py)
"""
import numpy as np

# ---------------------------------------------------------------------------
# Q13 — Pendulum dynamics + Euler simulation
# ---------------------------------------------------------------------------
# The pendulum ODE: theta_ddot = 1.25*u + 9.82*sin(theta)
# In first-order form: d/dt [theta, thetadot] = [thetadot, 1.25*u + 9.82*sin(theta)]


def a_dynamics_f(theta: float, thetadot: float, u: float) -> list:
    """Evaluate f(x, u) for the pendulum in first-order form.
    Returns [f1, f2] = [theta_dot, 1.25*u + 9.82*sin(theta)].

    Example: theta=pi/2, thetadot=0, u=1 → [0, 1.25 + 9.82 ≈ 11.07]
    """
    f1 = thetadot
    f2 = 1.25 * u + 9.82 * np.sin(theta)
    return [f1, f2]


def b_euler(theta0: float, thetadot0: float, delta: float, N: int) -> float:
    """Euler-integrate the pendulum (u=0) for N steps with step size delta.
    Returns the final angle theta_N (first coordinate of x_N).

    Example: theta0=0.1, thetadot0=0, delta=0.5, N=3 → ≈0.8353
    """
    x = np.array([theta0, thetadot0])
    for _ in range(N):
        f = np.array(a_dynamics_f(x[0], x[1], u=0.0))
        x = x + delta * f
    return float(x[0])


# ---------------------------------------------------------------------------
# Q14 — Cake-store inventory DP
# ---------------------------------------------------------------------------
from irlc.exam.exam2025spring.inventory import InventoryDPModel
from irlc.exam.exam2025spring.dp import DP_stochastic


class CakeInventoryModel(InventoryDPModel):
    """Bert's cake-store variant.
    - Demand: P(w=0)=0.1, P(w=1)=0.7, P(w=2)=0.2
    - Standard cost: g_k(x,u,w) = u + (x + u - w)^2
    - Modified cost (for parts b/c): g_k = c*u - min(w, x+u)  [profit model]
    - Terminal cost: g_N(x) = x^2
    - lazybaker=True: A_k = {2} on even steps, {0} on odd steps.
    """
    def __init__(self, N=3, cost_per_cake=1.0, lazybaker=False):
        self.lazybaker = lazybaker
        self.cost_per_cake = cost_per_cake
        super().__init__(N=N)

    def A(self, x, k):
        if self.lazybaker:
            return {2} if k % 2 == 0 else {0}
        return {0, 1, 2}

    def g(self, x, u, w, k):
        # Profit model: cost per cake baked minus cakes sold
        sold = min(w, x + u)
        return u * self.cost_per_cake - sold

    def gN(self, x):
        return float(x ** 2)


def a_expected_cost(x0: int, u0: int) -> float:
    """Compute E[g_0(x0, u0, w)] for the ORIGINAL cost g = u + (x+u-w)^2
    with P(w=0)=0.1, P(w=1)=0.7, P(w=2)=0.2.

    Note: This uses the BASE inventory cost (not the modified profit model).
    """
    # Use the base InventoryDPModel directly (which has the original cost)
    model = InventoryDPModel()
    expected_cost = sum(
        model.g(x0, u0, w, k=0) * pw
        for w, pw in model.Pw(x0, u0, k=0).items()
    )
    return expected_cost


def b_best_action(N: int, cost_per_cake: float, k: int, x: int) -> int:
    """Return the optimal action mu*_k(x) under the modified (profit) cost function."""
    model = CakeInventoryModel(N=N, cost_per_cake=cost_per_cake, lazybaker=False)
    J, pi = DP_stochastic(model)
    return pi[k][x]


def c_lazy_baker(N: int, cost_per_cake: float, x0: int) -> float:
    """Compute E[J_pi(x0)] when Bert follows the 'lazy baker' policy:
    bake 2 on even rounds, bake 0 on odd rounds."""
    model = CakeInventoryModel(N=N, cost_per_cake=cost_per_cake, lazybaker=True)
    J, pi = DP_stochastic(model)
    return J[0][x0]


# ---------------------------------------------------------------------------
# Q15 — Q-learning implementation
# ---------------------------------------------------------------------------
# Non-episodic problem with two actions {0, 1}.
# Q-values stored as dict q_values[(state, action)] = float.
# Unseen (s, a) pairs default to Q=0.


def a_greedy_policy(q_values: dict, state: int) -> int:
    """Return the greedy action a* = argmax_{a in {0,1}} Q(state, a)."""
    q0 = q_values.get((state, 0), 0.0)
    q1 = q_values.get((state, 1), 0.0)
    return 0 if q0 >= q1 else 1


def b_update_single_q(alpha: float, gamma: float, q_values: dict,
                      state: int, action: int, reward: float,
                      next_state: int) -> float:
    """Apply a single Q-learning update and return the NEW Q(state, action).
    The q_values dict is NOT modified.

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    q_sa = q_values.get((state, action), 0.0)
    max_q_next = max(q_values.get((next_state, a), 0.0) for a in [0, 1])
    td_error = reward + gamma * max_q_next - q_sa
    return q_sa + alpha * td_error


def c_update_all_q(alpha: float, gamma: float,
                   states_actions_rewards: list) -> dict:
    """Apply Q-learning to an entire trajectory and return the updated Q-table.

    Trajectory format: [(S0, A0, R1), (S1, A1, R2), ..., (S_{T-1}, A_{T-1}, R_T)]
    We process T-1 updates (we need S_{t+1} = S of the next tuple).
    All Q-values start at 0; unseen pairs are lazily initialised.
    """
    q = {}  # Q-table: (state, action) -> float

    def get_q(s, a):
        return q.get((s, a), 0.0)

    for t in range(len(states_actions_rewards) - 1):
        s, a, r = states_actions_rewards[t]
        s_next, _, _ = states_actions_rewards[t + 1]

        # Initialise unseen entries to 0
        for aa in [0, 1]:
            if (s, aa) not in q:
                q[(s, aa)] = 0.0
            if (s_next, aa) not in q:
                q[(s_next, aa)] = 0.0

        max_q_next = max(get_q(s_next, aa) for aa in [0, 1])
        td_error = r + gamma * max_q_next - get_q(s, a)
        q[(s, a)] = get_q(s, a) + alpha * td_error

    return q


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Q13
    print("Q13a) f(pi/2, 0, 1):", a_dynamics_f(np.pi/2, 0, 1), "→ expect [0, ≈11.07]")
    print("Q13b) theta_N (theta0=0.1, N=3, delta=0.5):",
          round(b_euler(0.1, 0.0, 0.5, 3), 4), "→ expect ≈0.8353")

    # Q14
    print("Q14a) expected cost (x0=0, u0=1):",
          round(a_expected_cost(0, 1), 2), "→ expect 1.3")
    print("Q14b) best action (N=3, c=0.8, k=0, x=1):",
          b_best_action(3, 0.8, 0, 1), "→ expect 1")
    print("Q14c) lazy baker cost (N=3, c=0.7, x0=0):",
          round(c_lazy_baker(3, 0.7, 0), 3), "→ expect ≈1.311")

    # Q15
    states = [0, 1, 2]
    q_example = {(s, a): s/2 + 2**a for s in states for a in [0, 1]}
    print("Q15a) greedy action at s=0:", a_greedy_policy(q_example, 0), "→ expect 1")

    alpha, gamma = 0.8, 0.9
    updated = b_update_single_q(alpha, gamma, q_example,
                                state=0, action=1, reward=0.8, next_state=2)
    print("Q15b) updated Q(0,1):", round(updated, 1), "→ expect 3.2")

    trajectory = [(0, 1, 0.5), (2, 0, -0.75), (0, 1, 0.5), (1, 0, 0.5)]
    updated_q = c_update_all_q(alpha, gamma, trajectory)
    print("Q15c) Q(0,1) after trajectory:", round(updated_q.get((0,1), 0), 2), "→ expect 0.48")
