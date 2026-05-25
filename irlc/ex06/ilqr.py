# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [Her25] Tue Herlau. Sequential decision making. (Freely available online), 2025.
  [TET12] Yuval Tassa, Tom Erez, and Emanuel Todorov. Synthesis and stabilization of complex behaviors through online trajectory optimization. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 4906–4913. IEEE, 2012. (See tassa2012.pdf).
  [Har20] James Harrison. Optimal and learning-based control combined course notes. (See AA203combined.pdf), 2020.
"""
r"""
This implements two methods: The basic ILQR method, described in (Her25, Algorithm 24), and the linesearch-based method
described in (Her25, Algorithm 25).

If you are interested, you can consult (TET12) (which contains generalization to DDP) and (Har20, Alg 1).
"""
import warnings
import numpy as np
from irlc.ex05.dlqr import LQR
from irlc.ex04.discrete_control_model import DiscreteControlModel

def ilqr_basic(model : DiscreteControlModel, N, x0, us_init : list = None, n_iterations=500, verbose=True):
    r"""
    Implements the basic ilqr algorithm, i.e. (Her25, Algorithm 24). Our notation (x_bar, etc.) will be consistent with the lecture slides
    """
    mu, alpha = 1, 1 # Hyperparameters. For now, just let them have defaults and don't change them
    # Create a random initial state-sequence
    n, m = model.state_size, model.action_size
    u_bar = [np.random.uniform(-1, 1,(model.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * N
    """
    Initialize nominal trajectory xs, us using us and x0 (i.e. simulate system from x0 using action sequence us).
    The simplest way to do this is to call forward_pass with all-zero sequence of control vector/matrix l, L.
    """
    L0 = [np.zeros((m, n))] * N
    l0 = [np.zeros(m)] * N
    x_bar, u_bar = forward_pass(model, x_bar, u_bar, L=L0, l=l0, alpha=0)
    J_hist = []
    for i in range(n_iterations):
        """
        Compute derivatives around trajectory and cost estimate J of trajectory. To do so, use the get_derivatives
        function. Remember the functions will return lists of derivatives.
        """
        A, B, c, c_x, c_u, c_xx, c_ux, c_uu = get_derivatives(model, x_bar, u_bar)
        J = cost_of_trajectory(model, x_bar, u_bar)
        """  Backward pass: Obtain feedback law matrices l, L using the backward_pass function.
        """
        L, l = backward_pass(A, B, c_x, c_u, c_xx, c_ux, c_uu, mu=mu)
        """ Forward pass: Given L, l matrices computed above, simulate new (optimal) action sequence.
        In the lecture slides, this is similar to how we compute u^*_k and x_k
        Once they are computed, iterate the iLQR algorithm by setting x_bar, u_bar equal to these values
        """
        x_bar, u_bar = forward_pass(model, x_bar, u_bar, L=L, l=l, alpha=alpha)
        if verbose:
            print(f"{i}> J={J:4g}, change in cost since last iteration {0 if i == 0 else J-J_hist[-1]:4g}")
        J_hist.append(J)
    return x_bar, u_bar, J_hist, L, l

def ilqr_linesearch(model : DiscreteControlModel, N, x0, n_iterations, us_init=None, tol=1e-6, verbose=True):
    r"""
    For linesearch implement method described in (Her25, Algorithm 25) (we will use regular iLQR, not DDP!)
    """
    # The range of alpha-values to try out in the linesearch
    # plus parameters relevant for regularization scheduling.
    alphas = 1.1 ** (-np.arange(10) ** 2)  # alphas = [1, 1.1^{-2}, ...]
    mu_min = 1e-6
    mu_max = 1e10
    Delta_0 = 2
    mu = 1.0
    Delta = Delta_0

    n, m = model.state_size, model.action_size
    u_bar = [np.random.uniform(-1, 1, (model.action_size,)) for _ in range(N)] if us_init is None else us_init
    x_bar = [x0] + [np.zeros(n, )] * (N)
    # Initialize nominal trajectory xs, us (same as in basic ilqr)
    L0 = [np.zeros((m, n))] * N
    l0 = [np.zeros(m)] * N
    x_bar, u_bar = forward_pass(model, x_bar, u_bar, L=L0, l=l0, alpha=0)
    J_hist = []

    converged = False
    for i in range(n_iterations):
        alpha_was_accepted = False
        """ Step 1: Compute derivatives around trajectory and cost estimate of trajectory.
        In our implementation, J_bar = J_{u^star}(x_0) """
        A, B, c, c_x, c_u, c_xx, c_ux, c_uu = get_derivatives(model, x_bar, u_bar)
        J_prime = cost_of_trajectory(model, x_bar, u_bar)
        try:
            """
            Step 2: Backward pass to obtain control law (l, L).
            """
            L, l = backward_pass(A, B, c_x, c_u, c_xx, c_ux, c_uu, mu=mu)
            """
            Step 3: Forward pass and alpha scheduling.
            Decrease alpha and check condition J_new < J_prime. Apply regularization scheduling as needed.
            """
            for alpha in alphas:
                x_hat, u_hat = forward_pass(model, x_bar, u_bar, L=L, l=l, alpha=alpha)
                J_new = cost_of_trajectory(model, x_hat, u_hat)

                if J_new < J_prime:
                    """ Linesearch proposed trajectory accepted! Set current trajectory equal to x_hat, u_hat. """
                    if np.abs((J_prime - J_new) / J_prime) < tol:
                        converged = True  # Method does not seem to decrease J; converged. Break and return.

                    J_prime = J_new
                    x_bar, u_bar = x_hat, u_hat
                    # Decrease regularization on accepted step
                    Delta = max(1 / Delta_0, Delta / Delta_0)
                    mu = max(mu_min, mu * Delta)
                    alpha_was_accepted = True  # accept this alpha
                    break
        except np.linalg.LinAlgError as e:
            # Matrix in dlqr was not positive-definite and this diverged
            warnings.warn(str(e))

        if not alpha_was_accepted:
            # No alphas accepted: increase regularization
            Delta = max(Delta_0, Delta * Delta_0)
            mu = min(mu_max, mu * Delta) if mu_max else mu * Delta

            if mu_max and mu >= mu_max:
                raise Exception("Exceeded max regularization term; we are stuffed.")

        dJ = 0 if i == 0 else J_prime-J_hist[-1]
        info = "converged" if converged else ("accepted" if alpha_was_accepted else "failed")
        if verbose:
            print(f"{i}> J={J_prime:4g}, decrease in cost {dJ:4g} ({info}).\nx[N]={x_bar[-1].round(2)}")
        J_hist.append(J_prime)
        if converged:
            break
    return x_bar, u_bar, J_hist, L, l

def backward_pass(A : list, B : list, c_x : list, c_u : list, c_xx : list, c_ux : list, c_uu : list, mu=1):
    r"""Given all derivatives, apply the LQR algorithm to get the control law.

    The input arguments are described in the online documentation and the lecture notes. You should use them to call your
    implementation of the :func:`~irlc.ex06.dlqr.LQR` method. Note that you should give a value of all inputs except for the ``d``-term.

    :param A: linearization of the dynamics matrices :math:`A_k`.
    :param B:  linearization of the dynamics matrices :math:`B_k`.
    :param c_x: Cost terms corresponding to :math:`\mathbf{q}_k`
    :param c_u: Cost terms corresponding to :math:`\mathbf{r}_k`
    :param c_xx: Cost terms corresponding to :math:`Q_k`
    :param c_ux: Cost terms corresponding to :math:`H_k`
    :param c_uu: Cost terms corresponding to :math:`R_k`
    :param mu: Regularization parameter for the LQR method
    :return: The control law :math:`L_k, \mathbf{l}_k` as two lists.
    """
    Q, QN = c_xx[:-1], c_xx[-1]
    R = c_uu[:-1]
    H = c_ux[:-1]
    q = c_x[:-1]
    qN = c_x[-1]
    r = c_u[:-1]
    # Define the inputs using the linearization inputs.
    (L, l), (V, v, vc) = LQR(A=A, B=B, R=R, Q=Q, QN=QN, H=H, q=q, qN=qN, r=r, mu=mu)
    return L, l

def cost_of_trajectory(model : DiscreteControlModel, xs : list, us : list) -> float:
    r"""Helper function which computes the cost of the trajectory.

    The cost is defined as:

    .. math::

        c_N( \bar {\mathbf x}_N, \bar {\mathbf u}_) + \sum_{k=0}^{N-1} c_k(\bar {\mathbf x}_k, \bar {\mathbf u}_k)

    and to compute it, you should use the two helper methods ``model.cost.c`` and ``model.cost.cN``
    (see :func:`~irlc.ex04.discrete_control_cost.DiscreteQRCost.c` and :func:`~irlc.ex04.discrete_control_cost.DiscreteQRCost.cN`).

    :param model: The control model used to compute the cost.
    :param xs: A list of length :math:`N+1` of the form :math:`\begin{bmatrix}\bar {\mathbf x}_0 & \dots & \bar {\mathbf x}_N \end{bmatrix}`
    :param us: A list of length :math:`N` of the form :math:`\begin{bmatrix}\bar {\mathbf x}_0 & \dots & \bar {\mathbf x}_{N-1} \end{bmatrix}`
    :return: The cost as a number.
    """
    N = len(us)
    JN = model.cost.cN(xs[-1])
    return sum(map(lambda args:  model.cost.c(*args), zip(xs[:-1], us, range(N)))) + JN

def get_derivatives(model : DiscreteControlModel, x_bar : list, u_bar : list):
    """Compute all the derivatives used in the model.

    The return type should match the meaning in (Her25, Subequation 17.8) and in the online documentation.

    - ``c`` should be a list of length :math:`N+1`
    - ``c_x`` should be a list of length :math:`N+1`
    - ``c_xx`` should be a list of length :math:`N+1`
    - ``c_u`` should be a list of length :math:`N`
    - ``c_uu`` should be a list of length :math:`N`
    - ``c_ux`` should be a list of length :math:`N`
    - ``A`` should be a list of length :math:`N`
    - ``B`` should be a list of length :math:`N`

    Use the model to compute these terms. For instance, this will compute the first terms ``A[0]`` and ``B[0]``::

        A0, B0 = model.f_jacobian(x_bar[0], u_bar[0], 0)

    Meanwhile, to compute the first terms of the cost-functions you should use::

        c[0], c_x[0], c_u[0], c_xx[0], c_ux[0], c_uu[0] = model.cost.c(x_bar[0], u_bar[0], k=0, compute_gradients=True)

    :param model: The model to use when computing the derivatives of the cost
    :param x_bar: The nominal state-trajectory
    :param u_bar: The nominal action-trajectory
    :return: Lists of all derivatives computed around the nominal trajectory (see the lecture notes).
    """
    N = len(u_bar)
    """ Compute A_k, B_k (lists of matrices of length N) as the jacobians of the dynamics. To do so, 
    recall from the online documentation that: 
        x, f_x, f_u = model.f(x, u, k, compute_jacobian=True)
    """
    A = [None]*N
    B = [None]*N
    c = [None] * (N+1)
    c_x = [None] * (N + 1)
    c_xx = [None] * (N + 1)

    c_u = [None] * (N+1)
    c_ux = [None] * (N + 1)
    c_uu = [None] * (N + 1)
    # Now update each entry correctly (i.e., ensure there are no None elements left).
    for i in range(N):
        A[i], B[i] = model.f_jacobian(x_bar[i], u_bar[i], i)
    for i in range(N):
        c[i], c_x[i], c_u[i], c_xx[i], c_ux[i], c_uu[i] = model.cost.c(x_bar[i], u_bar[i], k=i, compute_gradients=True)
    # Concatenate the derivatives associated with the last time point N.
    cN, c_xN, c_xxN = model.cost.cN(x_bar[N], compute_gradients=True)
    c[N] = cN
    c_x[N] = c_xN
    c_xx[N] = c_xxN
    return A, B, c, c_x, c_u, c_xx, c_ux, c_uu

def forward_pass(model : DiscreteControlModel, x_bar : list, u_bar : list, L : list, l : list, alpha=1.0):
    r"""Simulates the effect of the controller on the model

    We assume the system starts in :math:`\mathbf{x}_0 = \bar {\mathbf x}_0`, and then simulate the effect of
    generating actions using the closed-loop policy

    .. math::

        \mathbf{u}_k = \bar {\mathbf u}_k + \alpha \mathbf{l}_k + L_k (\mathbf{x}_k - \bar { \mathbf x}_k)

    (see  (Her25, eq. (17.16))).

    :param model: The model used to compute the dynamics.
    :param x_bar: A nominal list of states
    :param u_bar: A nominal list of actions (not used by the method)
    :param L: A list of control matrices :math:`L_k`
    :param l: A list of control vectors :math:`\mathbf{l}_k`
    :param alpha: The linesearch parameter.
    :return: A list of length :math:`N+1` of simulated states and a list of length :math:`N` of simulated actions.
    """
    N = len(u_bar)
    x = [None] * (N+1)
    u_star = [None] * N
    x[0] = x_bar[0].copy()

    for i in range(N):
        # u_k = u_bar_k + alpha * l_k + L_k (x_k - x_bar_k)
        u_star[i] = u_bar[i] + alpha * l[i] + L[i] @ (x[i] - x_bar[i])
        x[i + 1] = model.f(x[i], u_star[i], i)
    return x, u_star
