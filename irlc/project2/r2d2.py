# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import time
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from gymnasium.spaces import Box
# matplotlib.use('Qt5Agg') This line may be useful if you are having matplotlib problems on Linux.
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from irlc.ex05.dlqr import LQR
from irlc.ex06.linearization_agent import LinearizationAgent
from irlc.project2.utils import R2D2Viewer
from irlc import Agent, train, plot_trajectory, savepdf

dt = 0.05 # Time discretization Delta
Tmax = 5  # Total simulation time (in all instances). This means that N = Tmax/dt = 100.
x22 = (2, 2, np.pi / 2)  # Where we want to drive to: x_target

class R2D2Model(ControlModel): # This may help you get started.
    state_labels = ["$x$", "$y$", r"$\gamma$"]
    action_labels = ["Cart velocity $v$", r'Yaw rate $\omega$'] # Define constants as needed here (look at other environments); Note there is an easy way to add labels!

    def __init__(self, x_target=(2,2,np.pi/2) ): # This constructor is one possible choice.
        # x_target: The state we will drive towards.
        self.x_target = np.asarray(x_target)
        self.Tmax = 5  # Plan for a maximum of 5 seconds.
        # Set up a variable for rendering (optional) and call superclass.
        self.viewer = None
        super().__init__()

    def get_cost(self) -> SymbolicQRCost:
        # The cost function uses the target x^*.
        cost = SymbolicQRCost(Q=np.zeros(3), R=np.eye(2))
        cost += cost.goal_seeking_cost(x_target=self.x_target)
        return cost

    def x0_bound(self) -> Box:
        return Box(0, 0, shape=(self.state_size,))

    def sym_f(self, x, u, t=None):
        return [u[0] * sym.cos(x[2]),
                u[0] * sym.sin(x[2]),
                u[1]]

    """The following two methods allows the environment to be rendered as in:
    
    > env = R2D2Environment(render_mode='human') 
    
    You can otherwise ignore them.
    """
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def render(self, x, render_mode="human"): 
        if self.viewer is None:
            self.viewer = R2D2Viewer(x_target=self.x_target) # Target is the red cross.
        self.viewer.update(x)
        time.sleep(0.05)
        return self.viewer.blit(render_mode=render_mode) 


class R2D2Environment(ControlEnvironment):
    def __init__(self, Tmax=Tmax, x_target=x22, dt=None, render_mode=None):
        assert dt is not None, "Remember to specify the discretization time!"
        model = R2D2Model(x_target=x_target) # Create an R2D2 ControlModel with the given parameters.
        dmodel = DiscreteControlModel(model, dt=dt)   # Create a discrete version of the R2D2 ControlModel
        super().__init__(dmodel, Tmax=Tmax, render_mode=render_mode)

def f_euler(x : np.ndarray, u : np.ndarray, Delta=0.05) -> np.ndarray:
    """ Solve Problem 9. The function should compute
    > x_next = f_k(x, u)
    """
    x_next = x + Delta * np.array([u[0] * np.cos(x[2]),
                                    u[0] * np.sin(x[2]),
                                    u[1]])
    return x_next

def linearize(x_bar, u_bar, Delta=0.05):
    """ Linearize R2D2's dynamics around the two vectors x_bar, u_bar
    and return A, B, d so that

    x_{k+1} = A x_k + B u_k + d (approximately).

    The function should return linearization matrices A, B and d.
    """
    A = np.eye(3)
    A[0, 2] = -Delta * u_bar[0] * np.sin(x_bar[2])
    A[1, 2] =  Delta * u_bar[0] * np.cos(x_bar[2])
    B = np.array([[Delta * np.cos(x_bar[2]), 0],
                  [Delta * np.sin(x_bar[2]), 0],
                  [0,                        Delta]])
    d = f_euler(x_bar, u_bar, Delta) - A @ x_bar - B @ u_bar
    return A, B, d

def drive_to_linearization(x_target, plot=True):
    """
    Plan in a R2D2 model with specific value of x_target (in the cost function).

    this function will linearize the dynamics around xbar=0, ubar=0 to get a linear approximation of the model,
    and then use that to plan on a horizon of N=50 steps to get a control law (L_0, l_0). This is then applied
    to generate actions.

    Plot is an optional parameter to control plotting. the plot_trajectory(trajectory, env) method may be useful.

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
        * The control method is identical to one we have seen in the exercises/notes. You can re-purpose the code from that week.
    """
    env = R2D2Environment(x_target=x_target, dt=dt)
    xbar = np.zeros(env.discrete_model.state_size)
    ubar = np.zeros(env.discrete_model.action_size)
    agent = LinearizationAgent(env, model=env.discrete_model, xbar=xbar, ubar=ubar)
    _, traj = train(env, agent, num_episodes=1, return_trajectory=True)
    if plot:
        plot_trajectory(traj[0], env)
    return traj[0].state


def drive_to_mpc(x_target, plot=True) -> np.ndarray:
    """
    Plan in a R2D2 model with specific value of x_target (in the cost function) using iterative MPC (see problem text).

    Plot is an optional parameter to control plotting. the plot_trajectory(trajectory, env) method may be useful.

    The function should return the states visited as a (samples x state-dimensions) matrix, i.e. same format
    as the default output of trajectories when you use train(...).

    Hints:
     * The control method is *nearly* identical to the linearization control method. Think about the differences,
       and how a solution to one can be used in another.
     * A bit more specific: Linearization is handled similarly to the LinearizationAgent, however, we need to update
       (in each step) the xbar/ubar states/actions we are linearizing about, and then just use the immediate action computed
       by the linearization agent.
     * My approach was to implement a variant of the LinearizationAgent.
    """
    env = R2D2Environment(x_target=x_target, dt=dt)
    model = env.discrete_model
    N = 50

    class IterativeMPCAgent(Agent):
        def __init__(self_agent, env):
            self_agent.last_u = np.zeros(model.action_size)
            super().__init__(env)

        def pi(self_agent, x, k, info=None):
            ubar = self_agent.last_u
            Jx, Ju = model.f_jacobian(x, ubar)
            d_vec = np.asarray(model.f(x, ubar)).flatten() - Jx @ x - Ju @ ubar
            cost = model.cost
            (L, l), _ = LQR(A=[Jx]*N, B=[Ju]*N, d=[d_vec]*N,
                             Q=[cost.Q]*N, R=[cost.R]*N, q=[cost.q]*N,
                             QN=cost.QN, qN=cost.qN)
            u = L[0] @ x + l[0]
            self_agent.last_u = u
            return u

    agent = IterativeMPCAgent(env)
    _, traj = train(env, agent, num_episodes=1, return_trajectory=True)
    if plot:
        plot_trajectory(traj[0], env)
    return traj[0].state

if __name__ == "__main__":
    r2d2 = R2D2Model()
    print(r2d2) # This will print out details of your R2D2 model.

    # Problem 9: Euler discretization
    print("=" * 50)
    print("Problem 9: Euler discretization f_euler")
    x = np.asarray([0, 0, 0])
    u = np.asarray([1, 0])
    print(f"  x_k = {x}, u_k = {u}")
    print(f"  x_{{k+1}} = f_euler(x, u, dt={dt}) = {f_euler(x, u, dt)}")

    # Problem 10: Linearization
    print("=" * 50)
    print("Problem 10: Linearization around x_bar, u_bar")
    A, B, d = linearize(x_bar=x, u_bar=u, Delta=dt)
    print(f"  x_{{k+1}} ~ A x_k + B u_k + d")
    print(f"  A =\n{A}")
    print(f"  B =\n{B}")
    print(f"  d = {d}")

    # Problem 12: Drive to target using linearization
    print("=" * 50)
    print("Problem 12: Drive to (2,0,0) using linearization")
    states_lin1 = drive_to_linearization((2, 0, 0), plot=True)
    print(f"  Final state: {states_lin1[-1]}")
    savepdf('r2d2_linearization_1')
    plt.show()

    print("Problem 12: Drive to x22=(2,2,pi/2) using linearization")
    states_lin2 = drive_to_linearization(x22, plot=True)
    print(f"  Final state: {states_lin2[-1]}")
    savepdf('r2d2_linearization_2')
    plt.show()

    # Problem 13: Drive to target using iterative MPC
    print("=" * 50)
    print("Problem 13: Drive to x22=(2,2,pi/2) using iterative MPC")
    states_mpc = drive_to_mpc(x22, plot=True)
    print(f"  Final state: {states_mpc[-1]}")
    savepdf('r2d2_iterative_1')
    plt.show()
