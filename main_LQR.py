"""
5) Design an LQR (linear quadratic regulator) for the
linearized version of the cart and pole obtained in the
previous hands-on. Justify all the design choices and
perform all the required checks.

**LQR calculates the optimal control by minimizing a quadratic cost function.**
This cost function typically includes terms for tracking errors, control effort, and
other relevant variables.

Here's a breakdown of the process:

1. **Define the System:** The system to be controlled is described by its state-space
equations, which consist of the state equations and the output equations.
2. **Define the Cost Function:** The quadratic cost function is defined, specifying
the desired performance objectives. This function penalizes deviations from the desired
state and excessive control effort.
3. **Solve the Riccati Equation:** The LQR control law is computed by solving a matrix
equation known as the Riccati equation. This equation relates the system dynamics, the
cost function, and the optimal control gain matrix.
4. **Apply the Control Law:** The calculated control law is then applied to the system
to generate the control input that minimizes the cost function and drives the system
towards the desired state.

**Key Points:**

- **Optimality:** LQR guarantees that the control input will minimize the defined cost function.
- **Stability:** LQR ensures that the closed-loop system is stable, meaning that any
disturbances will eventually be damped out.
- **Robustness:** LQR controllers are generally robust to uncertainties in the system model.

**Example:**
Let's consider a simple pendulum. The desired state is for the pendulum to be at rest
at a specific angle. The LQR controller would calculate the optimal torque to apply to
the pendulum to achieve this desired state while minimizing the energy used.

**In essence, LQR provides a systematic approach to finding the best control input to
achieve a desired state for a linear system.** It is a powerful tool used in various
applications, including robotics, aerospace, and process control.

"""
import numpy as np
from main_linearization import compute_jacobian
from src.dynamics import cart_pole_dynamics
import scipy.linalg as la


def lqr(A, B, Q, R):
    """
    Linear Quadratic Regulator (LQR) for discrete-time systems.
    :param A: state matrix (M, M)
    :param B: input matrix (M, N)
    :param Q: state cost matrix (M, M)
    :param R: input cost matrix (N, N)
    :return: K: optimal feedback gain (N, M)
    """
    m, n = B.shape
    assert A.shape == (m, m)
    assert Q.shape == (m, m)
    assert R.shape == (n, n)

    # Solve the discrete-time algebraic Riccati equation (DARE)
    P = la.solve_discrete_are(A, B, Q, R)  # (M, M)

    # Compute the optimal feedback gain by solving the linear system
    mat = R + B.T @ P @ B   # (N, N)
    v = B.T @ P @ A         # (N, M)
    K = la.solve(mat, v)    # (N, M)

    return K


def solve_lqr(x_ini, input_cost=1., state_cost=1., dt=.01, delta=1e-6):
    """
    Compute the optimal control input
    using LQR for the cart-pole system.
    """

    # Define the equilibrium point (x, v, theta, omega)
    x_eq = np.zeros(4, dtype=np.float64)

    # Define the dynamics (function; time derivative of the state)
    dynamics = cart_pole_dynamics(m_cart=1., m_arm=0.5, g=1.,
                                  mu_cart=0.1, mu_arm=0.1)

    # State matrix: the Jacobian of the dynamics at the equilibrium_point
    # For a physical system, 'A' can be derived from the linearized equations of motion
    # around an equilibrium point. This involves taking the Jacobian of the system’s dynamics.
    A_ct = compute_jacobian(dynamics, x_eq, delta=delta)
    # print(f'\nA_ct\n{np.round(A_ct, 2)}')

    # A_dt = exp(A_ct * dt)
    A_dt = np.array(la.expm(A_ct * dt))
    # print(f'\nA_dt\n{np.round(A_dt, 2)}')

    # Input matrix
    # 'B' represents the effect of the control input on the state variables.
    # x'=0
    # v' = F_ext/m
    # theta' = 0
    # omega' = 0
    B = np.zeros((4, 1), dtype=np.float64)
    B[1, 0] = 1.
    # print(f'\nInput Matrix B\n{np.round(B, 2)}')

    # State cost matrix
    # 'Q' is typically chosen by the control system designer to reflect the relative importance
    # of different state variables. It is a symmetric, positive semi-definite matrix.
    Q = np.eye(4) * state_cost
    # print(f'\nState Cost Matrix Q\n{np.round(Q, 2)}')

    # input cost matrix
    # 'R' is chosen to reflect the cost associated with using control inputs.
    # It is a symmetric, positive definite matrix.
    R = np.eye(1) * input_cost
    # print(f'\nInput Cost Matrix R\n{np.round(R, 2)}')

    # compute the optimal feedback gain
    K = lqr(A_dt, B, Q, R)
    # print(f'\nOptimal Feedback Gain K\n{np.round(K, 2)}')

    # compute the optimal input
    u = - K @ x_ini
    # print(f'\nOptimal Control Input u\n{np.round(u, 5)}')

    return u


def main():
    # Define the initial state
    x = np.array([0., 0., 0.1, 0.], dtype=np.float64)

    # Solve the LQR problem
    u = solve_lqr(x)

    print(f'\nControl Input u\n{u}')


if __name__ == '__main__':
    main()
