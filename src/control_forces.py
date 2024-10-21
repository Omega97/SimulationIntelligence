"""List of simple control forces for the simulation."""
import numpy as np
import scipy.linalg as la


def no_force():
    """No control force"""
    def wrap(*_):
        return np.zeros(1)
    return wrap


def harmonic_oscillator_force(k=1.):
    """Harmonic oscillator force"""
    def wrap(state, _):
        """Harmonic oscillator"""
        x, v, *_ = state
        return np.array([- k * x])
    return wrap


def lqr(A, B, Q, R):
    """
    Linear Quadratic Regulator (LQR) for discrete-time systems.
    N = number of inputs
    M = number of state features
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
    # print(f'\nP\n{np.round(P, 2)}')

    # Compute the optimal feedback gain by solving the linear system
    mat = R + B.T @ P @ B   # (N, N)
    v = B.T @ P @ A         # (N, M)
    K = la.solve(mat, v)    # (N, M)

    return K


def solve_lqr(x_ini, A_dt, B, Q, R):
    """ Compute the optimal feedback gain and input """
    K = lqr(A_dt, B, Q, R)
    u = - K @ x_ini
    return u


def balancing_force(jacobian, dt, input_cost, state_cost, input_force):
    # Solve the LQR problem

    # A_dt = exp(A_ct * dt)
    A_dt = np.array(la.expm(jacobian * dt))
    print(f'\nA_dt\n{np.round(A_dt, 2)}')

    # Input matrix
    B = np.array([input_force]).T
    print(f'\nB\n{np.round(B, 2)}')

    # State cost matrix
    Q = np.diag(state_cost)
    print(f'\nQ\n{np.round(Q, 2)}')

    # input cost matrix
    R = np.eye(1) * input_cost
    print(f'\nR\n{np.round(R, 2)}')

    def wrap(state, _) -> np.ndarray:
        f = solve_lqr(state, A_dt=A_dt, B=B, Q=Q, R=R)
        return f

    return wrap
