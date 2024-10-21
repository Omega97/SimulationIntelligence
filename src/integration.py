import numpy as np


ONE_SIXTH = 1 / 6


def euler(x, v, t, dt):
    """
    Update step with Euler's rule
    :param x: state
    :param v: dx/dt
    :param t: time
    :param dt: time interval
    :return: x(t+dt)
    """
    return x + dt * v(x, t)


def rk4(x, v, t, dt):
    """
    Update step with the Runge-Kutta 4th order (RK4) method
    :param x: state
    :param v: dx/dt
    :param t: time
    :param dt: time interval
    :return: x(t+dt)
    """
    k1 = dt * v(x, t)
    k2 = dt * v(x + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * v(x + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * v(x + k3, t + dt)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * ONE_SIXTH


def compute_jacobian(dynamics, equilibrium_point, t=0., delta=1e-6):
    """
    Compute the Jacobian matrix of the dynamics at the equilibrium point
    :param dynamics: f(state, t) -> dx/dt
    :param equilibrium_point: x_eq
    :param t: time
    :param delta: perturbation size
    :return: Jacobian matrix
    """
    # Compute the Jacobian matrix
    n_dim = len(equilibrium_point)
    jacobian_matrix = np.zeros((n_dim, n_dim))
    perturbations = np.eye(n_dim) * delta

    # Compute the dynamics at the equilibrium point
    for i in range(n_dim):
        # Compute the dynamics at the perturbed points
        y2 = dynamics(equilibrium_point + perturbations[i], t)
        y1 = dynamics(equilibrium_point - perturbations[i], t)

        # Compute the partial derivatives (finite difference approximation)
        jacobian_matrix[:, i] = (y2 - y1) / (2 * delta)

    return jacobian_matrix


def _test_jacobian():
    A = np.random.randint(10, size=(2, 2))
    print(f'\nA\n{A}')

    def dynamics(x, _):
        return A @ x

    x_eq = np.zeros(2)

    jacobian = compute_jacobian(dynamics, x_eq)
    print(f'\nJacobian\n{jacobian}')


if __name__ == '__main__':
    _test_jacobian()
