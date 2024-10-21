import numpy as np


def free_particle_dynamics(force=None):
    """
    Free particle dynamics
    :param force: control input (returns state vector -> np.array)
    :return: function that computes the time-derivative of the state
    """
    def wrap(state, t):
        x, v = state
        x_der = v
        v_der = 0.
        if force:
            v_der += force(state, t)[0]
        return np.array([x_der, v_der])
    return wrap


def harmonic_oscillator_dynamics(force, omega):
    """
    Harmonic oscillator dynamics
    :param force: control input (returns state vector -> np.array)
    :param omega: angular frequency
    :return: function that computes the time-derivative of the state
    """
    omega_2 = omega ** 2

    def wrap(state, t):
        """harmonic oscillator"""
        x, v = state
        x_der = v
        v_der = - omega_2 * x
        if force:
            v_der += force(state, t)[0]
        return np.array([x_der, v_der])

    return wrap


def cart_pole_dynamics(force=None, l_arm=1., m_cart=0.8, m_arm=0.2,
                       g=1., mu_cart=0.1, mu_arm=0.0):
    """
    Cart-pole dynamics
    :param force: control input (returns state vector -> np.array)
    :param m_cart: cart mass
    :param m_arm: mass of the arm
    :param l_arm: length of the arm
    :param g: gravitational acceleration
    :param mu_cart: coefficient of friction of the cart
    :param mu_arm: coefficient of friction of the arm
    :return: function that computes the time-derivative of the state
    """
    def wrap(state, t):
        """Cart-pole dynamics"""
        # Unpack the state variables
        x, v, theta, omega = state
        f = force(state, t)[0] if force else 0.

        # # Precompute some useful quantities
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        omega_sq = omega ** 2
        m_tot = m_cart + m_arm
        eta = m_arm / m_tot
        I_arm = 1 / 3 * m_arm * l_arm ** 2

        # theta'
        theta_prime = omega

        # omega'
        f1 = 3 * g * sin_theta * I_arm / l_arm
        f2 = - 3 * sin_theta * cos_theta * I_arm * omega_sq * l_arm / m_tot
        f3 = - mu_arm * omega
        f4 = - cos_theta * f
        g1 = I_arm * (4 - 3 * eta * cos_theta ** 2)
        omega_prime = (f1 + f2 + f3 + f4) / g1

        # x'
        x_prime = v

        # v'
        v_prime = (f + m_arm * l_arm * (omega_sq * sin_theta - omega * cos_theta) - mu_cart * v) / m_tot

        return np.array([x_prime, v_prime, theta_prime, omega_prime])

    return wrap
