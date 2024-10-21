"""Cart-pole simulation"""
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics import cart_pole_dynamics
from src.simulators import simulation
from src.animation import animation
from src import control_forces


def main(n_steps=200, t_fin=8.,
         k=1.0, g=9.81,
         m_cart=5., m_arm=1.,
         l_arm=1.,
         mu_arm=0.1, mu_cart=0.2):

    # define the initial system
    state_0 = np.array([0.,     # x
                        0.,     # v
                        np.pi,  # theta
                        4.])    # omega

    # control force
    force = control_forces.harmonic_oscillator_force(k=k)

    # simulation
    dynamics = cart_pole_dynamics(force, g=g,
                                  m_cart=m_cart, m_arm=m_arm,
                                  l_arm=l_arm,
                                  mu_arm=mu_arm, mu_cart=mu_cart)

    def energy(x, v, theta, omega):
        # Kinetic Energy of the Cart
        T_cart = 0.5 * m_cart * v ** 2

        # Kinetic Energy of the Pole (Translational and Rotational)
        v_pole_x = v + l_arm * omega * np.cos(theta)  # Horizontal velocity of the pole's center of mass
        v_pole_y = l_arm * omega * np.sin(theta)  # Vertical velocity of the pole's center of mass
        T_pole = 0.5 * m_arm * (v_pole_x ** 2 + v_pole_y ** 2)

        # Potential Energy of the Cart
        U_cart = 0.5 * k * x ** 2

        # Potential Energy of the Pole
        U_pole = m_arm * g * l_arm * (1+np.cos(theta))

        return T_cart + T_pole, U_pole + U_cart

    print('Simulating...')
    t_, states_ = simulation(state_0, dynamics,
                             n_steps=n_steps, t_fin=t_fin)

    # plot
    fig, ax = plt.subplots()
    title = []
    if force.__doc__:
        title.append(force.__doc__)
    if dynamics.__doc__:
        title.append(dynamics.__doc__)
    if len(title) > 0:
        title = ", ".join(title)
    else:
        title = "Simulation"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    cart_x = states_[:, 0]
    cart_y = np.zeros_like(cart_x)
    tip_x = cart_x + l_arm * np.sin(states_[:, 2])
    tip_y = l_arm * np.cos(states_[:, 2])
    plt.plot(tip_x, tip_y, alpha=0.5)
    plt.plot(cart_x, cart_y, alpha=0.5)

    # animated plot
    print('Animating...')
    animation(cart_x, cart_y, tip_x, tip_y,
              fig=fig, ax=ax, fps=30,
              save=False)

    x_ = states_[:, 0]
    v_ = states_[:, 1]
    theta_ = states_[:, 2]
    omega_ = states_[:, 3]

    kinetic_, potential_ = energy(x_, v_, theta_, omega_)

    plt.title('Energy')
    plt.fill_between(t_, potential_, color='b', alpha=0.5, label='Potential')
    plt.fill_between(t_, potential_, kinetic_ + potential_, color='orange', alpha=0.5, label='Kinetic')
    plt.xlabel('t')
    plt.ylabel('E')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
