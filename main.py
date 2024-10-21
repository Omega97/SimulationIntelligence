"""Cart-pole simulation"""
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics import cart_pole_dynamics
from src.simulators import simulation
from src.animation import animation
from src import control_forces
from main_linearization import compute_jacobian


def plot_trajectories(cart_x, cart_y, tip_x, tip_y, force, dynamics, do_show=True):
    # plot
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

    plt.plot(tip_x, tip_y, alpha=0.5)
    plt.plot(cart_x, cart_y, alpha=0.5)
    if do_show:
        plt.show()


def main(n_steps=400, t_fin=20.,
         g=5., m_cart=0.5, m_arm=0.5,
         l_arm=1., mu_arm=0.1, mu_cart=0.1,
         delta=1e-8, save_animation=False):

    # linearization around equilibrium point
    natural_dynamics = cart_pole_dynamics(force=None, g=g, m_cart=m_cart, m_arm=m_arm,
                                          l_arm=l_arm, mu_arm=mu_arm, mu_cart=mu_cart)
    # natural_dynamics = harmonic_oscillator_dynamics(force=None, k=g, m=m_cart)

    # define the initial system (x, v, theta, omega)
    state_0 = np.array([0.0, 0.0, 0.2, 0.0])

    # equilibrium state
    x_eq = np.zeros(len(state_0), dtype=np.float64)

    # control parameters
    jacobian = compute_jacobian(natural_dynamics, x_eq, delta=delta)
    print(f'\nJacobian\n{np.round(jacobian, 2)}')

    # input cost R
    input_cost = 1.

    # state cost Q
    state_cost = (1., # low
                  100., # high
                  10.,
                  1., # low
                  )

    # Input force B
    # directly affects the acceleration of the cart, but has no direct effect on the other state variables
    # (x' = 0, v' = F_ext/m_tot, theta' = 0, omega' = 0)
    m_tot = m_cart + m_arm
    dx1_df = 1 / m_tot
    domega1_df = 1 / (l_arm * (4/3 * m_tot - m_arm))
    input_force = (0., dx1_df, 0., -domega1_df)

    force = control_forces.balancing_force(jacobian=jacobian,
                                           dt=t_fin/n_steps,
                                           input_cost=input_cost,
                                           state_cost=state_cost,
                                           input_force=input_force)

    # dynamics (with external forces)
    dynamics = cart_pole_dynamics(force, g=g, m_cart=m_cart, m_arm=m_arm,
                                  l_arm=l_arm, mu_arm=mu_arm, mu_cart=mu_cart)

    # simulation
    print('\nSimulating...')
    t_, states_ = simulation(state_0, dynamics,
                             n_steps=n_steps, t_fin=t_fin)

    # compute the trajectories
    cart_x = states_[:, 0]
    cart_y = np.zeros_like(cart_x)
    tip_x = cart_x + l_arm * np.sin(states_[:, 2])
    tip_y = l_arm * np.cos(states_[:, 2])

    # plot the trajectories
    print('\nAnimating...')
    fig, ax = plt.subplots()
    plot_trajectories(cart_x, cart_y, tip_x, tip_y,
                      force, dynamics, do_show=False)

    # animated plot
    animation(cart_x, cart_y, tip_x, tip_y,
              fig=fig, ax=ax, fps=30,
              save=save_animation)

    plt.show()


if __name__ == "__main__":
    main()
