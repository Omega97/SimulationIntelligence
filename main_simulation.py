"""Cart-pole simulation"""
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics import cart_pole_dynamics
from src.simulators import simulation
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


def main(n_steps=100, t_fin=3., delta=1e-8):

    # linearization around equilibrium point
    natural_dynamics = cart_pole_dynamics(force=None)

    # define the initial system (x, v)
    state_0 = np.array([0.0, 0.0, 0.2, 0.0])

    # equilibrium state
    x_eq = np.zeros(len(state_0), dtype=np.float64)

    # control parameters
    jacobian = compute_jacobian(natural_dynamics, x_eq, delta=delta)
    print(f'\nJacobian\n{np.round(jacobian, 3)}')
    force = control_forces.harmonic_oscillator_force()

    # dynamics (with external forces)
    dynamics = cart_pole_dynamics(force)

    # simulation
    print('\nSimulating...')
    t_, states_ = simulation(state_0, dynamics, n_steps=n_steps, t_fin=t_fin)

    # plot
    fig, ax = plt.subplots(ncols=len(state_0))
    for i in range(len(state_0)):
        ax[i].plot(t_, states_[:, i])
        ax[i].set_xlabel(f"t")
        ax[i].set_ylabel(f"x_{i}")
    plt.show()


if __name__ == "__main__":
    main()
