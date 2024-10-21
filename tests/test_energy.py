"""Cart-pole simulation"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from src.dynamics import free_particle_dynamics
from src.simulators import simulation
from src.control_forces import harmonic_oscillator_force


def main(n_steps=100,
         t_fin=11.,
         m=1., k=0.1):
    """Main function to simulate the cart-pole system and plot the energy of the system."""

    # define the system
    coordinates = {"x": 0.,
                   "v": 0.1,
                   }
    state_0 = np.array(list(coordinates.values()))

    force = harmonic_oscillator_force(k)

    state_der = free_particle_dynamics(force)

    t_sim = time()
    t_1, states_1 = simulation(state_0, state_der, n_steps=n_steps * 10, t_fin=t_fin)
    t_sim = time() - t_sim
    print(f"Simulation time: {t_sim:.3f} s")

    # energy
    def kinetic_energy(state):
        x, v = state
        return 0.5 * m * v**2

    def potential_energy(state):
        x, v = state
        return 0.5 * k * x**2

    kinetic_1 = np.array([kinetic_energy(x) for x in states_1])
    potential_1 = np.array([potential_energy(x) for x in states_1])
    energy_1 = kinetic_1 + potential_1

    plt.plot(t_1, kinetic_1, label='kinetic energy')
    plt.plot(t_1, energy_1, label='total energy')
    plt.ylim(0, None)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
