import numpy as np
from src.integration import rk4


def simulation(state_0, state_der, n_steps, t_fin, integration_alg=rk4):
    """Quadrature: Simulate the dynamics of the system
    returns: an array of times, and an array of states
    """

    # initialization
    state = state_0
    states_ = [state]
    t_ = np.linspace(0, t_fin, n_steps+1)
    dt = t_[1]

    # run the simulation
    for t in t_[:-1]:
        state = integration_alg(state, state_der, t, dt=dt)
        states_.append(state)

    return t_, np.array(states_)
