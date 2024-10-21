"""
4) Linearise the cart and pole system around the upper equilibrium point and
simulate the behavior of the linear systems given the same sequence of
control inputs.

"""
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics import cart_pole_dynamics
from src.integration import compute_jacobian


def plot_jacobian(mat):
    max_val = np.max(np.abs(mat))
    plt.xticks(np.arange(4), ['x', 'v', 'theta', 'omega'])
    plt.yticks(np.arange(4), ['x\'', 'v\'', 'theta\'', 'omega\''])
    plt.imshow(mat, cmap='bwr', vmin=-max_val, vmax=max_val)
    plt.colorbar()
    plt.title("Jacobian matrix")

    for i in range(4):
        for j in range(4):
            x = mat[i, j]
            if np.abs(x) < 1e-2:
                continue
            s = f"{x:.2f}"
            color = 'white' if x < -0.5 * max_val else 'black'
            plt.text(j, i, s, ha='center', va='center', color=color)

    plt.show()



def main(delta=1e-6):
    """
    Linearize the cart-pole dynamics around
    the stable equilibrium point (0, 0, pi, 0)
    """
    # Define the equilibrium point
    equilibrium_point = np.array([0, 0, np.pi, 0])

    # Define the dynamics
    dynamics = cart_pole_dynamics(m_cart=1., m_arm=0.5, g=1., mu_cart=0.1, mu_arm=0.1)
    jacobian_matrix = compute_jacobian(dynamics, equilibrium_point, delta=delta)

    plot_jacobian(jacobian_matrix)


if __name__ == '__main__':
    main()
