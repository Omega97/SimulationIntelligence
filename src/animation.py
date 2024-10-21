import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def animation(a_x, a_y, b_x, b_y, fig=None, ax=None,
              fps=30, save=False, directory='animations'):

    # Create the figure and axis
    if ax is None:
        fig, ax = plt.subplots()
    a = np.array([np.min([a_x, b_x]), np.min([a_y, b_y])])
    b = np.array([np.max([a_x, b_x]), np.max([a_y, b_y])])
    center = (a + b) / 2
    diff = np.max(b - a) / 2 * 1.1

    ax.set_xlim(center[0] - diff, center[0] + diff)
    ax.set_ylim(center[1] - diff, center[1] + diff)

    # Initialize a line object (starting with no data)
    line, = ax.plot([], [], 'r-', lw=2)

    # Function to initialize the plot
    def init_func():
        line.set_data([], [])
        return line,

    # Function to update the plot at each frame
    def update(frame):
        # Update the line to connect vertices (a_x[frame], a_y[frame]) and (b_x[frame], b_y[frame])
        x = [a_x[frame], b_x[frame]]
        y = [a_y[frame], b_y[frame]]

        # Set new data for the line
        line.set_data(x, y)
        return line,

    # Number of frames in the animation
    frames = len(a_x)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Create animation using FuncAnimation
    interval = 1000 / fps
    ani = FuncAnimation(fig, update, frames=frames, init_func=init_func,
                        blit=True, interval=interval)

    # Save the animation if required
    if save:
        print("Saving...")
        n = len(os.listdir(directory))
        name = f'{directory}\\animation_{n+1}.gif'
        ani.save(name, writer='pillow', fps=fps)

    # Display the animation
    plt.show()


def test():

    # Set the number of vertices and define the parametric variable t_
    size = 200  # Number of points in the animation
    t_ = np.linspace(0, 2 * np.pi, size)

    # Define the parametric equations for a_x, a_y, b_x, b_y
    a_x = np.sin(t_)
    a_y = np.cos(t_)
    b_x = a_x + 1
    b_y = a_y + 1

    animation(a_x, a_y, b_x, b_y)


if __name__ == "__main__":
    test()
