import matplotlib.pyplot as plt
import numpy as np


def plot_packet_trajectories(trajectories, ejecta_model):
    """
    Plots the 2D paths of photon packets through the ejecta shells.

    Parameters
    ----------
    trajectories : list of lists of tuples
        A list where each sublist contains (r, mu) tuples for a packet's path.
    ejecta_model : EjectaModel
        The ejecta model object to plot the shell boundaries.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot shell boundaries as circles
    for r_boundary in ejecta_model.radius_boundaries:
        circle = plt.Circle(
            (0, 0), r_boundary, color="gray", linestyle="--", fill=False, alpha=0.5
        )
        ax.add_artist(circle)

    # Plot packet paths
    # We create a random angle for visualization to separate the paths
    np.random.seed(0)  # for reproducible plots
    for i, path in enumerate(trajectories):
        angle = np.random.random() * 2 * np.pi
        x_coords = [p[0] * np.cos(angle) for p in path]
        y_coords = [p[0] * np.sin(angle) for p in path]
        ax.plot(x_coords, y_coords, marker="o", markersize=2, label=f"Packet {i}")

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Photon Packet Trajectories")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()


# In plotter.py, below the existing plot_packet_trajectories function


def plot_radius_vs_time(history):
    """
    Plots the radius of a single packet versus the simulation step.

    Parameters
    ----------
    history : list of tuples
        A list containing (r, mu) tuples for a single packet's path.
    """
    radii = [p[0] for p in history]
    steps = range(len(radii))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, radii, marker=".", linestyle="-", markersize=4)
    plt.xlabel("Simulation Step")
    plt.ylabel("Radius (cm)")
    plt.title("Single Packet Radius vs. Time (Step)")
    plt.grid(True)
    plt.show()


def plot_mu_vs_radius(history):
    """
    Plots the direction cosine (mu) of a packet versus its radius.

    Parameters
    ----------
    history : list of tuples
        A list containing (r, mu) tuples for a single packet's path.
    """
    radii = [p[0] for p in history]
    mus = [p[1] for p in history]

    plt.figure(figsize=(10, 5))
    plt.scatter(radii, mus, marker=".", alpha=0.6)
    plt.xlabel("Radius (cm)")
    plt.ylabel("Direction Cosine (mu)")
    plt.title("Packet Direction vs. Radius")
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.show()
