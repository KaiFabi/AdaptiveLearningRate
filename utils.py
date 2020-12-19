import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 8})

dpi = 120
bbox_inches = "tight"

def plot_gradient_descent(stats, f, x_opt, y_opt):
    markers=['s', 'o', 'p', 'P', 'x', '+', '<', 'v']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    domain_size = 3.2
    x_min, x_max = -domain_size, domain_size
    y_min, y_max = -domain_size, domain_size

    step_size = 0.01
    x = np.arange(x_min, x_max, step_size)
    y = np.arange(y_min, y_max, step_size)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy)
    ax.contour(x, y, z, levels=np.logspace(-1, 6, 128), norm=LogNorm(), cmap="viridis", linewidths=0.3)

    for (key, value), marker in zip(stats.items(), markers):
        ax.plot(stats[key]["x"], stats[key]["y"], label=key, linewidth=0.5, marker=marker, markersize=2)

    ax.plot(x_opt, y_opt, marker="*", markersize=10, color="k")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    plt.savefig("gd.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def plot_loss(stats, x_min, y_min):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    for key, value in stats.items():
        x = np.array(stats[key]["x"])
        y = np.array(stats[key]["y"])
        loss = np.sqrt((x_min - x)**2 + (y_min - y)**2)
        iteration = np.linspace(1, len(loss), len(loss))
        ax.plot(iteration, loss, label=key, linewidth=0.5)
        ax.grid("True", linestyle="--", linewidth=0.5)

    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

    n_points = 200
    ax2 = ax.inset_axes([0.07, 0.02, 0.42, 0.42])
    for key, value in stats.items():
        x = np.array(stats[key]["x"])
        y = np.array(stats[key]["y"])
        loss = np.sqrt((x_min - x)**2 + (y_min - y)**2)
        iteration = np.arange(len(loss))
        ax2.plot(iteration[:n_points], loss[:n_points], linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid("True", linestyle="--", linewidth=0.5)
    ax.indicate_inset_zoom(ax2, linewidth=1)
    
    plt.savefig("loss.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def plot_learning_rates(stats):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    i = 0
    axes = axes.flatten()
    for key, value in stats.items():
        if "+" in key:
            lr_x = np.array(stats[key]["eta_x"])
            lr_y = np.array(stats[key]["eta_y"])
            axes[i].plot(lr_x, label="$\eta_x$", linewidth=0.5)
            axes[i].plot(lr_y, label="$\eta_y$", linewidth=0.5)
            axes[i].set_title(key)
            i += 1

    for i, ax in enumerate(axes.flatten()):
        ax.legend(loc="lower right")
        if i > 1:
            ax.set_xlabel("Iterations")
        if (i+1) % 2 != 0:
            ax.set_ylabel("Learning rate")

    plt.savefig("lr.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

