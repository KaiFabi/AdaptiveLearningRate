import numpy as np
from utils import plot_gradient_descent, plot_loss, plot_learning_rates
from gradient_descent import GradientDescent


def f(x, y):
    """
    Beale's function 
    """
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def main():
    optimizers = ["GD", "GD+", "GDM", "GDM+", "NAG", "NAG+", "Adam", "Adam+"]
    learning_rates = [0.01, 0.01, 0.015, 0.01, 0.006, 0.006, 0.0005, 0.0005]
    alphas = [0.0, 1e-4, 0.0, 1e-5, 0.0, 1e-6, 0.0, 1e-8]

    x_min, y_min = 3.0, 0.5
    x_ini, y_ini = -2.0, -1.0

    stats = {optimizer : {"x" : None, "y" : None, "eta_x" : None, "eta_y" : None} for optimizer in optimizers}

    n_iterations = 1000 # 10000 adam # 4400 nag # 2000 gdm 1200 gd

    for optimizer, learning_rate, alpha in zip(optimizers, learning_rates, alphas):
        print(optimizer)
        model = GradientDescent(f, x_ini, y_ini, n_iterations, learning_rate, alpha, optimizer)
        x_hist, y_hist, lr_x_hist, lr_y_hist = model.minimize()
        stats[optimizer]["x"] = x_hist
        stats[optimizer]["y"] = y_hist
        stats[optimizer]["eta_x"] = lr_x_hist
        stats[optimizer]["eta_y"] = lr_y_hist

    plot_gradient_descent(stats, f, x_min, y_min)
    plot_loss(stats, x_min, y_min)
    plot_learning_rates(stats)


if __name__ == '__main__':
    main()

