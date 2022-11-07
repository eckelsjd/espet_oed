import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process, plot_convergence
from skopt import gp_minimize
from joblib import Parallel

from src.models import electrospray_current_model_cpu, nonlinear_model
from src.nmc import eig_nmc_pm, eig_nmc


def f_wo_noise(x):
    return f(x, noise_level=0)


def f(x, noise_level=0.1):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level


def toy_gp_problem():
    noise_level = 0.1
    obj_fun = lambda x: f(x, noise_level=noise_level)
    res = gp_minimize(obj_fun,
                      [(-2.0, 2.0)],
                      n_calls=20,
                      acq_func="EI", acq_optimizer='lbfgs', n_restarts_optimizer=5,
                      n_initial_points=5, initial_point_generator='random',
                      x0=None, y0=None, verbose=False,
                      kappa=1.96, xi=0.01,
                      noise=noise_level
                      )
    for n_iter in range(5):
        # Plot true function.
        plt.subplot(5, 2, 2 * n_iter + 1)

        if n_iter == 0:
            show_legend = True
        else:
            show_legend = False

        ax = plot_gaussian_process(res, n_calls=n_iter,
                                   objective=f_wo_noise,
                                   noise_level=noise_level,
                                   show_legend=show_legend, show_title=False,
                                   show_next_point=False, show_acq_func=False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        # Plot EI(x)
        plt.subplot(5, 2, 2 * n_iter + 2)
        ax = plot_gaussian_process(res, n_calls=n_iter,
                                   show_legend=show_legend, show_title=False,
                                   show_mu=False, show_acq_func=True,
                                   show_observations=False,
                                   show_next_point=True)
        ax.set_ylabel("")
        ax.set_xlabel("")
    plt.show()


def plot_gp(res):
    # Extract and sort the function evaluations
    x_loc = np.squeeze(np.atleast_1d(res.x_iters))
    eig = np.squeeze(np.atleast_1d(-res.func_vals))
    s = sorted(zip(x_loc, eig))
    x_loc, eig = [np.atleast_1d(list(tup)) for tup in zip(*s)]
    x_opt = res.x[0]
    eig_opt = -res.fun

    # Generate x grid from bounds
    bounds = res.space.bounds[0]
    x_grid = np.linspace(bounds[0], bounds[1], 1000).reshape((-1, 1))

    # Get GP(x) prediction
    y_pred, sigma = res.models[-1].predict(x_grid, return_std=True)

    # Plot GP and observations
    fig = plt.figure()
    plt.plot(x_grid, -y_pred, '--k', label='$m(x)$')
    plt.fill_between(np.squeeze(x_grid), -y_pred - 1.96*sigma, -y_pred + 1.96*sigma, alpha=0.5, edgecolor=(0.4, 0.4, 0.4),
                     facecolor=(0.8, 0.8, 0.8))
    plt.plot(x_loc, eig, '.r', markersize=8, label='Observations')
    plt.axvline(x=x_opt, label='Optimum', linestyle='-', color='g')
    plt.legend(loc='best')
    plt.xlabel('Operating condition $d$')
    plt.ylabel('Expected information gain')
    fig.tight_layout(pad=1.0)
    plt.show()


def eig_func(x, var=1e-4, ppool=None):
    theta_sampler = lambda shape: np.random.rand(*shape, 1)
    N = 5000
    M = 100
    Nx = 1
    x_loc = np.atleast_1d(x).reshape((Nx, 1))
    eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, noise_cov=var, reuse_samples=True, n_jobs=-1,
                  ppool=ppool)

    return -eig.item()


def test_nonlinear():
    noise_level = 1e-4
    with Parallel(n_jobs=-1, verbose=0) as ppool:
        obj_fun = lambda x: eig_func(x, var=noise_level, ppool=ppool)
        res = gp_minimize(obj_fun,
                          [(0.0, 1.0)],
                          n_calls=15,
                          acq_func="gp_hedge", acq_optimizer='lbfgs', n_restarts_optimizer=5,
                          n_initial_points=8, initial_point_generator='lhs',
                          x0=None, y0=None, verbose=False,
                          kappa=1.96, xi=0.01,
                          noise=noise_level*2
                          )
    plot_gp(res)


if __name__ == '__main__':
    # toy_gp_problem()
    test_nonlinear()
