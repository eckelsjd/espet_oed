import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path

from src.models import electrospray_current_model_cpu, nonlinear_model, linear_gaussian_model
from src.models import custom_nonlinear
from src.nmc import eig_nmc_pm, eig_nmc
from src.mcla import eig_mcla_pm, eig_mcla
from src.lg import eig_lg
from src.utils import model_1d_batch, ax_default, linear_eig, electrospray_samplers


def test_linear_gaussian_model(estimator='nmc'):
    # Linear gaussian model example
    N = 500
    M = 50
    Nx = 50
    Nr = 20
    noise_var = 0.01
    prior_mean = 0
    prior_cov = 1
    theta_sampler = lambda shape: np.random.randn(*shape, 1)
    eta_sampler = lambda shape: np.random.randn(*shape, 1)
    noise_cov = np.array([[noise_var, 0], [0, noise_var]])
    x_loc = np.linspace(0, 1, Nx).reshape((Nx, 1))
    d = np.squeeze(x_loc)  # (Nx, )

    # Compute analytical EIG
    y_dim = 2
    A = np.zeros((Nx, y_dim, 1))
    A[:, 0, 0] = x_loc[:, 0]
    sigma = np.array([[[prior_cov]]])
    eig_truth_marg = linear_eig(A, sigma, np.expand_dims(noise_cov, axis=0))  # (Nx,)
    A = np.zeros((Nx, y_dim, 2))
    A[:, 0, 0] = x_loc[:, 0]
    A[:, 1, 1] = 1 - x_loc[:, 0]
    sigma = np.expand_dims(prior_cov*np.eye(y_dim), axis=0)
    eig_truth_joint = linear_eig(A, sigma, np.expand_dims(noise_cov, axis=0))  # (Nx,)

    # Compute estimator EIG
    if estimator == 'nmc':
        # eig_estimate = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, linear_gaussian_model, N=N, M1=M, M2=M,
        #                           noise_cov=noise_cov, reuse_samples=False, n_jobs=-1, batch_size=-1, replicates=Nr)
        theta_sampler = lambda shape: np.random.randn(*shape, 2) * np.sqrt(prior_cov) + prior_mean
        eig_estimate = eig_nmc(x_loc, theta_sampler, linear_gaussian_model, N=N, M=M, replicates=Nr,
                               noise_cov=noise_cov, reuse_samples=False, n_jobs=-1)
    elif estimator == 'mcla':
        pass
        # Marginal
        # eig_estimate = eig_mcla_pm(x_loc, theta_sampler, eta_sampler, linear_gaussian_model, prior_mean, prior_cov,
        #                            N=N, M=M, Ne=10, noise_cov=noise_cov, n_jobs=-1, batch_size=-1, replicates=Nr)

        # Joint (sample theta and eta together, with eta along last axis)
        # theta_sampler = lambda shape: np.random.randn(*shape, 2)
        # prior_mean = np.array([0, 0])
        # prior_cov = np.eye(2)
        # eig_estimate = eig_mcla(x_loc, theta_sampler, linear_gaussian_model, prior_mean, prior_cov, N=N, Ne=10,
        #                         noise_cov=noise_cov, replicates=Nr, n_jobs=-1, batch_size=-1)

    # Compute percentiles over replicates
    # eig_lb = np.nanpercentile(eig_estimate, 5, axis=0)
    # eig_med = np.nanpercentile(eig_estimate, 50, axis=0)
    # eig_ub = np.nanpercentile(eig_estimate, 95, axis=0)

    fig, ax = plt.subplots()
    ax.plot(d, eig_truth_marg, '--k', label=r'Marginal $p(\theta)$')
    ax.plot(d, eig_truth_joint, '-k', label=r'Joint $p(\theta, \phi)$')
    # ax.plot(d, eig_med, '-r', label=r'Estimator')
    # ax.fill_between(d, eig_lb, eig_ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=True)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=-0.01)
    fig.set_size_inches(4.8, 3.6)
    plt.tight_layout()
    plt.show()
    fig.savefig(str(Path('../results/figs') / f'linear-eig.png'), dpi=300, format='png')


def test_1d_nonlinear_model():
    prior_mean = 0.5
    prior_cov = 0.4**2
    # theta_sampler = lambda shape: np.random.randn(*shape, 1)*np.sqrt(prior_cov) + prior_mean
    theta_sampler = lambda shape: np.random.rand(*shape, 1)
    N = 2000
    M = 1000
    Nx = 50
    x_loc = np.linspace(0, 1, Nx).reshape((Nx, 1))
    d = np.squeeze(x_loc)
    var = 1e-4
    eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, replicates=10, noise_cov=var, reuse_samples=False,
                  n_jobs=-1)
    eig_lb = np.nanpercentile(eig, 5, axis=0)
    eig_med = np.nanpercentile(eig, 50, axis=0)
    eig_ub = np.nanpercentile(eig, 95, axis=0)

    # MCLA estimator
    eig_est = eig_mcla(x_loc, theta_sampler, nonlinear_model, prior_mean, prior_cov, N=N, Ne=10, noise_cov=var,
                       replicates=10, n_jobs=-1, batch_size=-1)
    eig_est_lb = np.nanpercentile(eig_est, 5, axis=0)
    eig_est_med = np.nanpercentile(eig_est, 50, axis=0)
    eig_est_ub = np.nanpercentile(eig_est, 95, axis=0)

    fig, ax = plt.subplots()
    ax.plot(d, eig_med, '-k', label=r'NMC')
    ax.fill_between(d, eig_lb, eig_ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax.plot(d, eig_est_med, '-r', label=r'MCLA')
    ax.fill_between(d, eig_est_lb, eig_est_ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=True)
    ax.set_xlim(left=0, right=1)
    fig.set_size_inches(4.8, 3.6)
    plt.tight_layout()
    plt.show()


def test_2d_nonlinear_model():
    # Nonlinear model example, 2-d
    theta_sampler = lambda shape: np.random.rand(*shape, 1)
    N = 10000
    M = 1000
    Nx = [20, 20]  # [Nx, Ny, Nz, ..., Nd] - discretization in each batch dimension
    loc = [np.linspace(0, 1, n) for n in Nx]
    pt_grids = np.meshgrid(*loc)
    x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
    var = 1e-4 * np.eye(2)
    t1 = time.time()
    eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, noise_cov=var, reuse_samples=True, n_jobs=1)
    t2 = time.time()
    print(f'Total time: {t2 - t1:.2f} s')

    # Reform grids
    grid_d1, grid_d2 = [x_loc[:, i].reshape((Nx[1], Nx[0])) for i in range(2)]  # reform grids
    eig_grid = eig.reshape((Nx[1], Nx[0]))

    # Plot results
    plt.figure()
    c = plt.contourf(grid_d1, grid_d2, eig_grid, 60, cmap='jet')
    plt.colorbar(c)
    plt.cla()
    plt.contour(grid_d1, grid_d2, eig_grid, 15, cmap='jet')
    plt.xlabel('$d_1$')
    plt.ylabel('$d_2$')
    plt.show()


def test_array_current_model(dim=1):
    # Get samplers
    theta_sampler, eta_sampler = electrospray_samplers()

    # Experimental data
    exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
    var = 2 * np.max(exp_data[2, :])

    # Set sample sizes
    N = 500
    Nx = 40
    M1 = 300
    M2 = 300
    n_jobs = -1
    bs = -1

    if dim == 1:
        x_loc = np.linspace(800, 1840, Nx).reshape((Nx, 1))
        eig = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, electrospray_current_model_cpu, N=N, M1=M1, M2=M2,
                         noise_cov=var, reuse_samples=False, n_jobs=n_jobs, batch_size=bs)
        plt.figure()
        plt.plot(x_loc, eig, '-k')
        plt.xlabel('Voltage [V]')
        plt.ylabel('Expected information gain')
        plt.show()

        return eig

    if dim == 2:
        Ngrid = [Nx, Nx]
        loc = [np.linspace(1000, 1840, n) for n in Ngrid]
        pt_grids = np.meshgrid(*loc)
        x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
        noise_cov = np.eye(2)*var
        model_func = lambda x, theta, eta: model_1d_batch(x, theta, eta, electrospray_current_model_cpu)
        eig = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, model_func, N=N, M1=M1, M2=M2,
                         noise_cov=noise_cov, reuse_samples=True, n_jobs=n_jobs)

        # Reform grids
        grid_d1, grid_d2 = [x_loc[:, i].reshape((Ngrid[1], Ngrid[0])) for i in range(2)]  # reform grids
        eig_grid = eig.reshape((Ngrid[1], Ngrid[0]))

        # Plot results
        plt.figure()
        c = plt.contourf(grid_d1, grid_d2, eig_grid, 60, cmap='jet')
        plt.colorbar(c)
        plt.cla()
        plt.contour(grid_d1, grid_d2, eig_grid, 15, cmap='jet')
        plt.xlabel('$Voltage 1 [V]$')
        plt.ylabel('$Voltage 2 [V]$')
        plt.show()

        return eig


def test_custom_nonlinear():
    Nx = 50
    d = np.linspace(0, 1, Nx)
    std = 0.25
    mean = 0.5
    N = 500
    M = 500
    Nr = 100
    gamma = 0.01
    model_func = custom_nonlinear

    # Marginal
    theta_sampler = lambda shape: np.random.randn(*shape, 1) * std + mean
    eta_sampler = lambda shape: np.random.randn(*shape, 1) * std + mean
    eig_estimate_pm = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M,
                                 noise_cov=gamma, reuse_samples=False, n_jobs=-1, replicates=Nr)  # (Nr, Nx)
    eig_lb_pm = np.nanpercentile(eig_estimate_pm, 5, axis=0)
    eig_med_pm = np.nanpercentile(eig_estimate_pm, 50, axis=0)
    eig_ub_pm = np.nanpercentile(eig_estimate_pm, 95, axis=0)

    # Joint
    theta_sampler = lambda shape: np.random.randn(*shape, 2) * std + mean
    eig_estimate_joint = eig_nmc(d, theta_sampler, model_func, N=N, M=M, noise_cov=gamma, reuse_samples=False,
                                 n_jobs=-1, replicates=Nr)  # (Nr, Nx)
    eig_lb_joint = np.nanpercentile(eig_estimate_joint, 5, axis=0)
    eig_med_joint = np.nanpercentile(eig_estimate_joint, 50, axis=0)
    eig_ub_joint = np.nanpercentile(eig_estimate_joint, 95, axis=0)

    fig, ax = plt.subplots()
    ax.plot(d, eig_med_joint, '-k', label=r'Joint p($\theta$, $\phi$)')
    ax.fill_between(d, eig_lb_joint, eig_ub_joint, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='gray')
    ax.plot(d, eig_med_pm, '-r', label=r'Marginal p($\theta$)')
    ax.fill_between(d, eig_lb_pm, eig_ub_pm, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=-0.01, top=1.2)
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=True)
    fig.set_size_inches(4.8, 3.6)
    plt.tight_layout()
    fig.savefig(str(Path('../results/figs')/'nonlinear_joint_marg_eig.png'), dpi=300, format='png')
    plt.show()


def test_lg():
    Nx = 50
    theta_mean = np.array([0.5])
    eta = np.array([0.5])
    theta_var = 0.25**2
    noise_cov = 0.01
    x = np.linspace(0, 1, Nx)

    # Compute eig
    eig_estimate = eig_lg(x, custom_nonlinear, theta_mean, theta_var, eta, noise_cov)

    plt.figure()
    plt.plot(x, eig_estimate, '--r')
    plt.xlabel('Operating condition $d$')
    plt.ylabel('Expected information gain')
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # eig = test_array_current_model(dim=1)
    # test_custom_nonlinear()
    # test_1d_nonlinear_model()
    # test_linear_gaussian_model(estimator='mcla')
    test_lg()
