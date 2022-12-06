import numpy as np
import scipy.optimize
import pygtc
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

from src.utils import laplace_approx, batch_normal_pdf, electrospray_samplers, approx_hessian, batch_normal_sample
from src.utils import nearest_positive_definite, is_positive_definite, ax_default
from src.models import electrospray_current_model
from src.lg import eig_lg


def electrospray_log_posterior(theta, eta, x, y, noise_var):
    """Compute unnormalized pseudomarginal log posterior for data model:
       y = G(x, theta, eta) + xi, xi ~ N(0, noise_cov)

    x: (Nx,) Voltages of training data
    y: (Nx,) Currents of training data
    noise_var: (Nx,) Experimental variance for each training data point
    theta: (theta_dim,) Model parameters
    eta: (M, eta_dim) Nuisance parameters
    model: G(x, theta, eta) Forward model for array current prediction
    returns: (1,) Unnormalized posterior evaluations at theta
    """
    # Set the shapes of everything to run the model
    Nx = x.shape[0]
    x = x.reshape((Nx, 1))
    theta_dim = np.atleast_1d(theta).shape[0]
    theta = theta.reshape((1, theta_dim))
    theta = np.tile(theta, (Nx, 1)).reshape((1, Nx, theta_dim))
    M, eta_dim = eta.shape
    eta = eta.reshape((M, 1, eta_dim))
    eta = np.tile(eta, (1, Nx, 1))

    # Run the model for each eta
    g_model = electrospray_current_model(x, theta, eta)  # (M, Nx, y_dim)

    # Pseudomarginal likelihood (use log(sum(log())) trick by factoring out max log_like
    y = y.reshape((1, Nx, 1))
    noise_var = noise_var.reshape((1, Nx, 1, 1))
    log_like = batch_normal_pdf(y, g_model, noise_var, logpdf=True)    # (M, Nx)
    log_like = np.sum(log_like, axis=-1)  # (M,)
    max_log_like = np.max(log_like)
    log_like = np.atleast_1d(max_log_like) + np.log(np.sum(np.exp(log_like - max_log_like)))

    # Log posterior with uniform prior is log_like + K
    return np.squeeze(log_like)


def test_laplace():
    # Load data
    exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
    Ndata = exp_data.shape[1]
    xdata = exp_data[0, :]      # (Ndata,)
    ydata = exp_data[1, :]      # (Ndata,)
    vardata = exp_data[2, :]    # (Ndata,)

    # Form negative log posterior objective
    Nr = 100
    theta_sampler, eta_sampler = electrospray_samplers(Ne=576)
    eta = eta_sampler((Nr,))
    neg_logpost = lambda theta: -electrospray_log_posterior(theta, eta, xdata, ydata, vardata)

    # Run laplace approximation
    theta0 = np.array([2.57, 1.69e-2, 2e-5])
    res = scipy.optimize.minimize(neg_logpost, theta0, method='Nelder-Mead')
    map = res.x
    neg_logpost = lambda x, theta, eta: -electrospray_log_posterior(theta, eta, x, ydata, vardata)
    sigma_inv = approx_hessian(neg_logpost, xdata.reshape((Ndata, 1)), map, eta, pert=0.01)
    sigma = np.linalg.pinv(sigma_inv)
    print(f'Sigma: {sigma}')
    if not is_positive_definite(sigma):
        sigma = nearest_positive_definite(sigma)
        print(f'Sigma closest: {sigma}')

    # Compute linear gaussian estimate
    theta_mean = np.broadcast_to(map, (Nr, 1, 3))   # (Nr, 1, theta_dim)
    eta_mean = np.expand_dims(eta, axis=1)          # (Nr, 1, eta_dim)
    theta_cov = sigma                               # (theta_dim, theta_dim)
    # geo_cov = np.broadcast_to(np.array([0.25e-5, 5.23e-6, 3.596e-6, 4e-3, 5.13e-6, 7.5e-8]), (576, 6)).reshape((576*6,))
    # eta_cov = np.concatenate((np.array([6.04e-15, 0.06075, 0.0105e-2, 0.001e3, 0.201, 1.003e4]), geo_cov))  # (4039, )
    # eta_cov = np.diag(eta_cov)  # (4039, 4039)
    # eta_mean = np.concatenate((np.array([1.51e-13, 1.5115, 5.024e-2, 1.282e3, 3.014e-2, 5.5e5]),
    #                            np.broadcast_to(np.array([]), (576, 6)).reshape((576*6,))))
    Nx = 50
    x = np.linspace(800, 1845, Nx)
    noise_cov = np.mean(vardata)
    gamma = noise_cov * np.eye(1)
    model_func = electrospray_current_model
    data = np.load(str(Path('../results') / f'nmc_electrospray_eig_truth.npz'))
    eig_exact = np.nanmean(data['eig_truth'], axis=0).reshape((Nx,))

    # eig_estimate = np.zeros((Nr, Nx))
    # def parallel_func(idx):
    #     eig_estimate[idx, :] = eig_lg_marg(x, model_func, theta_mean[idx, ...], theta_cov, eta_mean[idx, ...], eta_cov, gamma)
    # with Parallel(n_jobs=-1, verbose=9) as ppool:
    #     ppool(delayed(parallel_func)(idx) for idx in range(Nr))

    eig_estimate = eig_lg(x, model_func, theta_mean, theta_cov, eta_mean, gamma)  # (Nr, Nx)

    eig_lb = np.nanpercentile(eig_estimate, 5, axis=0)
    eig_med = np.nanpercentile(eig_estimate, 50, axis=0)
    eig_ub = np.nanpercentile(eig_estimate, 95, axis=0)

    fig, ax = plt.subplots()
    sl = slice(0, -1)
    ax.plot(x[sl], eig_exact[sl], '-k', label='Exact')
    ax.plot(x[sl], eig_med[sl], '-r', label='Estimate')
    ax.fill_between(x[sl], eig_lb[sl], eig_ub[sl], alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=True)
    fig.set_size_inches(4.8, 3.6)
    fig.tight_layout()
    plt.show()

    # Ns = 10000
    # samples = batch_normal_sample(map, sigma, size=Ns)  # (Ns, 3)
    # fig = pygtc.plotGTC(np.squeeze(samples),
    #                     # chainLabels=['$\\theta_1$', '$\\theta_2$'],
    #                     paramNames=['$\\zeta_1$', '$\\zeta_2$', '$b_0$'],
    #                     panelSpacing='loose',
    #                     filledPlots=True,
    #                     nContourLevels=3,
    #                     nBins=int(0.05 * Ns),
    #                     smoothingKernel=1.5,
    #                     figureSize=4,
    #                     plotDensity=True,
    #                     colorsOrder=['greens', 'blues'],
    #                     sigmaContourLevels=True
    #                     )
    # fig.tight_layout()
    # plt.show()


if __name__ == '__main__':
    test_laplace()
