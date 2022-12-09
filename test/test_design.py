import numpy as np
import logging
import matplotlib.pyplot as plt
import pygtc

from src.models import custom_nonlinear
from src.nmc import eig_nmc_pm
from src.utils import batch_normal_sample, ax_default, fix_input_shape, batch_normal_pdf
from src.utils import autocorrelation, effective_sample_size, approx_hessian
from src.mcmc import dram


def log_posterior_pm(model, theta, eta, xd, yd, noise_cov, logprior):
    """Compute unnormalized pseudomarginal log posterior for data model:
       y = G(x, theta, eta) + xi, xi ~ N(0, noise_cov)

    model: G(x, theta, eta) Forward model
    theta: (1, ..., (1 or Nd), theta_dim) Model parameters
    eta: (M, ..., (1 or Nd), eta_dim) Nuisance parameters
    xd: (Nd, x_dim) Inputs of training data
    yd: (Nd, y_dim) Outputs of training data
    noise_cov: (..., y_dim, y_dim) Experimental covariance for each training data point
    logprior: P(theta) Log of Prior density
    returns: (...,) Unnormalized posterior evaluations at theta
    """
    # Evaluate prior
    log_prior = logprior(theta)

    # Set the shapes of everything to run the model
    xd = fix_input_shape(xd)
    yd = fix_input_shape(yd)
    Nd, x_dim = xd.shape
    Nd, y_dim = yd.shape
    theta = np.atleast_1d(theta)
    if len(theta.shape) == 1:
        theta = theta.reshape((1, 1, theta.shape[0]))
    eta = np.atleast_1d(eta)
    if len(eta.shape) == 1:
        eta = eta.reshape((1, 1, eta.shape[0]))  # M=1
    theta_dim = theta.shape[-1]
    shape = eta.shape[:-1]
    eta_dim = eta.shape[-1]

    # Run the model for each eta
    g_model = model(xd, theta, eta)  # (..., Nd, y_dim)

    # Pseudomarginal likelihood (use log(sum(log())) trick by factoring out max log_like
    yd = yd.reshape((1,) * (len(shape)-1) + (Nd, y_dim))                # (...1, Nd, y_dim)
    log_like = batch_normal_pdf(yd, g_model, noise_cov, logpdf=True)    # (M, ..., Nd)
    log_like = np.sum(log_like, axis=-1)                                # (M, ...)
    max_log_like = np.atleast_1d(np.max(log_like, axis=0))              # (...,)
    if len(log_like.shape) > 1:
        max_log_like = np.expand_dims(max_log_like, axis=0)             # (1, ...)
    log_like = np.atleast_1d(np.squeeze(max_log_like, axis=0)) + \
               np.log(np.sum(np.exp(log_like - max_log_like), axis=0)) - np.log(log_like.shape[0])  # (...,)

    # Log posterior ~ log_like + log_prior
    return log_like + log_prior


def nonlinear_design():
    # Setup testing parameters
    Nx = 50
    d = np.linspace(0, 1, Nx)
    prior_std = 0.25
    prior_mean = 0.5
    N = 200
    M = 200
    Nr = 100
    gamma = 0.01
    model_func = custom_nonlinear
    theta_true = 0.3
    phi_true = 0.2

    # Compute OED estimator
    logging.info(f'Starting OED estimator: NMC with N={N}, M={M}')
    theta_sampler = lambda shape: np.random.randn(*shape, 1) * prior_std + prior_mean
    eta_sampler = lambda shape: np.random.randn(*shape, 1) * prior_std + prior_mean
    eig_est = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M,
                                 noise_cov=gamma, reuse_samples=False, n_jobs=-1, replicates=Nr)  # (Nr, Nx)
    eig_lb = np.nanpercentile(eig_est, 5, axis=0)
    eig_med = np.nanpercentile(eig_est, 50, axis=0)
    eig_ub = np.nanpercentile(eig_est, 95, axis=0)

    fig, ax = plt.subplots()
    ax.plot(d, eig_med, '-r')
    ax.fill_between(d, eig_lb, eig_ub, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=-0.01, top=1.2)
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=False)
    fig.set_size_inches(4.8, 3.6)
    fig.tight_layout()
    plt.show()

    # Optimize (i.e. just grid search for now)
    d_star_idx = np.argmax(eig_med)
    d_star = d[d_star_idx]
    logging.info(f'Optimal design obtained by grid search at d={d_star}')

    # Simulate data
    logging.info(f'Collecting data at d={d_star}...')
    g_grid_true = model_func(d, theta_true, phi_true)
    y = batch_normal_sample(model_func(d_star, theta_true, phi_true), gamma)
    g_grid_prior = model_func(d, prior_mean, prior_mean)

    fig, ax = plt.subplots()
    ax.plot(d, g_grid_prior, '-k', label='Prior model')
    ax.plot(d, g_grid_true, '-r', label='True model')
    ax.plot(d_star, y, marker="*", markersize=10, linewidth=0, markeredgecolor="red", markerfacecolor="red",
            label=r'$y$')
    ax_default(ax, xlabel=r'Operating condition $d$', ylabel=r'Model output $G(\theta, \phi, d)$', legend=True)
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=-0.01, top=1.2)
    fig.tight_layout()
    plt.show()

    # Obtain posterior for optimal design
    logprior = lambda theta: batch_normal_pdf(theta, prior_mean, prior_std**2)
    xd = d_star
    yd = y

    def logpost(theta):
        Nm = 1000
        eta = eta_sampler((Nm, 1))  # (Nm, 1, 1) for marginalization
        return log_posterior_pm(model_func, theta, eta, xd, yd, gamma, logprior)

    Ns = 10000
    theta0 = np.atleast_1d(prior_mean)
    prop_sampler = batch_normal_sample
    prop_logpdf = batch_normal_pdf
    samples, accept_ratio = dram(logpost, theta0, Ns, prop_sampler, prop_logpdf, adaptive=False, delayed=False,
                                 symmetric_prop=True, show_iter=True)

    # Plot marginals
    fig = pygtc.plotGTC(chains=samples,
                        # chainLabels=['$\\theta_1$', '$\\theta_2$'],
                        paramNames=[r'$\theta_1$'],
                        panelSpacing='loose',
                        filledPlots=False,
                        nContourLevels=3,
                        nBins=int(0.01 * Ns),
                        smoothingKernel=1.5,
                        figureSize=4,
                        plotDensity=True,
                        colorsOrder=['greens', 'blues'],
                        sigmaContourLevels=True
                        )
    plt.show()

    # Plot and compare
    Nx = 50
    d = np.linspace(0, 1, Nx)
    theta = samples.reshape((samples.shape[0], 1, 1))
    eta = np.atleast_1d(prior_mean).reshape((1, 1, 1))
    g_model = model_func(d, theta, eta)  # (Ns, Nx, 1)
    l = np.percentile(g_model, 5, axis=0)
    m = np.percentile(g_model, 50, axis=0)
    u = np.percentile(g_model, 95, axis=0)
    fig, ax = plt.subplots()
    ax.plot(xd, yd, 'ok')
    ax.plot(d, np.squeeze(m), '-r')
    ax.fill_between(d, np.squeeze(l), np.squeeze(u), alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    nonlinear_design()
