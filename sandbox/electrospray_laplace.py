import numpy as np
import scipy.optimize
import pygtc
import matplotlib.pyplot as plt

from src.utils import laplace_approx, batch_normal_pdf, electrospray_samplers, approx_hessian, batch_normal_sample
from src.utils import nearest_positive_definite, is_positive_definite
from src.models import electrospray_current_model_cpu


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
    g_model = electrospray_current_model_cpu(x, theta, eta)  # (M, Nx, y_dim)

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

    Ns = 10000
    samples = batch_normal_sample(map, sigma, size=Ns)  # (Ns, 3)
    fig = pygtc.plotGTC(np.squeeze(samples),
                        # chainLabels=['$\\theta_1$', '$\\theta_2$'],
                        paramNames=['$\\zeta_1$', '$\\zeta_2$', '$b_0$'],
                        panelSpacing='loose',
                        filledPlots=True,
                        nContourLevels=3,
                        nBins=int(0.05 * Ns),
                        smoothingKernel=1.5,
                        figureSize=4,
                        plotDensity=True,
                        colorsOrder=['greens', 'blues'],
                        sigmaContourLevels=True
                        )
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_laplace()
