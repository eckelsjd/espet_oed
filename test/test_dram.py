import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
import pygtc

from src.mcmc import dram
from src.utils import laplace_approx, autocorrelation, effective_sample_size, batch_normal_pdf, batch_normal_sample


def test_laplace_1d():
    a = 2
    b = 5
    x = np.random.beta(a, b, 1000)
    x_grid = np.linspace(np.min(x), np.max(x), 10000)
    pdf = beta.pdf(x_grid, a, b)

    x0 = 0.5
    x0, cov0 = laplace_approx(x0, lambda xt: np.log(beta.pdf(xt, a, b)))
    print(x0, cov0)

    # Manual 2nd derivative of f
    # idx = np.argmax(pdf)
    # dx = x_grid[1] - x_grid[0]
    # hess = -(np.log(pdf)[idx+1] - 2*np.log(pdf)[idx] + np.log(pdf)[idx-1]) / dx**2
    # cov0 = 1/hess

    norm_pdf = norm.pdf(x_grid, x0, np.squeeze(np.sqrt(cov0)))

    plt.figure()
    plt.hist(x, density=True, bins=50, color='r', edgecolor='black', linewidth=1.2)
    plt.plot(x_grid, pdf, '-r', linewidth=4)
    plt.plot(x_grid, np.squeeze(norm_pdf), '-k', linewidth=4)
    plt.xlabel(r'X')
    plt.show()


def log_banana(x):
    if (len(x.shape) == 1):
        x = x[np.newaxis, :]
    N, d = x.shape
    x1p = x[:, 0]
    x2p = x[:, 1] + (np.square(x[:, 0]) + 1)
    xp = np.concatenate((x1p[:, np.newaxis], x2p[:, np.newaxis]), axis=1)
    sigma = np.array([[1, 0.9], [0.9, 1]])
    mu = np.array([0, 0])
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(sigma)**0.5
    diff = xp - np.tile(mu[np.newaxis, :], (N, 1))
    sol = np.linalg.solve(sigma, diff.T)
    inexp = np.einsum("ij,ij->j", diff.T, sol)
    return np.log(preexp) - 0.5 * inexp


def test_dram():
    # Construct a linear model
    dim = 2

    # Get analytical posterior

    # Define log posterior

    # Get DRAM posterior samples
    Ns = 20000
    x0 = np.random.rand(dim)
    # logpdf = lambda x: np.log(batch_normal_pdf(x, np.array([0, 0]), np.eye(2)))
    logpdf = log_banana
    prop_sampler = batch_normal_sample
    prop_logpdf = batch_normal_pdf
    samples, accept_ratio = dram(logpdf, x0, Ns, prop_sampler, prop_logpdf, adaptive=True, delayed=True,
                                 symmetric_prop=True, show_iter=True)
    lag, corr = autocorrelation(samples)
    ess = effective_sample_size(Ns, corr)

    # Plot mixing
    fig, axs = plt.subplots(dim, 1)
    for i in range(dim):
        axs[i].plot(samples[:, i], '-k')
        axs[i].set_ylabel(f'$\\theta _{i}$', fontsize=14)
    axs[-1].set_xlabel(r'Number of samples')
    plt.show()

    # Plot marginals
    fig = pygtc.plotGTC(chains=samples,
                        # chainLabels=['$\\theta_1$', '$\\theta_2$'],
                        paramNames=['$\\theta_1$', '$\\theta_2$'],
                        panelSpacing='loose',
                        filledPlots=False,
                        nContourLevels=3,
                        nBins=int(0.005*Ns),
                        smoothingKernel=1.5,
                        figureSize=4,
                        plotDensity=True,
                        colorsOrder=['greens', 'blues'],
                        sigmaContourLevels=True
                        )
    plt.show()

    # Plot acceptance ratio
    plt.figure()
    idx = np.arange(0, accept_ratio.shape[0])
    skip = 100
    plt.plot(idx[::skip], accept_ratio[::skip], '-k')
    plt.xlabel(r'Number of samples')
    plt.ylabel(r'Acceptance ratio')
    plt.ylim((0, 1))
    plt.show()

    # Plot auto-correlation and ESS
    plt.figure()
    for i in range(dim):
        plt.plot(lag, corr[:, i], '-o', label=f'$\\theta _{i}$, ESS=${round(ess[i])}$')
    plt.xlabel(r'Lag $l$')
    plt.ylabel(r'Auto-correlation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_laplace_1d()
    # test_laplace_2d()
    test_dram()
