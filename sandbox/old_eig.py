import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path
import os
import shutil
import time


def batch_normal_pdf(x, mu, cov):
    """
    Compute the multivariate normal pdf at each x location.
    Dimensions
    ----------
    d: dimension of the problem
    *: any arbitrary shape (a1, a2, ...)
    Parameters
    ----------
    x: (*, d) location to compute the multivariate normal pdf
    mu: (*, d) mean values to use at each x location
    cov: (d, d) covariance matrix, assumed same at all locations
    Returns
    -------
    pdf: (*) the multivariate normal pdf at each x location
    """
    # Make some checks on input
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(mu.shape) == 1:
        mu = mu[:, np.newaxis]

    assert cov.shape[0] == cov.shape[1] == dim
    assert x.shape[-1] == mu.shape[-1] == dim

    # Normalizing constant (scalar)
    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))

    # In exponential
    diff = x - mu  # can broadcast x - mu with x: (1, Nr, Nx, d) and mu: (Ns, Nr, Nx, d)
    diff_col = diff.reshape((*diff.shape, 1))                       # (Ns, Nr, Nx, d, 1)
    mat1 = np.linalg.inv(cov) @ diff_col                            # (d, d) x (*, d, 1) = (*, d, 1) broadcast matmult
    diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (Ns, Nr, Nx, 1, d)
    inexp = np.squeeze(diff_row @ mat1, axis=(-1, -2))              # (*, 1, d) x (*, d, 1) = (*, 1, 1)

    # Compute the pdf
    pdf = preexp * np.exp(-1/2 * inexp)
    return pdf.astype(np.float32)


def batch_normal_sample(mean, cov, size: "tuple | int" = ()):
    """
    Batch sample multivariate normal distributions.
    https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
    Arguments:
        mean: expected values of shape (…M, D)
        cov: covariance matrices of shape (…M, D, D)
        size: additional batch shape (…B)
    Returns: samples from the multivariate normal distributions
             shape: (…B, …M, D)
    """
    # Make some checks on input
    mean = np.atleast_1d(mean)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(mean.shape) == 1:
        mean = mean[:, np.newaxis]

    assert cov.shape[0] == cov.shape[1] == dim
    assert mean.shape[-1] == dim

    size = (size, ) if isinstance(size, int) else tuple(size)
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    X = np.random.standard_normal((*shape, 1))
    L = np.linalg.cholesky(cov)
    return (L @ X).reshape(shape) + mean


def fix_input_shape(x):
    """Make input shape: (Nx, xdim)
    Nx: number of experimental locations (inputs x) to evaluate at
    xdim: dimension of a single experimental input x
    """
    x = np.atleast_1d(x).astype(np.float32)
    if len(x.shape) == 1:
        # Assume one x dimension
        x = x[:, np.newaxis]
    elif len(x.shape) != 2:
        raise Exception('Incorrect input dimension')
    return x


def fix_theta_shape(theta):
    """Make theta shape: (Ns, Nx, theta_dim)
    Ns: Number of samples of model parameters for each input x
    Nx: number of experimental locations (inputs x) to evaluate at
    theta_dim: Number of model parameters
    """
    theta = np.atleast_1d(theta).astype(np.float32)
    if len(theta.shape) == 1:
        # Assume one model parameter and one location x
        theta = theta[:, np.newaxis, np.newaxis]
    elif len(theta.shape) == 2:
        # Assume only one model parameter
        theta = theta[:, :, np.newaxis]
    elif len(theta.shape) != 3:
        raise Exception('Incorrect input dimension')
    return theta


def fix_eta_shape(eta):
    """Make eta shape: (Nr, Nx, eta_dim)
    Nr: Number of realizations of nuisance params to use at each x location
    Nx: number of experimental locations (inputs x) to evaluate at
    theta_dim: Number of model parameters
    """
    eta = np.atleast_1d(eta).astype(np.float32)
    if len(eta.shape) == 1:
        # Assume one dimension and one location x
        eta = eta[:, np.newaxis, np.newaxis]
    elif len(eta.shape) == 2:
        # Assume only one parameter
        eta = eta[:, :, np.newaxis]
    elif len(eta.shape) != 3:
        raise Exception('Incorrect input dimension')
    return eta


# Nested monte carlo expected information gain estimator (pseudomarginal, sequential)
def eig_nmc_slow(Ns, Nr, x_loc, theta_sampler, eta_sampler, model, noise_cov):
    noise_cov = np.atleast_1d(noise_cov)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)  # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Sample parameters
    theta_samples = fix_theta_shape(theta_sampler(Ns, Nx))  # (Ns, Nx, theta_dim)
    eta_samples = fix_eta_shape(eta_sampler(Nr, Nx))  # (Nr, Nx, eta_dim)

    # Evaluate the model
    g_theta = model(x_loc, theta_samples, eta_samples)  # (Ns, Nr, Nx, y_dim)
    assert g_theta.shape == (Ns, Nr, Nx, y_dim)

    # Get samples of y
    y = batch_normal_sample(g_theta, noise_cov)  # (Ns, Nr, Nx, y_dim)

    # Marginalize over nuisance parameters
    likelihood = np.mean(batch_normal_pdf(y, g_theta, noise_cov), axis=1)  # (Ns, Nx)

    # Compute evidence p(y|d) = integrate(p(y|theta, eta, d), (theta, eta))
    evidence = np.zeros((Ns, Nx), dtype=np.float32)
    print(f'Samples processed: {0} out of {Ns}')
    for i in range(Ns):
        y_i = y[np.newaxis, i, :, :, :]  # (1, Nr, Nx, y_dim)
        like = batch_normal_pdf(y_i, g_theta, noise_cov)  # (Ns, Nr, Nx)
        marginal_like = np.mean(like, axis=(0, 1))  # (Nx,)
        evidence[i, :] = marginal_like
        if ((i+1) % 100) == 0:
            print(f'Samples processed: {i+1} out of {Ns}')

    # Expected information gain
    eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)  # (Nx,)
    return eig


# Combined nmc estimator, handles pseudomarginal, no for loops, lots of memory (N x M x Nx x y_dim)
def eig_nmc_comb(x_loc, theta_sampler, model, N=100, M1=1, M2=1, noise_cov=1.0, eta_sampler=None):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)      # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Sample model parameters
    theta_i = theta_sampler((N, Nx))    # (N, Nx, theta_dim)

    # Evaluate model
    if eta_sampler:
        # Outer loop
        eta_i = eta_sampler((N, Nx))                    # (N, Nx, eta_dim)
        g_theta_i = model(x_loc, theta_i, eta_i)        # (N, Nx, y_dim)

        # Marginal likelihood loop
        eta_j = eta_sampler((N, M1, Nx))
        theta_j = theta_sampler((N, M1, Nx))
        g_theta_j = model(x_loc, theta_j, eta_j)        # (N, M1, Nx, y_dim)

        # Conditional likelihood loop
        eta_k = eta_sampler((N, M2, Nx))                # (N, M2, Nx, eta_dim)
        g_theta_k = model(x_loc, theta_i[:, np.newaxis, :, :], eta_k)

    else:
        # Outer loop
        g_theta_i = model(x_loc, theta_i)               # (N, Nx, y_dim)

        # Marginal likelihood loop
        theta_j = theta_sampler((N, M1, Nx))
        g_theta_j = model(x_loc, theta_j)               # (N, M1, Nx, y_dim)

        # Conditional likelihood "loop"
        g_theta_k = np.expand_dims(g_theta_i, axis=1)   # (N, 1, Nx, y_dim)

    # Sample outer loop data y
    y_i = batch_normal_sample(g_theta_i, noise_cov)     # (N, Nx, y_dim)
    y_i = np.expand_dims(y_i, axis=1)                   # (N, 1, Nx, y_dim)

    # Marginal likelihood
    marg_like = np.mean(batch_normal_pdf(y_i, g_theta_j, noise_cov), axis=1)  # (N, Nx)

    # Conditional likelihood
    cond_like = np.mean(batch_normal_pdf(y_i, g_theta_k, noise_cov), axis=1)  # (N, Nx)

    # Expected information gain
    eig = np.mean(np.log(cond_like) - np.log(marg_like), axis=0)  # (Nx,)

    return eig


# Nested monte carlo expected information gain estimator (with parallel)
def par_eig_nmc(x_loc, theta_sampler, model, Ns=100, Nr=1, eta_sampler=None, noise_cov=1):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)  # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Sample parameters
    theta_samples = fix_theta_shape(theta_sampler(Ns, Nx))  # (Ns, Nx, theta_dim)

    # No nuisance parameters if sampler not provided
    if eta_sampler is None:
        Nr = 1

    # Allocate disk space
    mmap_folder = Path('./mmap_tmp')
    try:
        os.mkdir(mmap_folder)
    except FileExistsError:
        pass
    y_file = mmap_folder / 'y_mmap.dat'
    g_theta_file = mmap_folder / 'g_theta_mmap.dat'
    evidence_file = mmap_folder / 'evidence_mmap.dat'
    y = np.memmap(str(y_file), dtype='float32', mode='w+', shape=(Ns, Nr, Nx, y_dim))
    g_theta = np.memmap(str(g_theta_file), dtype='float32', mode='w+', shape=(Ns, Nr, Nx, y_dim))
    evidence = np.memmap(str(evidence_file), dtype='float32', mode='w+', shape=(Ns, Nx))

    # Evaluate model
    if eta_sampler is None:
        g_theta[:] = model(x_loc, theta_samples)
    else:
        eta_samples = fix_eta_shape(eta_sampler(Nr, Nx))  # (Nr, Nx, eta_dim)
        g_theta[:] = model(x_loc, theta_samples, eta_samples)
    assert g_theta.shape == (Ns, Nr, Nx, y_dim)

    # Get samples of y
    y[:] = batch_normal_sample(g_theta, noise_cov)  # y.shape = g_theta.shape

    # Get likelihood (marginalize over nuisance parameters if necessary)
    likelihood = np.mean(batch_normal_pdf(y, g_theta, noise_cov), axis=1)  # (Ns, Nx)

    # Parallel loop
    def parallel_func(idx, y, g_theta, evidence):
        like = batch_normal_pdf(y[np.newaxis, idx, :, :, :], g_theta, noise_cov)
        evidence[idx, :] = np.mean(like, axis=(0, 1))  # (Nx,)

    # Compute evidence p(y|d)
    Parallel(n_jobs=-1, verbose=1)(delayed(parallel_func)(idx, y, g_theta, evidence) for idx in range(Ns))

    # Expected information gain
    eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)  # (Nx,)

    # Clean up
    try:
        shutil.rmtree(mmap_folder)
    except:
        print('Could not clean up automatically')

    return eig