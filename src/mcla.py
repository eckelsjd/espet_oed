import numpy as np
from joblib import Parallel, delayed
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

from src.utils import batch_normal_sample, batch_normal_pdf, fix_input_shape, laplace_approx, approx_hessian


def log_posterior_pm(x, theta, eta, y, model, logprior, noise_cov, eta_sampler=None):
    """Compute unnormalized pseudomarginal log posterior for data model:
       y = G(x, theta, eta) + xi, xi ~ N(0, noise_cov)

    x: (Nx, x_dim) Operating condition locations
    theta: (1, *, theta_dim) Model parameters to evaluate posterior at
    eta: (M, *, eta_dim) nuisance parameters needed to run the model
    eta_sampler: eta = sampler() -> (M, *, eta_dim) to use, overrides provided value of eta
    y: (Ne, *, y_dim) Set of Ne observed data
    model: G(x, theta, eta) Forward model
    logprior: P(theta) Log of Prior density
    noise_cov: (d, d) Measurement noise covariance matrix
    returns: (*,) Unnormalized posterior evaluations at theta
    """
    # Override samples of eta
    if eta_sampler:
        eta = eta_sampler()

    # Get shapes
    if len(theta.shape) == 1:
        theta = theta[np.newaxis, np.newaxis, :]    # (1, 1, theta_dim)
    if len(y.shape) == 2:
        y = y[:, np.newaxis, :]                     # (Ne, 1, y_dim)
    Nx, x_dim = x.shape
    M = eta.shape[0]
    eta_dim = eta.shape[-1]
    Ne = y.shape[0]
    y_dim = y.shape[-1]
    theta_dim = theta.shape[-1]
    shape = theta.shape[:-1]

    # Pseudomarginal likelihood
    g_model = model(x, theta, eta)                          # (M, *, y_dim)
    mu = np.expand_dims(g_model, axis=0)                    # (1, M, *, y_dim)
    likelihood = batch_normal_pdf(y, mu, noise_cov)         # (Ne, M, *)
    likelihood = np.prod(likelihood, axis=0)                # (M, *)
    marg_like = np.mean(likelihood, axis=0)                 # (*, )
    log_likelihood = np.log(marg_like)                      # (*, )

    # Log posterior
    return np.expand_dims(log_likelihood + logprior(theta), axis=-1)  # (*, 1)


def log_posterior(x, theta, y, model, logprior, noise_cov):
    """Compute unnormalized log posterior for data model:
       y = G(x, theta) + xi, xi ~ N(0, noise_cov)

    x: (Nx, x_dim) Operating condition locations
    theta: (1, *, theta_dim) Model parameters to evaluate posterior at
    y: (Ne, *, y_dim) Set of Ne observed data
    model: G(x, theta) Forward model
    logprior: P(theta) Log of Prior density
    noise_cov: (d, d) Measurement noise covariance matrix
    returns: (*,) Unnormalized posterior evaluations at theta
    """
    # Get shapes
    if len(theta.shape) == 1:
        theta = theta[np.newaxis, np.newaxis, :]    # (1, 1, theta_dim)
    if len(y.shape) == 2:
        y = y[:, np.newaxis, :]                     # (Ne, 1, y_dim)
    Nx, x_dim = x.shape
    Ne = y.shape[0]
    y_dim = y.shape[-1]
    theta_dim = theta.shape[-1]
    shape = theta.shape[:-1]

    # Likelihood
    g_model = model(x, theta)                                                   # (1, *, y_dim)
    log_likelihood = batch_normal_pdf(y, g_model, noise_cov, logpdf=True)       # (Ne, *)
    log_likelihood = np.sum(log_likelihood, axis=0)                             # (*, )

    # Log posterior
    return np.expand_dims(log_likelihood + logprior(theta), axis=-1)  # (*, 1)


# Monte Carlo Laplace approximation (MCLA) with Gaussian prior
# @np.errstate(divide='ignore', invalid='ignore')
def eig_mcla(x_loc, theta_sampler, model, prior_mean, prior_cov, N=100, Ne=10, noise_cov=np.asarray(1.0), n_jobs=1,
             replicates=1, batch_size=-1, ppool=None):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)  # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Get shape of parameters
    theta_temp = theta_sampler((1,))
    theta_dim = theta_temp.shape[-1]
    del theta_temp

    # Prior distribution
    prior_mean = np.atleast_1d(prior_mean).reshape((1, 1, 1, theta_dim))    # (1, Nr, bs, theta_dim)
    prior_cov = np.atleast_1d(prior_cov).reshape((theta_dim, theta_dim))    # (theta_dim, theta_dim)

    # Create temporary files for data
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as g_theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as kl_divergence_fd:
        pass

    # Allocate space for data
    Nr = replicates
    theta_i = np.memmap(theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, theta_dim))
    y_i = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=(Ne, N, Nr, Nx, y_dim))
    g_theta_i = np.memmap(g_theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
    kl_divergence = np.memmap(kl_divergence_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))

    # Start memory logging
    # daemon = Thread(target=log_memory_usage, args=(10,), daemon=True, name="Memory logger")
    # daemon.start()

    # Sample parameters
    theta_i[:] = theta_sampler((N, Nr, Nx)).astype(np.float32)      # (N, Nr, Nx, theta_dim)

    # Evaluate model
    g_theta_i[:] = model(x_loc, theta_i)                            # (N, Nr, Nx, y_dim)

    # Ne samples of data y
    y_i[:] = batch_normal_sample(g_theta_i, noise_cov, size=Ne)     # (Ne, N, Nr, Nx, y_dim)

    # Break input into batches
    if batch_size < 0:
        batch_size = Nx
    num_batches = int(np.floor(Nx / batch_size))
    if Nx % batch_size > 0:
        # Extra batch for remainder
        num_batches += 1

    # Parallel loop
    def parallel_func(idx, theta_i, y_i, kl_divergence):
        for curr_batch in range(num_batches):
            # Index the current batch
            start_idx = curr_batch * batch_size
            end_idx = start_idx + batch_size
            b_slice = slice(start_idx, end_idx) if end_idx < Nx else slice(start_idx, None)

            # Laplace approximation of posterior
            theta_curr = theta_i[idx, np.newaxis, :, b_slice, :]                    # (1, Nr, bs, theta_dim)
            y_curr = y_i[:, idx, :, b_slice, :]                                     # (Ne, Nr, bs, y_dim)
            x_curr = x_loc[b_slice, :]                                              # (bs, x_dim)
            bs = x_curr.shape[0]

            logprior = lambda theta: batch_normal_pdf(theta, prior_mean, prior_cov, logpdf=True)
            # logprior = lambda theta: np.zeros((Nr, bs))  # For a uniform prior U(0, 1)
            neg_logpost = lambda x, theta, eta: -log_posterior(x, theta, y_curr, model, logprior, noise_cov)
            sigma_inv = approx_hessian(neg_logpost, x_curr, theta_curr)

            # Fix shapes of prior (P2) and posterior (P1) Gaussian distributions
            mu_1 = theta_curr.reshape((Nr, bs, theta_dim, 1))
            mu_2 = prior_mean.reshape((1, 1, theta_dim, 1))
            sigma_1 = np.linalg.pinv(sigma_inv).reshape((Nr, bs, theta_dim, theta_dim))
            sigma_2 = prior_cov.reshape((1, 1, theta_dim, theta_dim))
            sigma_2_inv = np.linalg.inv(sigma_2)

            # Compute Dkl(P1 || P2) == Dkl(Posterior || prior) between two Gaussians
            # gg_h = approx_hessian(lambda x, theta, eta: batch_normal_pdf(theta, prior_mean, prior_cov, logpdf=True),
            #                       x_curr, theta_curr)
            # dkl = (-1/2)*np.log((2*np.pi)**(theta_dim)*np.linalg.det(sigma_1)) - theta_dim/2 - logprior(theta_curr) - np.trace()
            diff_col = mu_1 - mu_2                                  # (Nr, bs, theta_dim, 1)
            diff_row = np.transpose(diff_col, axes=(0, 1, -1, -2))  # (Nr, bs, 1, theta_dim)
            dkl = (1/2) * (np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1)) +
                           np.squeeze(diff_row @ sigma_2_inv @ diff_col, axis=(-2, -1)))  # (Nr, bs)

            # dkl = (-1/2)*np.log((2*np.pi)**theta_dim*np.linalg.det(sigma_1)) - theta_dim/2 - logprior(theta_curr)
            # dkl = dkl.reshape((Nr, bs))
            # diff_col = mu_2 - mu_1                                  # (Nr, bs, theta_dim, 1)
            # diff_row = np.transpose(diff_col, axes=(0, 1, -1, -2))  # (Nr, bs, 1, theta_dim)
            # dkl = (1/2) * (np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1)) - theta_dim +
            #                np.trace(sigma_2_inv @ sigma_1, axis1=-2, axis2=-1) +
            #                np.squeeze(diff_row @ sigma_2_inv @ diff_col, axis=(-2, -1)))  # (Nr, bs)

            # Store result (store a nan for unphysical cases)
            kl_divergence[idx, :, b_slice] = np.nan_to_num(dkl, posinf=np.nan, neginf=np.nan, nan=np.nan)
            # kl_divergence[idx, :, b_slice] = dkl

    # Compute KL divergence in parallel over outer-loop samples N
    if ppool is None:
        Parallel(n_jobs=n_jobs, verbose=9)(delayed(parallel_func)(idx, theta_i, y_i, kl_divergence) for idx in range(N))
    else:
        ppool(delayed(parallel_func)(idx, theta_i, y_i, kl_divergence) for idx in range(N))

    # Expected information gain
    eig = np.nanmean(kl_divergence, axis=0)    # (Nr, Nx)

    # Clean up
    del theta_i
    del y_i
    del g_theta_i
    del kl_divergence
    os.remove(theta_fd.name)
    os.remove(y_fd.name)
    os.remove(g_theta_fd.name)
    os.remove(kl_divergence_fd.name)

    return eig


# Monte Carlo Laplace approximation (MCLA) with Gaussian prior
@np.errstate(divide='ignore', invalid='ignore')
def eig_mcla_pm(x_loc, theta_sampler, eta_sampler, model, prior_mean, prior_cov, N=100, M=100, Ne=10,
                noise_cov=np.asarray(1.0), n_jobs=1, replicates=1, batch_size=-1, ppool=None):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)  # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Get shape of parameters
    theta_temp = theta_sampler((1,))
    eta_temp = eta_sampler((1,))
    theta_dim = theta_temp.shape[-1]
    eta_dim = eta_temp.shape[-1]
    del theta_temp
    del eta_temp

    # Prior distribution
    prior_mean = np.atleast_1d(prior_mean).reshape((1, 1, 1, theta_dim))    # (1, Nr, bs, theta_dim)
    prior_cov = np.atleast_1d(prior_cov).reshape((theta_dim, theta_dim))    # (theta_dim, theta_dim)

    # Create temporary files for data
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as eta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as g_theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as kl_divergence_fd:
        pass

    # Allocate space for data
    Nr = replicates
    theta_i = np.memmap(theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, theta_dim))
    eta_i = np.memmap(eta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, eta_dim))
    y_i = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=(Ne, N, Nr, Nx, y_dim))
    g_theta_i = np.memmap(g_theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
    kl_divergence = np.memmap(kl_divergence_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))

    # Start memory logging
    # daemon = Thread(target=log_memory_usage, args=(10,), daemon=True, name="Memory logger")
    # daemon.start()

    # Sample parameters
    theta_i[:] = theta_sampler((N, Nr, Nx)).astype(np.float32)      # (N, Nr, Nx, theta_dim)
    eta_i[:] = eta_sampler((N, Nr, Nx)).astype(np.float32)          # (N, Nr, Nx, eta_dim)

    # Evaluate model
    g_theta_i[:] = model(x_loc, theta_i, eta_i)                     # (N, Nr, Nx, y_dim)

    # Ne samples of data y
    y_i[:] = batch_normal_sample(g_theta_i, noise_cov, size=Ne)     # (Ne, N, Nr, Nx, y_dim)

    # Break input into batches
    if batch_size < 0:
        batch_size = Nx
    num_batches = int(np.floor(Nx / batch_size))
    if Nx % batch_size > 0:
        # Extra batch for remainder
        num_batches += 1

    # Parallel loop
    def parallel_func(idx, y_i, kl_divergence):
        for curr_batch in range(num_batches):
            # Index the current batch
            start_idx = curr_batch * batch_size
            end_idx = start_idx + batch_size
            b_slice = slice(start_idx, end_idx) if end_idx < Nx else slice(start_idx, None)

            # Laplace approximation of posterior
            theta_curr = theta_i[idx, np.newaxis, :, b_slice, :]                    # (1, Nr, bs, theta_dim)
            y_curr = y_i[:, idx, np.newaxis, :, b_slice, :]                         # (Ne, 1, Nr, bs, y_dim)
            x_curr = x_loc[b_slice, :]                                              # (bs, x_dim)
            bs = x_curr.shape[0]
            # sampler = lambda: eta_sampler((M, Nr, bs))                              # (M, Nr, bs, eta_dim)
            eta_curr = eta_sampler((M, Nr, bs))                                     # (M, Nr, bs, eta_dim)

            logprior = lambda theta: batch_normal_pdf(theta, prior_mean, prior_cov, logpdf=True)
            neg_logpost = lambda x, theta, eta: -log_posterior_pm(x, theta, eta, y_curr, model, logprior, noise_cov)
            sigma_inv = approx_hessian(neg_logpost, x_curr, theta_curr, eta=eta_curr)

            # Fix shapes of prior (P2) and posterior (P1) Gaussian distributions
            mu_1 = theta_curr.reshape((Nr, bs, theta_dim, 1))
            mu_2 = prior_mean.reshape((1, 1, theta_dim, 1))
            sigma_1 = np.linalg.pinv(sigma_inv).reshape((Nr, bs, theta_dim, theta_dim))
            sigma_2 = prior_cov.reshape((1, 1, theta_dim, theta_dim))
            sigma_2_inv = np.linalg.inv(sigma_2)

            # Compute Dkl(P1 || P2) == Dkl(Posterior || prior) between two Gaussians
            diff_col = mu_1 - mu_2                                      # (Nr, bs, theta_dim, 1)
            diff_row = np.transpose(diff_col, axes=(0, 1, -1, -2))      # (Nr, bs, 1, theta_dim)
            dkl = (1 / 2) * (np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1)) +
                             np.squeeze(diff_row @ sigma_2_inv @ diff_col, axis=(-2, -1)))  # (Nr, bs)
            # diff_col = mu_2 - mu_1                                  # (Nr, bs, theta_dim, 1)
            # diff_row = np.transpose(diff_col, axes=(0, 1, -1, -2))  # (Nr, bs, 1, theta_dim)
            # dkl = (1/2) * (np.log(np.linalg.det(sigma_2) / np.linalg.det(sigma_1)) - theta_dim +
            #                np.trace(sigma_2_inv @ sigma_1, axis1=-2, axis2=-1) +
            #                np.squeeze(diff_row @ sigma_2_inv @ diff_col, axis=(-2, -1)))  # (Nr, bs)

            # Store result (store a nan for unphysical cases)
            kl_divergence[idx, :, b_slice] = np.nan_to_num(dkl, posinf=np.nan, neginf=np.nan, nan=np.nan)

    # Compute KL divergence in parallel over outer-loop samples N
    if ppool is None:
        Parallel(n_jobs=n_jobs, verbose=9)(delayed(parallel_func)(idx, y_i, kl_divergence) for idx in range(N))
    else:
        ppool(delayed(parallel_func)(idx, y_i, kl_divergence) for idx in range(N))

    # Expected information gain
    eig = np.nanmean(kl_divergence, axis=0)    # (Nr, Nx)

    # Clean up
    del theta_i
    del eta_i
    del y_i
    del g_theta_i
    del kl_divergence
    os.remove(theta_fd.name)
    os.remove(eta_fd.name)
    os.remove(y_fd.name)
    os.remove(g_theta_fd.name)
    os.remove(kl_divergence_fd.name)

    return eig
