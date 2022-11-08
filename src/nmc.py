# Nested monte carlo estimators

import numpy as np
from joblib import Parallel, delayed
from threading import Thread
import tempfile
import os

from src.utils import batch_normal_sample, batch_normal_pdf, fix_input_shape, memory, log_memory_usage


# Nested monte carlo expected information gain estimator
def eig_nmc(x_loc, theta_sampler, model, N=100, M=100, noise_cov=1.0, reuse_samples=False, n_jobs=1, ppool=None):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)                              # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Get shape of parameters
    theta_temp = theta_sampler((1,))
    theta_dim = theta_temp.shape[-1]
    del theta_temp

    # Create temporary files for data
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as g_theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as evidence_fd:
        pass

    # Allocate space for data
    theta_i = np.memmap(theta_fd.name, dtype='float32', mode='r+', shape=(N, Nx, theta_dim))
    y_i = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=(N, Nx, y_dim))
    g_theta_i = np.memmap(g_theta_fd.name, dtype='float32', mode='r+', shape=(N, Nx, y_dim))
    evidence = np.memmap(evidence_fd.name, dtype='float32', mode='r+', shape=(N, Nx))

    # Sample model parameters
    theta_i[:] = theta_sampler((N, Nx)).astype(np.float32)          # (N, Nx, theta_dim)

    # Evaluate model
    g_theta_i[:] = model(x_loc, theta_i)                            # (N, Nx, y_dim)

    # Sample outer loop data y
    y_i[:] = batch_normal_sample(g_theta_i, noise_cov)              # (N, Nx, y_dim)

    # Parallel loop
    def parallel_func(idx, y_i, g_theta_i, evidence):
        y_curr = y_i[idx, np.newaxis, :, :]                         # (1, Nx, y_dim)
        if reuse_samples:
            evidence[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_i, noise_cov), axis=0)
        else:
            theta_j = theta_sampler((M, Nx)).astype(np.float32)     # (M, Nx, theta_dim)
            g_theta_j = model(x_loc, theta_j)                       # (M, Nx, y_dim)
            evidence[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_j, noise_cov), axis=0)

        if ((idx + 1) % 100) == 0 and n_jobs == 1:
            print(f'Samples processed: {idx + 1} out of {N}')

    # Compute evidence p(y|d)
    if ppool is None:
        Parallel(n_jobs=n_jobs, verbose=1)(delayed(parallel_func)(idx, y_i, g_theta_i, evidence) for idx in range(N))
    else:
        ppool(delayed(parallel_func)(idx, y_i, g_theta_i, evidence) for idx in range(N))

    # Compute likelihood
    likelihood = batch_normal_pdf(y_i, g_theta_i, noise_cov)        # (N, Nx)

    # Expected information gain
    eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)    # (Nx,)
    eig = np.nan_to_num(eig, nan=0, posinf=0, neginf=0)

    # Clean up
    del theta_i
    del y_i
    del g_theta_i
    del evidence
    os.remove(theta_fd.name)
    os.remove(y_fd.name)
    os.remove(g_theta_fd.name)
    os.remove(evidence_fd.name)

    return eig


# Nested monte carlo expected information gain estimator (pseudomarginal)
# @memory(percentage=1.1)
def eig_nmc_pm(x_loc, theta_sampler, eta_sampler, model, N=100, M1=100, M2=100, noise_cov=np.asarray(1.0),
               reuse_samples=False, n_jobs=1, batch_size=-1, replicates=1):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)                              # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Get shape of parameters
    theta_temp = theta_sampler((1,))
    eta_temp = eta_sampler((1,))
    theta_dim = theta_temp.shape[-1]
    eta_dim = eta_temp.shape[-1]
    del theta_temp
    del eta_temp

    # Create temporary files for data
    with tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as eta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as y_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as g_theta_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as likelihood_fd, \
            tempfile.NamedTemporaryFile(suffix='.dat', mode='w+b', delete=False) as evidence_fd:
        pass

    # Allocate space for data
    Nr = replicates
    theta_i = np.memmap(theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, theta_dim))
    eta_i = np.memmap(eta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, eta_dim))
    y_i = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
    g_theta_i = np.memmap(g_theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
    likelihood = np.memmap(likelihood_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))
    evidence = np.memmap(evidence_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))

    # Start memory logging
    # daemon = Thread(target=log_memory_usage, args=(10,), daemon=True, name="Memory logger")
    # daemon.start()

    # Sample parameters
    theta_i[:] = theta_sampler((N, Nr, Nx)).astype(np.float32)      # (N, Nr, Nx, theta_dim)
    eta_i[:] = eta_sampler((N, Nr, Nx)).astype(np.float32)          # (N, Nr, Nx, eta_dim)

    # Evaluate model
    g_theta_i[:] = model(x_loc, theta_i, eta_i)                     # (N, Nr, Nx, y_dim)

    # Sample outer loop data y
    y_i[:] = batch_normal_sample(g_theta_i, noise_cov)              # (N, Nr, Nx, y_dim)

    # Break input into batches
    if batch_size < 0:
        batch_size = Nx
    num_batches = int(np.floor(Nx / batch_size))
    if Nx % batch_size > 0:
        # Extra batch for remainder
        num_batches += 1

    # Parallel loop
    def parallel_func(idx, theta_i, eta_i, y_i, g_theta_i, likelihood, evidence):
        for curr_batch in range(num_batches):
            # Index the current batch
            start_idx = curr_batch * batch_size
            end_idx = start_idx + batch_size
            b_slice = slice(start_idx, end_idx) if end_idx < Nx else slice(start_idx, None)

            y_curr = y_i[idx, np.newaxis, :, b_slice, :]                               # (1, Nr, bs, y_dim)
            theta_curr = theta_i[idx, np.newaxis, :, b_slice, :]                       # (1, Nr, bs, theta_dim)
            if reuse_samples:
                # Compute evidence
                evidence[idx, :, b_slice] = np.mean(batch_normal_pdf(y_curr, g_theta_i[:, :, b_slice, :], noise_cov),
                                                    axis=0)
                # Compute likelihood
                g_theta_k = model(x_loc[b_slice, :], theta_curr, eta_i[:, :, b_slice, :])  # (N, Nr, bs, y_dim)
                likelihood[idx, :, b_slice] = np.mean(batch_normal_pdf(y_curr, g_theta_k, noise_cov), axis=0)
            else:
                # Compute evidence
                eta_j = eta_sampler((M1, Nr, batch_size)).astype(np.float32)            # (M1, Nr, bs, eta_dim)
                theta_j = theta_sampler((M1, Nr, batch_size)).astype(np.float32)        # (M1, Nr, bs, theta_dim)
                g_theta_j = model(x_loc[b_slice, :], theta_j, eta_j)                    # (M1, Nr, bs, y_dim)
                evidence[idx, :, b_slice] = np.mean(batch_normal_pdf(y_curr, g_theta_j, noise_cov), axis=0)

                # Compute likelihood
                eta_k = eta_sampler((M2, Nr, batch_size)).astype(np.float32)            # (M2, Nr, bs, eta_dim)
                g_theta_k = model(x_loc[b_slice, :], theta_curr, eta_k)                 # (M2, Nr, bs, y_dim)
                likelihood[idx, :, b_slice] = np.mean(batch_normal_pdf(y_curr, g_theta_k, noise_cov), axis=0)

    # Compute evidence p(y|d) and likelihood p(y|theta, d)
    Parallel(n_jobs=n_jobs, verbose=9)(delayed(parallel_func)(idx, theta_i, eta_i, y_i, g_theta_i, likelihood, evidence)
                                       for idx in range(N))

    # Expected information gain
    eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)    # (Nr, Nx)

    # If the likelihood or evidence are 0, np.log -> -inf, but eig -> 0
    # eig = np.nan_to_num(eig, nan=0, posinf=0, neginf=0)

    # Clean up
    del theta_i
    del eta_i
    del y_i
    del g_theta_i
    del likelihood
    del evidence
    os.remove(theta_fd.name)
    os.remove(eta_fd.name)
    os.remove(y_fd.name)
    os.remove(g_theta_fd.name)
    os.remove(likelihood_fd.name)
    os.remove(evidence_fd.name)

    return eig
