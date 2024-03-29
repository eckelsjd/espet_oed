# Nested monte carlo estimators

import numpy as np
from joblib import Parallel, delayed
from threading import Thread
import tempfile
import os
import time
import logging
import sys
sys.path.append('..')

from src.utils import batch_normal_sample, batch_normal_pdf, fix_input_shape, memory, log_memory_usage


# Nested monte carlo expected information gain estimator
def eig_nmc(x_loc, theta_sampler, model, N=100, M=100, noise_cov=1.0, reuse_samples=False, n_jobs=1, replicates=1,
            ppool=None):
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
    Nr = replicates
    theta_i = np.memmap(theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, theta_dim))
    y_i = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
    g_theta_i = np.memmap(g_theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
    evidence = np.memmap(evidence_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))

    # Sample model parameters
    theta_i[:] = theta_sampler((N, Nr, Nx)).astype(np.float32)          # (N, Nr, Nx, theta_dim)

    # Evaluate model
    g_theta_i[:] = model(x_loc, theta_i)                                # (N, Nr, Nx, y_dim)

    # Sample outer loop data y
    y_i[:] = batch_normal_sample(g_theta_i, noise_cov)                  # (N, Nr, Nx, y_dim)

    # Parallel loop
    def parallel_func(idx, y_i, g_theta_i, evidence):
        y_curr = y_i[idx, np.newaxis, :, :, :]                          # (1, Nr, Nx, y_dim)
        if reuse_samples:
            log_like = batch_normal_pdf(y_curr, g_theta_i, noise_cov, logpdf=True)  # (N, Nr, Nx)
            max_log_like = np.expand_dims(np.max(log_like, axis=0), axis=0)  # (1, Nr, Nx)
            evidence[idx, :, :] = -np.log(N) + np.squeeze(max_log_like, axis=0) + \
                                  np.log(np.sum(np.exp(log_like - max_log_like), axis=0))
            # evidence[idx, :, :] = np.mean(batch_normal_pdf(y_curr, g_theta_i, noise_cov), axis=0)
        else:
            theta_j = theta_sampler((M, Nr, Nx)).astype(np.float32)     # (M, Nr, Nx, theta_dim)
            g_theta_j = model(x_loc, theta_j)                           # (M, Nr, Nx, y_dim)
            log_like = batch_normal_pdf(y_curr, g_theta_j, noise_cov, logpdf=True)  # (M, Nr, Nx)
            max_log_like = np.expand_dims(np.max(log_like, axis=0), axis=0)  # (1, Nr, bs)
            evidence[idx, :, :] = -np.log(M) + np.squeeze(max_log_like, axis=0) + \
                                  np.log(np.sum(np.exp(log_like - max_log_like), axis=0))
            # evidence[idx, :, :] = np.mean(batch_normal_pdf(y_curr, g_theta_j, noise_cov), axis=0)

        if ((idx + 1) % 100) == 0 and n_jobs == 1:
            logging.info(f'Samples processed: {idx + 1} out of {N}')

    # Compute evidence p(y|d)
    if ppool is None:
        Parallel(n_jobs=n_jobs, verbose=9)(delayed(parallel_func)(idx, y_i, g_theta_i, evidence) for idx in range(N))
    else:
        ppool(delayed(parallel_func)(idx, y_i, g_theta_i, evidence) for idx in range(N))

    # Compute likelihood
    likelihood = batch_normal_pdf(y_i, g_theta_i, noise_cov, logpdf=True)        # (N, Nr, Nx)

    # Expected information gain
    likelihood = np.nan_to_num(likelihood, neginf=np.nan, posinf=np.nan)
    evidence = np.nan_to_num(evidence, neginf=np.nan, posinf=np.nan)
    eig = np.nanmean(likelihood - evidence, axis=0)  # (Nr, Nx)
    # eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)    # (Nr, Nx)

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
# @memory(percentage=0.95)
def eig_nmc_pm(x_loc, theta_sampler, eta_sampler, model, N=100, M1=100, M2=100, noise_cov=np.asarray(1.0),
               reuse_samples=False, n_jobs=1, batch_size=-1, replicates=1, ppool=None):
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

    try:
        # Allocate space for data
        Nr = replicates
        theta_i = np.memmap(theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, theta_dim))
        eta_i = np.memmap(eta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, eta_dim))
        y_i = np.memmap(y_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
        g_theta_i = np.memmap(g_theta_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx, y_dim))
        likelihood = np.memmap(likelihood_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))
        evidence = np.memmap(evidence_fd.name, dtype='float32', mode='r+', shape=(N, Nr, Nx))

        # Start memory logging
        # daemon = Thread(target=log_memory_usage, args=(30,), daemon=True, name="Memory logger")
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
            t1 = time.time()
            for curr_batch in range(num_batches):
                # Index the current batch
                start_idx = curr_batch * batch_size
                end_idx = start_idx + batch_size
                b_slice = slice(start_idx, end_idx) if end_idx < Nx else slice(start_idx, None)

                y_curr = y_i[idx, np.newaxis, :, b_slice, :]                               # (1, Nr, bs, y_dim)
                theta_curr = theta_i[idx, np.newaxis, :, b_slice, :]                       # (1, Nr, bs, theta_dim)
                if reuse_samples:
                    # Compute evidence
                    log_like = batch_normal_pdf(y_curr, g_theta_i[:, :, b_slice, :], noise_cov, logpdf=True)  # (M1, Nr, bs)
                    max_log_like = np.expand_dims(np.max(log_like, axis=0), axis=0)  # (1, Nr, bs)
                    evidence[idx, :, b_slice] = -np.log(N) + np.squeeze(max_log_like, axis=0) + \
                                                np.log(np.sum(np.exp(log_like - max_log_like), axis=0))

                    # Compute likelihood
                    g_theta_k = model(x_loc[b_slice, :], theta_curr, eta_i[:, :, b_slice, :])  # (N, Nr, bs, y_dim)
                    log_like = batch_normal_pdf(y_curr, g_theta_k, noise_cov, logpdf=True)
                    max_log_like = np.expand_dims(np.max(log_like, axis=0), axis=0)
                    likelihood[idx, :, b_slice] = -np.log(N) + np.squeeze(max_log_like, axis=0) + \
                                                  np.log(np.sum(np.exp(log_like - max_log_like), axis=0))
                else:
                    # Compute evidence
                    eta_j = eta_sampler((M1, Nr, batch_size)).astype(np.float32)            # (M1, Nr, bs, eta_dim)
                    theta_j = theta_sampler((M1, Nr, batch_size)).astype(np.float32)        # (M1, Nr, bs, theta_dim)
                    g_theta_j = model(x_loc[b_slice, :], theta_j, eta_j)                    # (M1, Nr, bs, y_dim)
                    log_like = batch_normal_pdf(y_curr, g_theta_j, noise_cov, logpdf=True)  # (M1, Nr, bs)
                    max_log_like = np.expand_dims(np.max(log_like, axis=0), axis=0)         # (1, Nr, bs)
                    evidence[idx, :, b_slice] = -np.log(M1) + np.squeeze(max_log_like, axis=0) + \
                                                np.log(np.sum(np.exp(log_like - max_log_like), axis=0))

                    # Compute likelihood
                    eta_k = eta_sampler((M2, Nr, batch_size)).astype(np.float32)            # (M2, Nr, bs, eta_dim)
                    g_theta_k = model(x_loc[b_slice, :], theta_curr, eta_k)                 # (M2, Nr, bs, y_dim)
                    log_like = batch_normal_pdf(y_curr, g_theta_k, noise_cov, logpdf=True)
                    max_log_like = np.expand_dims(np.max(log_like, axis=0), axis=0)
                    likelihood[idx, :, b_slice] = -np.log(M2) + np.squeeze(max_log_like, axis=0) + \
                                                  np.log(np.sum(np.exp(log_like - max_log_like), axis=0))

            if idx % 500 == 0:
                print(f'Parallel idx {idx}: {time.time()-t1:.02} s')

        # Compute evidence p(y|d) and likelihood p(y|theta, d)
        if ppool is None:
            Parallel(n_jobs=n_jobs, verbose=9)(delayed(parallel_func)(idx, theta_i, eta_i, y_i, g_theta_i,
                                                                      likelihood, evidence) for idx in range(N))
        else:
            ppool(delayed(parallel_func)(idx, theta_i, eta_i, y_i, g_theta_i, likelihood, evidence) for idx in range(N))

        # Expected information gain
        likelihood = np.nan_to_num(likelihood, neginf=np.nan, posinf=np.nan)
        evidence = np.nan_to_num(evidence, neginf=np.nan, posinf=np.nan)
        eig = np.nanmean(likelihood - evidence, axis=0)  # (Nr, Nx)

    finally:
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
