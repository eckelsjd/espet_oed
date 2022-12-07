import numpy as np
from pathlib import Path
from joblib import Parallel
import logging
import sys
import time

sys.path.append('..')
from src.utils import electrospray_samplers
from src.models import electrospray_current_model
from src.nmc import eig_nmc_pm


def test_nmc_n2m(iter, N_to_M):
    """Test nmc estimators on electrospray model"""
    # Testing parameters
    N_MC = 50                                   # number of MC replicates
    N_cost = 6                                  # number of total costs (i.e. number of model evaluations)
    cost = np.floor(np.logspace(4, 6, N_cost))

    # EIG run parameters
    Nx = 50
    d = np.linspace(800, 1845, Nx)
    theta_sampler, eta_sampler = electrospray_samplers()
    model_func = electrospray_current_model
    exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
    gamma = np.mean(exp_data[2, :])

    # Allocate space
    eig_store = np.zeros((1, N_cost, N_MC, Nx), dtype=np.float32)
    real_cost = np.zeros((1, N_cost))  # due to rounding issues, actual cost will be different

    # Loop over each cost
    for i, total_cost in enumerate(cost):
        # Cost = 2*N*M, N/M = ratio, for NMC estimator (assuming M1=M2=M)
        if N_to_M >= 1:
            M = int(np.sqrt(total_cost / (2*N_to_M)))
            N = int(M * N_to_M)
        else:
            N = int(np.sqrt((total_cost * N_to_M) / 2))
            M = int(N / N_to_M)
        print(f'N/M: {N_to_M}, Total cost: {total_cost}, N: {N}, M: {M}, 2NM = {2*N*M}')
        real_cost[0, i] = 2*N*M

        # Run NMC estimator
        eig_estimate = np.zeros((N_MC, Nx))
        t1 = time.time()
        with Parallel(n_jobs=-1, verbose=9) as ppool:
            for k in range(N_MC):
                eig = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M, noise_cov=gamma,
                                 reuse_samples=False, n_jobs=-1, batch_size=1, replicates=1, ppool=ppool)
                eig_estimate[k, :] = np.squeeze(eig, axis=0)
        t2 = time.time()
        print(f'Total time for N={N} M={M} C={total_cost} N2M={N_to_M}: {t2 - t1:.02} s')

        # Filter arithmetic underflow
        eig_store[0, i, :, :] = np.nan_to_num(eig_estimate, posinf=np.nan, neginf=np.nan)

    # Save results
    np.savez(Path('../results')/f'nmc_electrospray_n2m_{N_to_M}_{iter}.npz', d=d, eig=eig_store, cost=real_cost,
             N2M=N_to_M)


if __name__ == '__main__':
    """sys.argv[1] = iteration number, sys.argv[2] = N2M ratio"""
    logging.basicConfig(level=logging.INFO)
    test_nmc_n2m(int(sys.argv[1]), float(sys.argv[2]))
