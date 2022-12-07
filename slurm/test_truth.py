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


def test_nmc_truth(iter):
    """Test nmc estimators on electrospray model"""
    # Testing parameters
    Nr = 15
    N = 3000
    M = 2000
    bs = 1
    Nx = 50
    d = np.linspace(800, 1845, Nx)
    theta_sampler, eta_sampler = electrospray_samplers()
    model_func = electrospray_current_model
    exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
    gamma = np.mean(exp_data[2, :])

    t1 = time.time()
    eig_estimate = np.zeros((Nr, Nx))
    with Parallel(n_jobs=-1, verbose=9) as ppool:
        for i in range(Nr):
            eig = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M, noise_cov=gamma,
                             reuse_samples=False, n_jobs=-1, batch_size=bs, replicates=1, ppool=ppool)
            eig_estimate[i, :] = np.squeeze(eig, axis=0)
    t2 = time.time()
    print(f'Total time for N={N} M={M} Nr={Nr} bs={bs}: {t2 - t1:.02} s')
    np.savez(Path('../results') / f'nmc_electrospray_truth_{iter}.npz', d=d, eig_truth=eig_estimate)


if __name__ == '__main__':
    """sys.argv[1] = iteration number, sys.argv[2] = N2M ratio"""
    logging.basicConfig(level=logging.INFO)
    test_nmc_truth(int(sys.argv[1]))
