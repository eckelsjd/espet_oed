import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel
import logging
import sys
import time

sys.path.append('..')
from src.utils import electrospray_samplers, ax_default
from src.models import electrospray_current_model
from src.nmc import eig_nmc_pm


def test_nmc():
    """Test nmc estimators on electrospray model"""
    # EIG run parameters
    Nx = 20
    d = np.linspace(800, 1845, Nx)
    theta_sampler, eta_sampler = electrospray_samplers()
    model_func = electrospray_current_model
    exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
    gamma = np.mean(exp_data[2, :])

    # Cost = 2*N*M, N/M = ratio, for NMC estimator (assuming M1=M2=M)
    total_cost = 10 ** 5
    N_to_M = 1
    if N_to_M >= 1:
        M = int(np.sqrt(total_cost / (2*N_to_M)))
        N = int(M * N_to_M)
    else:
        N = int(np.sqrt((total_cost * N_to_M) / 2))
        M = int(N / N_to_M)
    print(f'N/M: {N_to_M}, Total cost: {total_cost}, N: {N}, M: {M}, 2NM = {2*N*M}')

    # Run NMC estimator
    compute = False
    if compute:
        eig = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M, noise_cov=gamma,
                         reuse_samples=False, n_jobs=-1, batch_size=1, replicates=1)
        # Save results
        np.savez(Path('../results') / f'electrospray_test_{N_to_M}_{total_cost}.npz', d=d, eig=eig, cost=total_cost,
                 N2M=N_to_M)
    else:
        data = np.load(str(Path('../results')/f'electrospray_test_{N_to_M}_{total_cost}.npz'))
        eig = data['eig']
        d = data['d']

    sl = slice(0, None)
    fig, ax = plt.subplots()
    ax.plot(d[sl], np.squeeze(eig)[sl], '-k')
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_nmc()
