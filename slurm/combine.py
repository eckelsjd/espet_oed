import numpy as np
from pathlib import Path


def combine_truth():
    Nr = 10
    Nx = 50
    eig_truth = np.zeros((Nr*5, Nx))
    for idx in range(5):
        start_idx = idx * Nr
        end_idx = start_idx + Nr
        data = np.load(str(Path('../results') / f'nmc_electrospray_truth_{idx+1}.npz'))
        eig_truth[start_idx:end_idx, :] = data['eig_truth']

    np.savez(Path('../results') / f'nmc_electrospray_truth.npz', d=data['d'], eig_truth=eig_truth)


def combine_mse():
    n2m = np.array([0.01, 0.1, 1., 10., 100.])
    N_cost = 6
    N_est = n2m.shape[0]
    N_MC = 100
    Nx = 50
    eig_store = np.zeros((N_est, N_cost, N_MC, Nx))
    real_cost = np.zeros((N_est, N_cost))
    truth_data = np.load(str(Path('../results') / f'nmc_electrospray_truth.npz'))
    eig_truth = truth_data['eig_truth']

    Nr = 25
    for i, nm_ratio in enumerate(n2m):
        for idx in range(4):
            start_idx = idx*Nr
            end_idx = start_idx + Nr
            data = np.load(str(Path('../results')/f'nmc_electrospray_n2m_{float(nm_ratio)}_{idx+1}.npz'))
            eig_store[i, :, start_idx:end_idx, :] = data['eig']
            real_cost[i, :] = data['cost']

    np.savez(Path('../results') / f'nmc_electrospray.npz', d=truth_data['d'], eig_truth=eig_truth, eig=eig_store,
             cost=real_cost, N2M=n2m)


if __name__ == '__main__':
    combine_mse()
    # combine_truth()
