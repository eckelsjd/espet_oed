import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

from src.utils import linear_eig, ax_default, get_cycle
from src.models import linear_gaussian_model
from src.nmc import eig_nmc_pm


def test_nmc_linear():
    """Test nmc estimators on the linear Gaussian model"""
    # Simulation parameters
    N_MC = 100  # number of MC replicates
    N_to_M = np.array([0.01, 0.1, 1, 10, 100])
    # N_to_M = np.array([1/100, 1/10, 1, 10, 100])  # N:M sample ratios
    N_est = N_to_M.shape[0]  # number of estimators to compare
    Nx = 50  # number of input locations
    d = np.linspace(0, 1, Nx)
    N_cost = 6  # number of total costs (i.e. number of model evaluations)
    cost = np.floor(np.logspace(4, 6, N_cost))

    # Allocate space
    eig_store = np.zeros((N_est, N_cost, N_MC, Nx), dtype=np.float32)
    mse_store = np.zeros((N_est, N_cost, N_MC), dtype=np.float32)
    real_cost = np.zeros((N_est, N_cost))  # due to rounding issues, actual cost will be different

    # Generate ground truth comparison
    y_dim = 2
    A = np.zeros((Nx, y_dim, 1))
    A[:, 0, 0] = d
    sigma = np.array([[[1]]])
    eps_var = 0.01
    gamma = eps_var * np.eye(y_dim)
    eig_truth = linear_eig(A, sigma, np.expand_dims(gamma, axis=0))  # (Nx,)
    eig_truth = np.expand_dims(eig_truth, axis=0)   # (1, Nx)

    # Loop over each estimator
    for i, nm_ratio in enumerate(N_to_M):
        # Loop over each cost
        for j, total_cost in enumerate(cost):
            # Cost = 2*N*M, N/M = ratio, for NMC estimator (assuming M1=M2=M)
            if nm_ratio >= 1:
                M = int(np.sqrt(total_cost / (2*nm_ratio)))
                N = int(M * nm_ratio)
            else:
                N = int(np.sqrt((total_cost * nm_ratio) / 2))
                M = int(N / nm_ratio)
            print(f'N/M: {nm_ratio}, Total cost: {total_cost}, N: {N}, M: {M}, 2NM = {2*N*M}')
            real_cost[i, j] = 2*N*M

            # Run NMC estimator
            theta_sampler = lambda shape: np.random.randn(*shape, 1)
            eta_sampler = lambda shape: np.random.randn(*shape, 1)
            eig_estimate = eig_nmc_pm(d, theta_sampler, eta_sampler, linear_gaussian_model, N=N, M1=M, M2=M,
                                      noise_cov=gamma, reuse_samples=False, n_jobs=1, replicates=N_MC)  # (N_MC, Nx)
            # Filter arithmetic underflow
            eig_store[i, j, :, :] = np.nan_to_num(eig_estimate, posinf=np.nan, neginf=np.nan)
            mse_store[i, j, :] = np.nanmean((eig_store[i, j, :, :] - eig_truth) ** 2, axis=-1)  # (N_MC,)

    # Save results
    np.savez(Path('../results/nmc_linear.npz'), d=d, eig=eig_store, mse=mse_store, eig_truth=eig_truth,
             cost=real_cost, N2M=N_to_M)


def plot_nmc_linear():
    # Load data from npz file
    data = np.load(str(Path('../results/nmc_linear.npz')))

    eig_store = data['eig']
    eig_truth = data['eig_truth']
    N_to_M = data['N2M']
    d = data['d']
    cost = data['cost']
    N_est, N_cost, N_MC, Nx = eig_store.shape

    # Percentiles over MC replicates
    eig_lb = np.nanpercentile(eig_store, 5, axis=-2)
    eig_med = np.nanpercentile(eig_store, 50, axis=-2)
    eig_ub = np.nanpercentile(eig_store, 95, axis=-2)  # (N_est, N_cost, Nx)

    # Plot EIG results
    fig, axs = plt.subplots(N_est, N_cost, sharex='col', sharey='row')
    for i, nm_ratio in enumerate(N_to_M):
        # Loop over each cost
        for j in range(N_cost):
            axs[i, j].plot(d, eig_med[i, j, :], '-r', label='NMC')
            axs[i, j].fill_between(d, eig_lb[i, j, :], eig_ub[i, j, :], alpha=0.3, edgecolor=(0.5, 0.5, 0.5),
                                   facecolor='r')
            axs[i, j].plot(d, eig_truth[0, :], '-k', label='Exact')
            ax_default(axs[i, j], legend=False)
            axs[i, j].set_xbound(lower=np.min(d), upper=np.max(d))
            axs[i, j].set_ybound(lower=0)
            axs[i, j].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[i, j].grid()

            # Axis labels
            if i == N_est - 1:
                xlabel = f"Cost = $10^{{{np.log10(cost[i, j]):.1f}}}$"
                axs[i, j].set_xlabel(xlabel)
            if j == 0:
                ylabel = f"N:M = {int(nm_ratio)}:1" if nm_ratio >= 1 else f"N:M = 1:{int(1/nm_ratio)}"
                axs[i, j].set_ylabel(ylabel)

            # Legend in top right subplot
            if i == 0 and j == N_cost - 1:
                leg = axs[i, j].legend(fancybox=True)
                frame = leg.get_frame()
                frame.set_edgecolor('k')
                frame.set_facecolor([1, 1, 1, 1])

    fig.text(0.5, 0.02, r'Operating condition $d$', ha='center', fontweight='bold')
    fig.text(0.02, 0.5, r'Expected information gain', va='center', fontweight='bold', rotation='vertical')
    fig.set_size_inches(N_cost*2.5, N_est*2.5)
    fig.tight_layout(pad=3, w_pad=1, h_pad=1)
    plt.show()

    # Plot MSE log plot
    mse_store = data['mse']  # (N_est, N_cost, N_MC)
    fig, ax = plt.subplots()
    colors = get_cycle("tab10", N=6)
    ax.set_prop_cycle(color=colors.by_key()['color'], marker=['o', 'D', '^', 's', '*', 'X'])
    ax.plot(np.NaN, np.NaN, '-', color='none', label='N:M')
    for i, nm_ratio in enumerate(N_to_M):
        # Get N:M ratio label
        label = f"{int(nm_ratio)}:1" if nm_ratio >= 1 else f"1:{int(1/nm_ratio)}"
        mean_mse = np.mean(mse_store[i, :, :], axis=-1)    # (N_cost,)
        var = np.var(mse_store[i, :, :], axis=-1, ddof=1)  # (N_cost,)
        std_err = np.sqrt(var/N_MC)
        plt.errorbar(cost[i, :], mean_mse, yerr=std_err, linestyle='-',
                     markersize=4, linewidth=1.5, capsize=2, label=label)
    ax.set_xlabel(r'Model evaluations ($C=2NM$)')
    ax.set_ylabel(r'Estimator MSE')
    ax.grid()
    fig.set_size_inches(4.5, 3)
    fig.subplots_adjust(right=0.8)
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fancybox=True)
    frame = leg.get_frame()
    frame.set_edgecolor('k')
    frame.set_facecolor([1, 1, 1, 1])
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_nmc_linear()
    plot_nmc_linear()
