import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from joblib import Parallel
import logging
import time
import sys
sys.path.extend('..')
from src.utils import linear_eig, ax_default, get_cycle, fix_input_shape, electrospray_samplers
from src.models import linear_gaussian_model, custom_nonlinear, electrospray_current_model_cpu
from src.nmc import eig_nmc_pm
from src.lg import eig_lg


def test_lg(model='nonlinear'):
    Nx = 50

    if model == 'nonlinear':
        theta_mean = np.array([0.5])
        eta = np.array([0.5])
        theta_var = 0.25 ** 2
        noise_cov = 0.01
        x = np.linspace(0, 1, Nx)
        model_func = custom_nonlinear

        # Compute eig
        eig_estimate = eig_lg(x, model_func, theta_mean, theta_var, eta, noise_cov)

    # Compare to ground truth
    data = np.load(str(Path('../results') / f'nmc_{model}.npz'))
    eig_truth = data['eig_truth']  # (1, Nx)
    mse = np.mean((eig_estimate - eig_truth)**2)
    print(f'MSE of LG estimate: {mse}')

    fig, ax = plt.subplots()
    ax.plot(x, np.squeeze(eig_truth), '-k', label=r'Ground truth')
    ax.plot(x, np.squeeze(eig_estimate), '--r', label=r'Linear Gaussian')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=-0.01)
    ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain', legend=True)
    fig.set_size_inches(4.8, 3.6)
    plt.tight_layout()
    fig.savefig(str(Path('../results/figs') / 'nonlinear_lg_eig.png'), dpi=300, format='png')
    plt.show()


def test_nmc(model='linear'):
    """Test nmc estimators on the specified model"""
    # Testing parameters
    N_MC = 50                                   # number of MC replicates
    N_to_M = np.array([0.01, 0.1, 1, 10, 100])  # N:M sample ratios
    N_est = N_to_M.shape[0]                     # number of estimators to compare
    N_cost = 6                                  # number of total costs (i.e. number of model evaluations)
    cost = np.floor(np.logspace(4, 6, N_cost))

    # Model-specific (Specify d, Nx, theta_sampler, eta_sampler, eig_truth, gamma, model_func)
    Nx = 0; d=None; theta_sampler=None; eta_sampler=None; eig_truth=None; gamma=0; model_func=None
    if model == 'linear':
        Nx = 50
        d = np.linspace(0, 1, Nx)
        theta_sampler = lambda shape: np.random.randn(*shape, 1)
        eta_sampler = lambda shape: np.random.randn(*shape, 1)
        model_func = linear_gaussian_model

        # Generate ground truth comparison
        y_dim = 2
        A = np.zeros((Nx, y_dim, 1))
        A[:, 0, 0] = d
        sigma = np.array([[[1]]])
        eps_var = 0.01
        gamma = eps_var * np.eye(y_dim)
        eig_truth = linear_eig(A, sigma, np.expand_dims(gamma, axis=0))  # (Nx,)
        eig_truth = np.expand_dims(eig_truth, axis=0)  # (1, Nx)

    elif model == 'nonlinear':
        Nx = 50
        d = np.linspace(0, 1, Nx)
        std = 0.25; mean = 0.5
        theta_sampler = lambda shape: np.random.randn(*shape, 1)*std + mean
        eta_sampler = lambda shape: np.random.randn(*shape, 1)*std + mean
        model_func = custom_nonlinear

        # Generate ground truth comparison (high-fidelity NMC)
        N = 1000
        M = 10000
        Nr = 50
        gamma = 0.01
        eig_estimate = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M,
                                  noise_cov=gamma, reuse_samples=False, n_jobs=-1, replicates=Nr)  # (N_MC, Nx)
        eig_truth = np.nanmean(eig_estimate, axis=0).reshape((1, Nx))

    elif model == 'electrospray':
        Nx = 50
        d = np.linspace(800, 1845, Nx)
        theta_sampler, eta_sampler = electrospray_samplers()
        model_func = electrospray_current_model_cpu
        N = 256  # 2000
        M = 10  # 5000
        Nr = 10
        bs = 5
        exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
        gamma = np.mean(exp_data[2, :])
        logging.info('Starting main electrospray ground truth')
        t1 = time.time()
        eig_estimate = np.zeros((Nr, Nx))
        with Parallel(n_jobs=-1, verbose=9) as ppool:
            for i in range(Nr):
                eig = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M, noise_cov=gamma,
                                 reuse_samples=False, n_jobs=-1, batch_size=bs, replicates=1, ppool=ppool)
                eig_estimate[i, :] = np.squeeze(eig, axis=0)
        t2 = time.time()
        logging.info(f'Total time for N={N} M={M} Nr={Nr} bs={bs}: {t2-t1:.02} s')
        eig_truth = np.nanmean(eig_estimate, axis=0).reshape((1, Nx))
        np.savez(Path('../results') / f'nmc_{model}_eig_truth.npz', d=d, eig_truth=eig_estimate)
        eig_lb_pm = np.nanpercentile(eig_estimate, 5, axis=0)
        eig_med_pm = np.nanpercentile(eig_estimate, 50, axis=0)
        eig_ub_pm = np.nanpercentile(eig_estimate, 95, axis=0)

        fig, ax = plt.subplots()
        ax.plot(d, eig_med_pm, '-r', label=r'Marginal p($\theta$)')
        ax.fill_between(d, eig_lb_pm, eig_ub_pm, alpha=0.3, edgecolor=(0.5, 0.5, 0.5), facecolor='red')
        ax.set_ylim(bottom=-0.01)
        ax_default(ax, xlabel='Operating condition $d$', ylabel='Expected information gain')
        fig.set_size_inches(4.8, 3.6)
        fig.tight_layout()
        fig.savefig(str(Path('../results/figs') / f'nmc_{model}_eig_truth.png'), dpi=300, format='png')
        return

    else:
        raise NotImplementedError('Not working with whatever model you specified')

    # Allocate space
    eig_store = np.zeros((N_est, N_cost, N_MC, Nx), dtype=np.float32)
    mse_store = np.zeros((N_est, N_cost, N_MC), dtype=np.float32)
    real_cost = np.zeros((N_est, N_cost))  # due to rounding issues, actual cost will be different

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
            eig_estimate = np.zeros((N_MC, Nx))
            with Parallel(n_jobs=-1, verbose=9) as ppool:
                for i in range(N_MC):
                    eig = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M, noise_cov=gamma,
                                     reuse_samples=False, n_jobs=-1, batch_size=5, replicates=1, ppool=ppool)
                    eig_estimate[i, :] = np.squeeze(eig, axis=0)
            # eig_estimate = eig_nmc_pm(d, theta_sampler, eta_sampler, model_func, N=N, M1=M, M2=M,
            #                           noise_cov=gamma, reuse_samples=False, n_jobs=-1, replicates=N_MC)  # (N_MC, Nx)
            # Filter arithmetic underflow
            eig_store[i, j, :, :] = np.nan_to_num(eig_estimate, posinf=np.nan, neginf=np.nan)
            mse_store[i, j, :] = np.nanmean((eig_store[i, j, :, :] - eig_truth) ** 2, axis=-1)  # (N_MC,)

    # Save results
    np.savez(Path('../results')/f'nmc_{model}.npz', d=d, eig=eig_store, mse=mse_store, eig_truth=eig_truth,
             cost=real_cost, N2M=N_to_M)


def plot_nmc(model='linear', estimator='nmc'):
    # Load data from npz file
    data = np.load(str(Path('../results')/f'{estimator}_{model}.npz'))

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
            axs[i, j].plot(d, eig_med[i, j, :], '-r', label='Estimator')
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
    fig.savefig(str(Path('../results/figs') / f'{estimator}_{model}_N2M_cost_eig.png'), dpi=100, format='png')
    plt.show()

    # Plot MSE log plot
    # mse_store = data['mse']  # (N_est, N_cost, N_MC)
    # # mse = np.nanmean((eig_store - eig_truth.reshape((1, 1, 1, Nx)))**2, axis=(-2, -1))  # (N_est, N_cost)
    # for i, nm_ratio in enumerate(N_to_M):
    #     # Get N:M ratio label
    #     label = f"{int(nm_ratio)}:1" if nm_ratio >= 1 else f"1:{int(1/nm_ratio)}"
    #     mean_mse = np.mean(mse_store[i, :, :], axis=-1)    # (N_cost,)
    #     var = np.var(mse_store[i, :, :], axis=-1, ddof=1)  # (N_cost,)
    #     std_err = np.sqrt(var/N_MC)
    #     ax.errorbar(cost[i, :], mean_mse, yerr=1.96*std_err, linestyle='-',
    #                  markersize=0, linewidth=1.5, capsize=2, label=label)

    # Plot Bias, variance, MSE
    bias = np.nanmean(eig_store - eig_truth.reshape((1, 1, 1, Nx)), axis=-2)    # (N_est, N_cost, Nx)
    var = np.nanvar(eig_store, axis=-2)                                         # (N_est, N_cost, Nx)
    mse = bias ** 2 + var                                                       # (N_est, N_cost, Nx)
    fig, axs = plt.subplots(1, 3, sharey='row')
    colors = get_cycle("tab10", N=6)
    for i in range(3):
        axs[i].set_prop_cycle(color=colors.by_key()['color'], marker=['o', 'D', '^', 's', '*', 'X'])
        axs[i].set_xlabel(r'Number of model evaluations')
        axs[i].grid()
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
    axs[1].plot(np.NaN, np.NaN, '-', color='none', label='N:M')
    for i, nm_ratio in enumerate(N_to_M):
        # Plot bias
        bias_mean = np.nanmean(np.abs(bias), axis=-1)   # (N_est, N_cost)
        nx_var = np.nanvar(np.abs(bias), axis=-1)       # (N_est, N_cost)
        std_err = np.sqrt(nx_var / Nx)
        axs[0].set_ylabel(r'Estimator bias')
        axs[0].errorbar(cost[i, :], bias_mean[i, :], yerr=1.96 * std_err[i, :], linestyle='-',
                        markersize=0, linewidth=1.3, capsize=2)
        # axs[0].plot(cost[i, :], np.abs(bias_mean[i, :]), linestyle='-', markersize=1, linewidth=1.3)

        # Plot variance
        label = f"{int(nm_ratio)}:1" if nm_ratio >= 1 else f"1:{int(1 / nm_ratio)}"
        var_mean = np.nanmean(var, axis=-1)     # (N_est, N_cost)
        nx_var = np.nanvar(var, axis=-1)        # (N_est, N_cost)
        std_err = np.sqrt(nx_var / Nx)
        axs[1].set_ylabel(r'Estimator variance')
        axs[1].errorbar(cost[i, :], var_mean[i, :], yerr=1.96 * std_err[i, :], linestyle='-',
                        markersize=0, linewidth=1.3, capsize=2, label=label)
        # Plot MSE
        mse_mean = np.nanmean(mse, axis=-1)     # (N_est, N_cost)
        nx_var = np.nanvar(mse, axis=-1)        # (N_est, N_cost)
        std_err = np.sqrt(nx_var / Nx)
        axs[2].set_ylabel(r'Estimator MSE')
        axs[2].errorbar(cost[i, :], mse_mean[i, :], yerr=1.96 * std_err[i, :], linestyle='-',
                        markersize=0, linewidth=1.3, capsize=2)

    axs[0].set_ylim(bottom=3e-5)
    fig.set_size_inches(9, 3.5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    leg = fig.legend(loc='upper center', fancybox=True, ncol=N_est+1)
    frame = leg.get_frame()
    frame.set_edgecolor('k')
    frame.set_facecolor([1, 1, 1, 1])
    fig.savefig(str(Path('../results/figs') / f'{estimator}_{model}_mse.png'), dpi=150, format='png')
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_nmc(model='electrospray')
    # plot_nmc(model='nonlinear')
    # test_lg(model='nonlinear')
