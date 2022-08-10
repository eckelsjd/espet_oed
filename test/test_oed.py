import numpy as np
import matplotlib.pyplot as plt
import time

from src.models import electrospray_current_model_cpu, nonlinear_model, linear_gaussian_model, linear_gaussian_eig
from src.models import electrospray_current_model_gpu
from src.nmc import eig_nmc_pm, eig_nmc
from src.utils import model_1d_batch


def test_linear_gaussian_model(N, M1=100, M2=100, Nx=50, var=0.01, reuse_samples=False):
    # Linear gaussian model example
    theta_sampler = lambda shape: np.random.randn(*shape, 1)
    eta_sampler = lambda shape: np.random.randn(*shape, 1)
    noise_cov = np.array([[var, 0], [0, var]])
    x_loc = np.linspace(0, 1, Nx).reshape((Nx, 1))
    eig_analytical = linear_gaussian_eig(x_loc, var)
    eig_estimate = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, linear_gaussian_model, N=N, M1=M1, M2=M2,
                              noise_cov=noise_cov, reuse_samples=reuse_samples, use_parallel=True)
    plt.figure()
    plt.plot(x_loc, eig_analytical, '-k')
    plt.plot(x_loc, eig_estimate, '--r')
    plt.xlabel('d')
    plt.ylabel('Expected information gain')
    plt.legend((r'Analytical $U(d)$', r'$\hat{U}^{NMC}(d)$'))
    plt.show()


def test_nonlinear_model(test_1d=True, test_2d=False):
    # Nonlinear model example, 1-d
    if test_1d:
        theta_sampler = lambda shape: np.random.rand(*shape, 1)
        N = 10000
        M = 1000
        Nx = 50
        x_loc = np.linspace(0, 1, Nx).reshape((Nx, 1))
        var = 1e-4
        t1 = time.time()
        eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, noise_cov=var, reuse_theta=False,
                      use_parallel=True)
        t2 = time.time()
        print(f'Total time: {t2-t1:.2f} s')

        plt.figure()
        plt.plot(x_loc, eig, '-k')
        plt.xlabel('d')
        plt.ylabel('Expected information gain')
        plt.show()

    # Nonlinear model example, 2-d
    if test_2d:
        theta_sampler = lambda shape: np.random.rand(*shape, 1)
        N = 10000
        M = 1000
        Nx = [20, 20]  # [Nx, Ny, Nz, ..., Nd] - discretization in each batch dimension
        loc = [np.linspace(0, 1, n) for n in Nx]
        pt_grids = np.meshgrid(*loc)
        x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
        var = 1e-4 * np.eye(2)
        t1 = time.time()
        eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, noise_cov=var, reuse_theta=True, use_parallel=True)
        t2 = time.time()
        print(f'Total time: {t2 - t1:.2f} s')

        # Reform grids
        grid_d1, grid_d2 = [x_loc[:, i].reshape((Nx[1], Nx[0])) for i in range(2)]  # reform grids
        eig_grid = eig.reshape((Nx[1], Nx[0]))

        # Plot results
        plt.figure()
        c = plt.contourf(grid_d1, grid_d2, eig_grid, 60, cmap='jet')
        plt.colorbar(c)
        plt.cla()
        plt.contour(grid_d1, grid_d2, eig_grid, 15, cmap='jet')
        plt.xlabel('$d_1$')
        plt.ylabel('$d_2$')
        plt.show()


def test_array_current_model(dim=1):
    # Array current model
    params = np.loadtxt('../data/Nr100_noPr_samples__2021_12_07T11_41_27.txt', dtype=np.float32, delimiter='\t',
                        skiprows=1)
    geoms = np.loadtxt('../data/mr_geoms.dat', dtype=np.float32, delimiter='\t')             # (Nr*Ne, geo_dim)
    emax_sim = np.loadtxt('../data/mr_geoms_tipE.dat', dtype=np.float32, delimiter='\t')     # (Nr*Ne, )

    def theta_sampler(shape):
        # Use preset samples from posterior
        ind = np.random.randint(0, params.shape[0], shape)
        samples = params[ind, :]  # (*, 3)
        return samples

    def beam_sampler(shape):
        qm_ratio = np.random.randn(*shape, 1) * 1.003e4 + 5.5e5
        return qm_ratio  # (*, 1)

    def prop_sampler(shape):
        k = np.random.rand(*shape, 1) * (1.39 - 1.147) + 1.147
        gamma = np.random.rand(*shape, 1) * (5.045e-2 - 5.003e-2) + 5.003e-2
        rho = np.random.rand(*shape, 1) * (1.284e3 - 1.28e3) + 1.28e3
        mu = np.random.rand(*shape, 1) * (3.416e-2 - 2.612e-2) + 2.612e-2
        props = np.concatenate((k, gamma, rho, mu), axis=-1)
        return props  # (*, 4)

    def subs_sampler(shape):
        # rpr = np.random.rand(1, Nr) * (8e-6 - 5e-6) + 5e-6
        rpr = np.ones((*shape, 1)) * 8e-6
        kappa = np.random.randn(*shape, 1) * 6.04e-15 + 1.51e-13
        subs = np.concatenate((rpr, kappa), axis=-1)
        return subs  # (*, 2)

    def eta_sampler(shape):
        # Load emitter data
        Ne = 300
        geo_dim = geoms.shape[1]
        sim_data = np.concatenate((geoms, emax_sim[:, np.newaxis]), axis=1)
        # g = geoms.reshape((Nr, 1, Ne * geo_dim))  # (Nr, 1, Ne*geo_dim)

        # Randomly sample emitter geometries that we have data for
        ind = np.random.randint(0, geoms.shape[0], (*shape, Ne))  # (*, Ne)
        geo_data = sim_data[ind, :]  # (*, Ne, geo_dim+1)
        geo_data = np.reshape(geo_data, (*shape, Ne * (geo_dim + 1)))

        # Sample other parameters
        subs = subs_sampler(shape)   # (*, 2)
        props = prop_sampler(shape)  # (*, 4)
        beams = beam_sampler(shape)  # (*, 1)

        # Combine
        eta = np.concatenate((subs, props, beams, geo_data), axis=-1)  # (*, 7 + Ne*(geo_dim+1))

        return eta.astype(np.float32)

    exp_data = np.loadtxt('../data/training_data.txt', dtype=np.float32, delimiter='\t')
    var = np.max(exp_data[2, :])

    # Set sample sizes
    N = 500
    Nx = 30
    M1 = 100
    M2 = 100
    n_jobs = 2

    if dim == 1:
        x_loc = np.linspace(800, 1840, Nx).reshape((Nx, 1))
        eig = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, electrospray_current_model_gpu, N=N, M1=M1, M2=M2,
                         noise_cov=var, reuse_samples=True, n_jobs=n_jobs, use_gpu=False)
        plt.figure()
        plt.plot(x_loc, eig, '-k')
        plt.xlabel('Voltage [V]')
        plt.ylabel('Expected information gain')
        plt.show()

    if dim == 2:
        Ngrid = [Nx, Nx]
        loc = [np.linspace(1000, 1840, n) for n in Ngrid]
        pt_grids = np.meshgrid(*loc)
        x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
        noise_cov = np.eye(2)*var
        model_func = lambda x, theta, eta: model_1d_batch(x, theta, eta, electrospray_current_model_cpu)
        eig = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, model_func, N=N, M1=M1, M2=M2,
                         noise_cov=noise_cov, reuse_samples=True, n_jobs=n_jobs, use_gpu=False)

        # Reform grids
        grid_d1, grid_d2 = [x_loc[:, i].reshape((Ngrid[1], Ngrid[0])) for i in range(2)]  # reform grids
        eig_grid = eig.reshape((Ngrid[1], Ngrid[0]))

        # Plot results
        plt.figure()
        c = plt.contourf(grid_d1, grid_d2, eig_grid, 60, cmap='jet')
        plt.colorbar(c)
        plt.cla()
        plt.contour(grid_d1, grid_d2, eig_grid, 15, cmap='jet')
        plt.xlabel('$Voltage 1 [V]$')
        plt.ylabel('$Voltage 2 [V]$')
        plt.show()


if __name__ == '__main__':
    # test_linear_gaussian_model(N=800, M1=800, M2=800, reuse_samples=False)
    test_array_current_model(dim=1)
