import numpy as np
import matplotlib.pyplot as plt
from lib.SciTech2022_inference import current_model
# from lib.esi_surrogate import forward
from joblib import Parallel, delayed
from pathlib import Path
import os
import shutil
import time


def batch_normal_pdf(x, mu, cov):
    """
    Compute the multivariate normal pdf at each x location.
    Dimensions
    ----------
    d: dimension of the problem
    *: any arbitrary shape (a1, a2, ...)
    Parameters
    ----------
    x: (*, d) location to compute the multivariate normal pdf
    mu: (*, d) mean values to use at each x location
    cov: (d, d) covariance matrix, assumed same at all locations
    Returns
    -------
    pdf: (*) the multivariate normal pdf at each x location
    """
    # Make some checks on input
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    if len(mu.shape) == 1:
        mu = mu[:, np.newaxis]

    assert cov.shape[0] == cov.shape[1] == dim
    assert x.shape[-1] == mu.shape[-1] == dim

    # Normalizing constant (scalar)
    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))

    # In exponential
    diff = x - mu  # can broadcast x - mu with x: (1, Nr, Nx, d) and mu: (Ns, Nr, Nx, d)
    diff_col = diff.reshape((*diff.shape, 1))                       # (Ns, Nr, Nx, d, 1)
    mat1 = np.linalg.inv(cov) @ diff_col                            # (d, d) x (*, d, 1) = (*, d, 1) broadcast matmult
    diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (Ns, Nr, Nx, 1, d)
    inexp = np.squeeze(diff_row @ mat1, axis=(-1, -2))              # (*, 1, d) x (*, d, 1) = (*, 1, 1)

    # Compute the pdf
    pdf = preexp * np.exp(-1/2 * inexp)
    return pdf.astype(np.float32)


def batch_normal_sample(mean, cov, size: "tuple | int" = ()):
    """
    Batch sample multivariate normal distributions.
    https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
    Arguments:
        mean: expected values of shape (…M, D)
        cov: covariance matrices of shape (…M, D, D)
        size: additional batch shape (…B)
    Returns: samples from the multivariate normal distributions
             shape: (…B, …M, D)
    """
    # Make some checks on input
    mean = np.atleast_1d(mean)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(mean.shape) == 1:
        mean = mean[:, np.newaxis]

    assert cov.shape[0] == cov.shape[1] == dim
    assert mean.shape[-1] == dim

    size = (size, ) if isinstance(size, int) else tuple(size)
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    X = np.random.standard_normal((*shape, 1)).astype(np.float32)
    L = np.linalg.cholesky(cov)
    return (L @ X).reshape(shape) + mean


def fix_input_shape(x):
    """Make input shape: (Nx, xdim)
    Nx: number of experimental locations (inputs x) to evaluate at
    xdim: dimension of a single experimental input x
    """
    x = np.atleast_1d(x).astype(np.float32)
    if len(x.shape) == 1:
        # Assume one x dimension
        x = x[:, np.newaxis]
    elif len(x.shape) != 2:
        raise Exception('Incorrect input dimension')
    return x


# Test estimator on Linear-Gaussian model
def linear_gaussian_model(x, theta, eta):
    """
    Linear Gaussian model with analytical solution for eig:
    [y1, y2] = [d, 0; 0, 1-d] * [theta, eta]
    Parameters
    ----------
    x: (Nx, x_dim) input locations, or operating conditions
    theta: (*, Nx, theta_dim) model parameters
    eta: (*, Nx, eta_dim) nuisance parameters

    Nx: Number of input locations
    x_dim: Dimension of operating conditions
    y_dim: Dimension of output
    theta_dim: Dimension of model parameters
    eta_dim: Dimension of nuisance parameters

    Returns
    -------
    g_theta: (*, Nx, y_dim) model output
    """
    Nx, x_dim = x.shape
    theta_shape = theta.shape[:-2]
    theta_dim = theta[-1]
    eta_shape = eta.shape[:-2]
    eta_dim = eta[-1]
    assert theta.shape[-2] == eta.shape[-2] == Nx
    y_dim = 2

    x = x.reshape((1,) * len(eta_shape) + (Nx, x_dim))  # (...1, Nx, x_dim)
    model_eval = np.zeros((*eta_shape, Nx, y_dim), dtype=np.float32)

    # 1 model param, 1 nuisance param and 1 input location
    x = np.squeeze(x, axis=-1)              # (...1, Nx)
    theta = np.squeeze(theta, axis=-1)      # (*, Nx)
    eta = np.squeeze(eta, axis=-1)          # (*, Nx)

    y1 = x * theta                          # (*, Nx) or (1, Nx)
    if y1.shape[0] == 1:
        # If reusing samples of theta
        y1 = np.tile(y1, (*eta_shape, 1))   # (*, Nx)

    y2 = (1 - x) * eta                      # (*, Nx)

    y = np.concatenate((np.expand_dims(y1, axis=-1), np.expand_dims(y2, axis=-1)), axis=-1)  # (*, Nx, 2)
    return y


# Analytical solution for linear gaussian model
def linear_gaussian_eig(d, var):
    return 0.5 * np.log(1 + d ** 2 / var)


# Simple nonlinear model example
def nonlinear_model(x, theta):
    # theta: (*, Nx, theta_dim)
    shape = theta.shape[:-2]
    theta_dim = theta.shape[-1]
    Nx, x_dim = x.shape
    y_dim = x_dim  # one output per x_dim
    assert theta_dim == 1
    assert theta.shape[-2] == Nx
    theta = np.squeeze(theta, axis=-1)            # (*, Nx)
    x = x.reshape((1,)*len(shape) + (Nx, x_dim))  # (...1, Nx, x_dim)

    model_eval = np.zeros((*shape, Nx, y_dim), dtype=np.float32)
    for i in range(x_dim):
        # Select all x for each x_dim
        ind = tuple([slice(None)]*(len(shape) + 1) + [i])  # slice(None) == : indexing
        x_i = x[ind]  # (...1, Nx)

        # Evaluate model for the x_i dimension
        model_eval[ind] = np.square(x_i) * np.power(theta, 3) + np.exp(-abs(0.2 - x_i)) * theta  # (*, Nx)

    return model_eval  # (*, Nx, y_dim)


def electrospray_current_model(x, theta, eta):
    """
    Predicts total array current for an electrospray thruster
    Parameters
    ----------
    x: (Nx, x_dim) voltage operating conditions [V]
    theta: (*, Nx, theta_dim) model parameters
    eta: (*, Nx, eta_dim) nuisance parameters

    Returns
    -------
    current: (*, Nx, y_dim) model output [A]
    """
    voltage = np.squeeze(x, axis=-1)    # (Nx,)
    Ne = 576
    N, Nx, eta_dim = eta.shape
    current = np.zeros((N, Nx, 1), dtype=np.float32)
    for i in range(N):
        for j in range(Nx):
            # Input location
            v_loc = np.atleast_1d(voltage[j])

            # Model parameters
            model_params = theta[0, j, :] if theta.shape[0] == 1 else theta[i, j, :]

            # Extract nuisance parameters
            nuis_params = eta[i, j, :]                          # (eta_dim,)
            subs = nuis_params[0:2, np.newaxis]                 # (2, 1)
            props = nuis_params[2:6, np.newaxis]                # (4, 1)
            beams = nuis_params[6, np.newaxis, np.newaxis]      # (1, 1)
            geo_data = nuis_params[7:].reshape((Ne, 7))         # (Ne, 7)
            geoms = geo_data[:, :6].T                           # (6, Ne)
            geoms = geoms[:, :, np.newaxis]                     # (6, Ne, 1)

            # Extract electrostatic simulation results
            emax_sim = geo_data[:, -1].reshape((1, Ne, 1))

            # Compute array current
            current[i, j, 0] = current_model(model_params, v_loc, subs, props, beams, geoms, es_models=emax_sim)

    return current


# Nested monte carlo expected information gain estimator
def eig_nmc(x_loc, theta_sampler, model, N=100, M=100, noise_cov=1.0, reuse_theta=False, use_parallel=True):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]
    mmap_folder = Path('./mmap_tmp')

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)                              # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Sample model parameters
    theta_i = theta_sampler((N, Nx)).astype(np.float32)         # (N, Nx, theta_dim)

    if use_parallel:
        # Allocate space
        try:
            os.mkdir(mmap_folder)
        except FileExistsError:
            pass
        y_file = mmap_folder / 'y_mmap.dat'
        g_theta_file = mmap_folder / 'g_theta_mmap.dat'
        evidence_file = mmap_folder / 'evidence_mmap.dat'
        y_i = np.memmap(str(y_file), dtype='float32', mode='w+', shape=(N, Nx, y_dim))
        g_theta_i = np.memmap(str(g_theta_file), dtype='float32', mode='w+', shape=(N, Nx, y_dim))
        evidence = np.memmap(str(evidence_file), dtype='float32', mode='w+', shape=(N, Nx))

        # Evaluate model
        g_theta_i[:] = model(x_loc, theta_i)                        # (N, Nx, y_dim)

        # Sample outer loop data y
        y_i[:] = batch_normal_sample(g_theta_i, noise_cov)          # (N, Nx, y_dim)

    else:
        # Evaluate model
        g_theta_i = model(x_loc, theta_i)                           # (N, Nx, y_dim)
        y_i = batch_normal_sample(g_theta_i, noise_cov)             # (N, Nx, y_dim)
        evidence = np.zeros((N, Nx), dtype=np.float32)

    # Parallel loop
    def parallel_func(idx, y_i, g_theta_i, evidence):
        y_curr = y_i[idx, np.newaxis, :, :]                         # (1, Nx, y_dim)
        if reuse_theta:
            evidence[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_i, noise_cov), axis=0)
        else:
            theta_j = theta_sampler((M, Nx)).astype(np.float32)     # (M, Nx, theta_dim)
            g_theta_j = model(x_loc, theta_j)                       # (M, Nx, y_dim)
            evidence[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_j, noise_cov), axis=0)

        if ((idx + 1) % 100) == 0 and not use_parallel:
            print(f'Samples processed: {idx + 1} out of {N}')

    # Compute evidence p(y|d)
    if use_parallel:
        Parallel(n_jobs=-1, verbose=5)(delayed(parallel_func)(idx, y_i, g_theta_i, evidence) for idx in range(N))
    else:
        for idx in range(N):
            parallel_func(idx, y_i, g_theta_i, evidence)

    # Compute likelihood
    likelihood = batch_normal_pdf(y_i, g_theta_i, noise_cov)        # (N, Nx)

    # Expected information gain
    eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)    # (Nx,)

    # Clean up
    try:
        shutil.rmtree(mmap_folder)
    except:
        pass

    return eig


# Nested monte carlo expected information gain estimator
def eig_nmc_pm(x_loc, theta_sampler, eta_sampler, model, N=100, M1=100, M2=100, noise_cov=np.asarray(1.0),
               reuse_samples=False, use_parallel=True):
    # Get problem dimension
    noise_cov = np.atleast_1d(noise_cov).astype(np.float32)
    y_dim = noise_cov.shape[0]
    mmap_folder = Path('./mmap_tmp')

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)                              # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Sample parameters
    theta_i = theta_sampler((N, Nx)).astype(np.float32)         # (N, Nx, theta_dim)
    eta_i = eta_sampler((N, Nx)).astype(np.float32)             # (N, Nx, eta_dim)

    if use_parallel:
        # Allocate space
        try:
            os.mkdir(mmap_folder)
        except FileExistsError:
            pass
        y_file = mmap_folder / 'y_mmap.dat'
        g_theta_file = mmap_folder / 'g_theta_mmap.dat'
        likelihood_file = mmap_folder / 'likelihood_mmap.dat'
        evidence_file = mmap_folder / 'evidence_mmap.dat'
        y_i = np.memmap(str(y_file), dtype='float32', mode='w+', shape=(N, Nx, y_dim))
        g_theta_i = np.memmap(str(g_theta_file), dtype='float32', mode='w+', shape=(N, Nx, y_dim))
        likelihood = np.memmap(str(likelihood_file), dtype='float32', mode='w+', shape=(N, Nx))
        evidence = np.memmap(str(evidence_file), dtype='float32', mode='w+', shape=(N, Nx))

        # Evaluate model
        g_theta_i[:] = model(x_loc, theta_i, eta_i)                 # (N, Nx, y_dim)

        # Sample outer loop data y
        y_i[:] = batch_normal_sample(g_theta_i, noise_cov)          # (N, Nx, y_dim)

    else:
        # Evaluate model
        g_theta_i = model(x_loc, theta_i, eta_i)                    # (N, Nx, y_dim)
        y_i = batch_normal_sample(g_theta_i, noise_cov)             # (N, Nx, y_dim)
        likelihood = np.zeros((N, Nx), dtype=np.float32)
        evidence = np.zeros((N, Nx), dtype=np.float32)

    # Parallel loop
    def parallel_func(idx, y_i, g_theta_i, likelihood, evidence):
        y_curr = y_i[idx, np.newaxis, :, :]                         # (1, Nx, y_dim)
        theta_curr = theta_i[idx, np.newaxis, :, :]                 # (1, Nx, theta_dim)
        if reuse_samples:
            # Compute evidence
            evidence[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_i, noise_cov), axis=0)

            # Compute likelihood
            g_theta_k = model(x_loc, theta_curr, eta_i)             # (N, Nx, y_dim)
            likelihood[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_k, noise_cov), axis=0)
        else:
            # Compute evidence
            eta_j = eta_sampler((M1, Nx)).astype(np.float32)        # (M1, Nx, eta_dim)
            theta_j = theta_sampler((M1, Nx)).astype(np.float32)    # (M1, Nx, theta_dim)
            g_theta_j = model(x_loc, theta_j, eta_j)                # (M1, Nx, y_dim)
            evidence[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_j, noise_cov), axis=0)

            # Compute likelihood
            eta_k = eta_sampler((M2, Nx)).astype(np.float32)        # (M2, Nx, eta_dim)
            g_theta_k = model(x_loc, theta_curr, eta_k)             # (M2, Nx, y_dim)
            likelihood[idx, :] = np.mean(batch_normal_pdf(y_curr, g_theta_k, noise_cov), axis=0)

        if ((idx + 1) % 100) == 0 and not use_parallel:
            print(f'Samples processed: {idx + 1} out of {N}')

    # Compute evidence p(y|d) and likelihood p(y|theta, d)
    if use_parallel:
        Parallel(n_jobs=-1, verbose=5)(delayed(parallel_func)(idx, y_i, g_theta_i, likelihood, evidence)
                                       for idx in range(N))
    else:
        for idx in range(N):
            parallel_func(idx, y_i, g_theta_i, likelihood, evidence)

    # Expected information gain
    eig = np.mean(np.log(likelihood) - np.log(evidence), axis=0)    # (Nx,)

    # Clean up
    try:
        shutil.rmtree(mmap_folder)
    except:
        pass

    return eig


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


def test_nonlinear_model():
    # Nonlinear model example, 1-d
    theta_sampler = lambda shape: np.random.rand(*shape, 1)
    N = 10000
    M = 1000
    Nx = 50
    x_loc = np.linspace(0, 1, Nx).reshape((Nx, 1))
    var = 1e-4
    t1 = time.time()
    eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, noise_cov=var, reuse_theta=False, use_parallel=True)
    t2 = time.time()
    print(f'Total time: {t2-t1:.2f} s')

    plt.figure()
    plt.plot(x_loc, eig, '-k')
    plt.xlabel('d')
    plt.ylabel('Expected information gain')
    plt.show()

    # Nonlinear model example, 2-d
    # theta_sampler = lambda shape: np.random.rand(*shape, 1)
    # N = 10000
    # M = 1000
    # Nx = [20, 20]  # [Nx, Ny, Nz, ..., Nd] - discretization in each batch dimension
    # loc = [np.linspace(0, 1, n) for n in Nx]
    # pt_grids = np.meshgrid(*loc)
    # x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
    # var = 1e-4 * np.eye(2)
    # t1 = time.time()
    # eig = eig_nmc(x_loc, theta_sampler, nonlinear_model, N=N, M=M, noise_cov=var, reuse_theta=True, use_parallel=True)
    # t2 = time.time()
    # print(f'Total time: {t2 - t1:.2f} s')

    # Reform grids
    # grid_d1, grid_d2 = [x_loc[:, i].reshape((Nx[1], Nx[0])) for i in range(2)]  # reform grids
    # eig_grid = eig.reshape((Nx[1], Nx[0]))

    # Plot results
    # plt.figure()
    # c = plt.contourf(grid_d1, grid_d2, eig_grid, 60, cmap='jet')
    # plt.colorbar(c)
    # plt.cla()
    # plt.contour(grid_d1, grid_d2, eig_grid, 15, cmap='jet')
    # plt.xlabel('$d_1$')
    # plt.ylabel('$d_2$')
    # plt.show()


def test_array_current_model():
    # Array current model
    def theta_sampler(shape):
        # Use preset samples from posterior
        params = np.loadtxt('../data/Nr100_noPr_samples__2021_12_07T11_41_27.txt', dtype=float, delimiter='\t',
                            skiprows=1)
        ind = np.random.randint(0, params.shape[0], shape)
        samples = params[ind, :]  # (*, 3)
        return samples

    def eta_sampler(shape):
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

        # Load emitter data
        Nr = 100
        Ne = 576
        geoms = np.loadtxt('../data/mr_geoms.dat', dtype=float, delimiter='\t')  # (Nr*Ne, geo_dim)
        emax_sim = np.loadtxt('../data/mr_geoms_tipE.dat', dtype=float, delimiter='\t')  # (Nr*Ne, )
        geo_dim = geoms.shape[1]
        sim_data = np.concatenate((geoms, emax_sim[:, np.newaxis]), axis=1)
        # g = geoms.reshape((Nr, 1, Ne * geo_dim))  # (Nr, 1, Ne*geo_dim)

        # Randomly sample emitter geometries that we have data for
        ind = np.random.randint(0, Nr * Ne, (*shape, Ne))  # (*, Ne)
        geo_data = sim_data[ind, :]  # (*, Ne, geo_dim+1)
        geo_data = np.reshape(geo_data, (*shape, Ne * (geo_dim + 1)))

        # Sample other parameters
        subs = subs_sampler(shape)  # (*, 2)
        props = prop_sampler(shape)  # (*, 4)
        beams = beam_sampler(shape)  # (*, 1)

        # Combine
        eta = np.concatenate((subs, props, beams, geo_data), axis=-1)  # (*, 7 + Ne*(geo_dim+1))

        return eta

    N = 200
    M1 = 100
    M2 = 100
    Nx = 40
    exp_data = np.loadtxt('../data/training_data.txt', dtype=float, delimiter='\t')
    var = np.max(exp_data[2, :])
    x_loc = np.linspace(800, 1840, Nx).reshape((Nx, 1))
    eig = eig_nmc_pm(x_loc, theta_sampler, eta_sampler, electrospray_current_model, N=N, M1=M1, M2=M2,
                     noise_cov=var, reuse_samples=True, use_parallel=True)
    plt.figure()
    plt.plot(x_loc, eig, '-k')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Expected information gain')
    plt.show()


if __name__ == '__main__':
    test_linear_gaussian_model(N=800, M1=800, M2=800, reuse_samples=False)
