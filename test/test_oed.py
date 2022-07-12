# import math
import cupy as cp   # for GPU
import numpy as np  # for CPU
from numba import jit, prange
import matplotlib.pyplot as plt
import time

from SciTech2022_inference import current_model
from esi_surrogate import forward

"""
LIMITATIONS
- experiments performed in batch are assumed to be governed by the same model
- batch experiments are independent from one another, so that the likelihood of the batch data
  is the product of the likelihood of each individual experiment data in the batch
- experimental noise is assumed Gaussian N~(0,sigma) with known variance
- single scalar output y for each experiment (can only measure one thing per experiment)
  - if output space were N-D, would need multi-dimensional Gaussian model?

TODO:
- GPU/parallel speed up
- multi-output OED
- use with Collin's current model (x=V, y=I, theta=[b0,zeta0,zeta1])

QUESTIONS:
- PC surrogate
- stochastic optimization
- Combining multiple models
- Multifidelity models to reduce computation
- Utility function for predictive posterior (or other design goals for JANUS)

ISSUES
- for loops required if sample size is too big to fit in memory (batch computation is difficult)
- numba can't perform actions along an axis
- numba can't do np.tile
"""


# Predict thruster array current
def array_current(theta, x):
    Ns, Nx, theta_dim = theta.shape
    bs, Nx, x_dim = x.shape

    Nr = 1  # number of realizations
    Ne = 576  # number of emitters

    # Substrate properties
    rp = np.mean([5e-6, 8e-6])  # pore radius [m]
    kappa = 1.51e-13  # permittivity [m^-2]
    subs = np.array([[rp], [kappa]])
    subs.reshape(2, Nr)

    # Material properties
    k = np.mean([1.147, 1.39])  # conductivity [Sm^-1]
    gamma = np.mean([5.003e-2, 5.045e-2])  # surface tension [N/m]
    rho = np.mean([1.28e3, 1.284e3])  # density [kg/m^3]
    mu = np.mean([2.612e-2, 3.416e-2])  # viscosity [N-s/m^2]
    props = np.array([k, gamma, rho, mu]).T.reshape(4, Nr)

    # Propellant properties
    qm_ratio = 5.5e5  # charge to mass ratio [C/kg]
    beams = np.array([qm_ratio]).reshape(1, Nr)

    # Geometry parameters
    rc = np.mean([1e-5, 2e-5])  # [m]
    d = 3e-6                    # [m]
    ra = 2.486e-4               # [m]
    alpha = 2.678e-1            # [rad]
    h = 3.018e-4                # [m]
    rpl = 6.5e-7                # [m]
    geoms = np.array([rc, d, ra, alpha, h, rpl])
    geoms = np.tile(geoms[:, np.newaxis], (1, Ne)).reshape(6, Ne, Nr)

    # Get surrogate Efield solution
    emitter_geo = np.array([d, rc, alpha, h, ra])
    V0 = 1000
    net_path = '../../data/base/models/esi_surrogate.onnx'

    # Calculate array current for each batch and set of params
    current = np.zeros((bs, Nx, Ns))
    for i in range(bs):
        for j in range(Nx):
            for k in range(Ns):
                params = theta[k, j, :]
                voltage = x[i, j, :]
                E_max = forward(emitter_geo, net_file=net_path, V0=voltage)
                es_model = np.divide(E_max, voltage)
                es_model = np.tile(es_model[:, np.newaxis, np.newaxis], (1, Ne, 1))
                current[i, j, k] = current_model(params, voltage, subs, props, beams, geoms, es_models=es_model)

    return current


# Simple nonlinear model example
def nonlinear_model(theta, x):
    Ns, Nx, theta_dim = theta.shape
    bs, Nx, x_dim = x.shape

    # 1 model param and one input for this model
    assert theta_dim == 1 and x_dim == 1
    x = cp.squeeze(x, axis=2)
    theta = cp.squeeze(theta, axis=2)

    model_eval = cp.zeros((bs, Nx, Ns))
    for i in range(Nx):
        theta_row = theta[:, i]
        theta_row = theta_row[cp.newaxis, :]
        for j in range(bs):
            x_col = x[j, :]
            x_col = x_col[:, cp.newaxis]
            model_eval[j, :, :] = cp.square(x_col) @ cp.power(theta_row, 3) + \
                                  cp.exp(-abs(0.2 - x_col)) @ theta_row  # (Nx, Ns)

    return model_eval  # (bs, Nx, Ns)


@jit(nopython=False)
def gaussian_1d_jit(x, mu, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp((-0.5 / var) * np.square(x - mu))


def gaussian_1d(x, mu, var):
    return 1 / cp.sqrt(2 * cp.pi * var) * cp.exp((-0.5 / var) * cp.square(x - mu))


@jit(nopython=False, parallel=True)
def compute_utility_jit(y, g_theta, noise_var):
    bs, Nx, Ns = y.shape

    # Compute likelihood (product over batch size
    likelihood = np.prod(gaussian_1d_jit(y, g_theta, noise_var), axis=0)     # (Nx, Ns) grid

    # Compute evidence (average marginal likelihood)
    evidence = np.zeros((Nx, Ns), dtype=np.float32)
    print(f'Sample processed: {0} out of {Ns}')
    for j in prange(Ns):
        y_j = y[:, :, j]  # (bs, Nx)
        y_tile = y_j.repeat(Ns).reshape((bs, Nx, Ns))  # jit compatible to np.tile
        like = np.prod(gaussian_1d_jit(y_tile, g_theta, noise_var), axis=0)
        marginal_like = np.mean(like, axis=1)  # (Nx,)
        evidence[:, j] = marginal_like
        if ((j+1) % 5) == 0:
            print(f'Sample processed: {j+1} out of {Ns}')

    # Expected information gain
    utility = np.mean(np.log(likelihood) - np.log(evidence), axis=1)  # (Nx,)

    return utility


# GPU version
def compute_utility(y, g_theta, noise_var):
    bs, Nx, Ns = y.shape

    # Compute likelihood
    likelihood = gaussian_1d(y, g_theta, noise_var)  # (bs, Nx, Ns) grid
    likelihood = cp.prod(likelihood, axis=0)         # product over batch size - (Nx, Ns)

    # Compute evidence (average marginal likelihood)
    evidence = cp.zeros((Nx, Ns), dtype=cp.float32)
    print(f'Sample processed: {0} out of {Ns}')
    for j in range(Ns):
        y_j = y[:, :, j]  # (bs, Nx)
        y_tile = cp.tile(y_j[:, :, cp.newaxis], (1, 1, Ns))  # (bs, Nx, Ns)
        like = cp.prod(gaussian_1d(y_tile, g_theta, noise_var), axis=0)  # (Nx, Ns)
        marginal_like = cp.mean(like, axis=1)  # (Nx,)
        evidence[:, j] = marginal_like
        if ((j+1) % 1000) == 0:
            print(f'Sample processed: {j+1} out of {Ns}')

    # Expected information gain
    utility = cp.mean(cp.log(likelihood) - cp.log(evidence), axis=1)  # (Nx,)

    return utility


def expected_information(Ns, Ndata, x_sampler, prior_sampler, model, noise_var=1e-4, use_gpu=True):
    def fix_input_shape(x):
        """Make input shape: (bs, Nx, xdim)
        bs: batch size, number of experiments to run in one go
        Nx: number of experimental locations (inputs x) to evaluate at
        xdim: dimension of a single experimental input x
        """
        x = np.atleast_1d(x).astype(np.float32)
        if len(x.shape) == 1:
            # Assume one x dimension and batch size of 1
            x = x[np.newaxis, :, np.newaxis]
        elif len(x.shape) == 2:
            # Assume only one x dimension
            x = x[:, :, np.newaxis]
        elif len(x.shape) != 3:
            raise Exception('Incorrect input dimension')
        return x

    def fix_theta_shape(theta):
        """Make theta shape: (Ns, Nx, theta_dim)
        Ns: Number of samples from the prior for each input x
        Nx: number of experimental locations (inputs x) to evaluate at
        theta_dim: Number of model parameters
        """
        theta = np.atleast_1d(theta).astype(np.float32)
        if len(theta.shape) == 1:
            # Assume one model parameter and one location x
            theta = theta[:, np.newaxis, np.newaxis]
        elif len(theta.shape) == 2:
            # Assume only one model parameter
            theta = theta[:, :, np.newaxis]
        elif len(theta.shape) != 3:
            raise Exception('Incorrect input dimension')
        return theta

    # Sample experimental input locations x
    x_samples = fix_input_shape(x_sampler(Ndata))  # (bs, Nx, xdim)
    bs = x_samples.shape[0]

    # Sample the prior
    Nx = int(np.prod(np.asarray(Ndata)))  # total number of grid points in input-space x
    theta_samples = fix_theta_shape(prior_sampler(Ns, Nx))  # (Ns, Nx, theta_dim)

    # Evaluate the model
    g_theta = model(theta_samples, x_samples)  # (bs, Nx, Ns)
    assert g_theta.shape == (bs, Nx, Ns)

    # Get samples of y
    y = np.random.normal(loc=g_theta, scale=np.sqrt(noise_var))
    # y = cp.random.normal(loc=g_theta, scale=cp.sqrt(noise_var), size=(bs, Nx, Ns))  # (bs, Nx, Ns)

    # HIGHEST OVERHEAD HERE O(Ns)
    t1 = time.time()
    if use_gpu:
        # Run with cupy on GPU
        utility = compute_utility(y, g_theta, noise_var)
    else:
        # Run with numba in parallel on CPU
        utility = compute_utility_jit(y, g_theta, noise_var)
    print(f'Finished. Time: {time.time() - t1:.2f}')
    return cp.asnumpy(x_samples), cp.asnumpy(utility)


def test_1d_case(x_sampler, theta_sampler, Ndata, Ns=100, use_gpu=True):
    d, U_d = expected_information(Ns, Ndata, x_sampler, theta_sampler, nonlinear_model, use_gpu=use_gpu)

    # Plot results
    plt.plot(cp.squeeze(d), U_d, '-k')
    plt.xlabel('$d$')
    plt.ylabel('$\hat{U}(d)$')
    plt.show()


def test_2d_case(x_sampler, theta_sampler, Ndata, Ns=100, use_gpu=True):
    dim = len(Ndata)
    d, U_d = expected_information(Ns, Ndata, x_sampler, theta_sampler, nonlinear_model, use_gpu=use_gpu)
    grid_d1, grid_d2 = [d[i, :].reshape((Ndata[1], Ndata[0])) for i in range(dim)]  # reform grids
    U_grid = U_d.reshape((Ndata[1], Ndata[0]))

    # Plot results
    plt.figure()
    c = plt.contourf(grid_d1, grid_d2, U_grid, 60, cmap='jet')
    plt.colorbar(c)
    plt.cla()
    plt.contour(grid_d1, grid_d2, U_grid, 15, cmap='jet')
    plt.xlabel('$d_1$')
    plt.ylabel('$d_2$')
    plt.show()


if __name__ == '__main__':
    def x_sampler(N):
        """N: [Nx, Ny, Nz, ..., Nd] - discretization in each batch dimension"""
        N = np.atleast_1d(N)
        loc = [np.linspace(0, 1, n) for n in N]
        pt_grids = np.meshgrid(*loc)
        pts = np.vstack([grid.ravel() for grid in pt_grids])
        return pts  # (bs, np.prod(N))

    def theta_sampler(Ns, Nx):
        return np.random.rand(Ns, Nx)

    def voltage_sampler(N):
        return np.linspace(1, 2000, N)

    def current_theta_sampler(Ns, Nx):
        # "Positive reals" prior
        lb = 0
        ub = 3
        samples = np.random.rand(Ns, Nx, 3)
        return lb + (ub-lb)/(1-0) * samples


    # test_1d_case(x_sampler, theta_sampler, Ns=100000, Ndata=101, use_gpu=True)
    # test_2d_case(x_sampler, theta_sampler, Ns=30000, Ndata=[30, 30], use_gpu=True)

    # Test deterministic current model
    Ns = 1000
    Ndata = 100
    d, U_d = expected_information(Ns, Ndata, voltage_sampler, current_theta_sampler,
                                  array_current, use_gpu=True, noise_var=0.5e-4)

    # Plot results
    plt.plot(cp.squeeze(d), U_d, '-k')
    plt.xlabel('$d$')
    plt.ylabel('$\hat{U}(d)$')
    plt.show()
