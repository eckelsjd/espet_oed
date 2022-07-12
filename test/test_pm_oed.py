# import math
import cupy as cp   # for GPU
import numpy as np  # for CPU
import matplotlib.pyplot as plt
import time

# from ..scripts.SciTech2022_inference import current_model
# from ..scripts.esi_surrogate import forward


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


def gaussian_1d(x, mu, var):
    return 1 / cp.sqrt(2 * cp.pi * var) * cp.exp((-0.5 / var) * cp.square(x - mu))


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


def expected_information(Ns, Ndata, x_sampler, prior_sampler, model, noise_var=1e-4):
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

    # Run with cupy on GPU
    utility = compute_utility(y, g_theta, noise_var)

    print(f'Finished. Time: {time.time() - t1:.2f}')
    return cp.asnumpy(x_samples), cp.asnumpy(utility)


def nuisance_sampler(M, Ne=576):
    # Substrate properties
    lb = 5e-6
    ub = 8e-6
    rpr = lb + (ub-lb) * np.random.rand(M)  # pore radius [m]
    mu = 1.51e-13
    std = 6.04e-15
    kappa = mu + std * np.random.randn(M)  # permeability [m^-2]
    subs_props = np.concatenate((rpr[np.newaxis, :], kappa[np.newaxis, :]), axis=0)

    # Material properties [conductivity (Sm^-1), surface tension (Nm^-1), density (kg m^-3), viscosity (Nsm^-2)]
    lb = np.array([1.146519, .050038, 1279.782, .026119]).reshape(4, 1)
    ub = np.array([1.389983, .050452, 1284.369, .034162]).reshape(4, 1)
    mat_props = lb + (ub-lb) * np.random.rand(4, M)

    # Beam properties
    mu = 5.49932e5
    std = 1.0034e4
    qm_ratio = mu + std * np.random.randn(M)  # [C kg^-1]

    # Geometry parameters
    lb = 1e-5
    ub = 2e-5
    rc = lb + (ub - lb) * np.random.rand(M*Ne)  # radius of curvature [m]

    # [tip-to-extractor distance, radius of aperture, cone half-angle, height, local pore radius] in [m,m,rad,m,m]
    mu = np.array([3.0e-6, 4.9714e-04 / 2, 0.26782, 3.0181e-04, 1.3e-06 / 2]).reshape(5, 1)
    std = np.array([5.2302e-6, 7.1914e-06 / 2, 0.0040, 5.1302e-06, 1.5e-07 / 2]).reshape(5, 1)
    geoms = mu + std * np.random.randn(5, M*Ne)
    geom_props = np.concatenate((rc[np.newaxis, :], geoms), axis=0).reshape((6, Ne, M))

    return subs_props, mat_props, qm_ratio[np.newaxis, :], geom_props


if __name__ == '__main__':
    subs, mat, beam, geoms = nuisance_sampler(10, 3)
    print('hello')
