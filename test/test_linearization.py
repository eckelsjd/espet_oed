import numpy as np
import matplotlib.pyplot as plt
import time

from src.utils import fix_input_shape, linear_eig, approx_jacobian
from src.models import nonlinear_model
from src.nmc import eig_nmc


def linearize_model(model, d, theta_L, eta):
    """Return A,c to linearize the model about theta_L
    Parameters
    ----------
    model: expects to be called as model(d, theta, eta), returns (*, Nx, y_dim)
    d: (Nx, x_dim) model input locations
    theta_L: (*, theta_dim) model parameter points to linearize model about
    eta: (*, Nx, eta_dim) nuisance parameters needed to run the model

    Returns
    -------
    A: (*, Nx, y_dim, theta_dim) Jacobian matrices at (*, Nx) locations
    c: (*, Nx, y_dim) linear offsets at (*, Nx) locations
    """
    # Fix input shapes
    theta_L = np.atleast_1d(theta_L)
    if len(theta_L.shape) == 1:
        theta = np.expand_dims(theta_L, axis=0)
    d = fix_input_shape(d)                          # (Nx, x_dim)
    A = approx_jacobian(model, d, theta_L, eta)     # (*, Nx, y_dim, theta_dim)
    f_L = model(d, theta_L, eta)                    # (*, Nx, y_dim)
    delta = A @ np.expand_dims(theta_L, axis=-1)    # (*, Nx, y_dim, 1)
    c = f_L - np.squeeze(delta, axis=-1)            # (*, Nx, y_dim)
    return A, c


def test_nonlinear():
    Nx = 100
    theta_val = 0.5
    x = fix_input_shape(np.linspace(0, 1, Nx))
    theta = np.array([theta_val]).reshape((1, 1))
    y = nonlinear_model(x, theta)               # (Nx, y_dim)

    # Exact linearization
    M = 50
    t = np.linspace(.95*theta_val, 1.05*theta_val, M)
    y_lin_exact = np.zeros((M, Nx))
    dydtheta = lambda x, theta: 3*theta**2 * x**2 + np.exp(-np.abs(0.2 - x))
    for i in range(M):
        theta_new = np.array(t[i]).reshape((1, 1))
        model_eval = nonlinear_model(x, theta) + dydtheta(x, theta) * (theta_new - theta)
        y_lin_exact[i, :] = np.squeeze(model_eval, axis=-1)
    lower_y_exact = np.percentile(y_lin_exact, 5, axis=0)
    upper_y_exact = np.percentile(y_lin_exact, 95, axis=0)

    # Approximate linearization
    A, c = linearize_model(nonlinear_model, x, theta, None)
    y_lin = np.zeros((M, Nx))
    for i in range(M):
        y_lin[i, :] = A[:, 0, 0] * t[i] + c[:, 0]
    lower_y_approx = np.percentile(y_lin, 5, axis=0)
    upper_y_approx = np.percentile(y_lin, 95, axis=0)

    plt.figure()
    plt.plot(x, y, '-k', label='Exact')
    plt.plot(x, lower_y_exact, '-g', label='5th percentile')
    plt.plot(x, upper_y_exact, '-b', label='95th percentile')
    plt.plot(x, lower_y_approx, '--r', label='Approx 5th')
    plt.plot(x, upper_y_approx, '--y', label='Approx 95th')
    plt.xlabel(r'$d$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()


def test_nonlinear_eig():
    Nx = 100
    theta_mean = 0.9
    theta_var = 0.01
    noise_cov = 1e-4
    y_dim = 1
    theta_dim = 1
    x = fix_input_shape(np.linspace(0, 1, Nx))
    theta_L = np.array([theta_mean]).reshape((1, theta_dim))  # (Nx, theta_dim) -- broadcast Nx
    sigma = np.array([theta_var]).reshape((1, theta_dim, theta_dim))
    gamma = np.array([noise_cov]).reshape((1, y_dim, y_dim))

    # Linearization
    s_1 = time.time()
    A, c = linearize_model(nonlinear_model, x, theta_L, None)     # A: (Nx, 1, 1), c: (Nx, 1)
    lin_eig = linear_eig(A, sigma, gamma)
    e_1 = time.time()

    # Nested Monte Carlo
    N = 1000
    theta_sampler = lambda shape: np.random.randn(*shape, 1)*np.sqrt(theta_var) + theta_mean
    s_2 = time.time()
    nmc_eig = eig_nmc(x, theta_sampler, nonlinear_model, N=10000, noise_cov=noise_cov, reuse_samples=True, n_jobs=-1)
    e_2 = time.time()

    # Show results
    plt.figure()
    plt.plot(x, nmc_eig, '-k', label='Nested Monte Carlo')
    plt.plot(x, lin_eig, '--r', label='Linear Gaussian')
    plt.xlabel('$d$')
    plt.ylabel('Expected information gain')
    plt.legend()
    plt.show()

    # Print runtimes
    print('RUNTIMES:')
    print(f'Linear Gaussian: {e_1-s_1:.04} s')
    print(f'Nested monte carlo: {e_2 - s_2:.04} s')


def test_analytical():
    N = 100
    y_dim = 2
    theta_dim = 2
    sigma = 1*np.eye(theta_dim)
    eps_var = 0.01
    gamma = eps_var*np.eye(y_dim)
    d = np.linspace(0, 1, N)
    A = np.zeros((N, y_dim, theta_dim))
    A[:, 0, 0] = d
    A[:, 1, 1] = 1 - d
    sigma = np.expand_dims(sigma, axis=0)
    gamma = np.expand_dims(gamma, axis=0)

    eig = linear_eig(A, sigma, gamma)
    eig_anal = (1/2)*np.log((((1-d)**2 + eps_var)*(d**2 + eps_var)) / (eps_var ** 2))

    # Marginal
    A = np.zeros((N, y_dim, 1))
    A[:, 0, 0] = d
    sigma = np.array([[[1]]])
    eig_marg = linear_eig(A, sigma, gamma)
    eig_marg_anal = (1/2)*np.log(1 + d**2/eps_var)

    plt.figure()
    plt.plot(d, eig_anal, '-k', label='Analytical joint')
    plt.plot(d, eig, '--r', label='Numerical joint')
    plt.plot(d, eig_marg_anal, '-g', label='Analytical marg')
    plt.plot(d, eig_marg, '--y', label='Numerical marg')
    plt.xlabel(r'$d$')
    plt.ylabel('Expected information gain')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # test_nonlinear()
    # test_analytical()
    test_nonlinear_eig()
