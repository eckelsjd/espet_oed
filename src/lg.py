import numpy as np

from src.utils import fix_input_shape, approx_jacobian


def linearize_model(model, d, theta_L, eta_L):
    """Return AB,c to linearize the model about theta_L/eta_L
    Parameters
    ----------
    model: expects to be called as model(d, theta, eta), returns (*, Nx, y_dim)
    d: (Nx, x_dim) model input locations
    theta_L: (*, theta_dim) model parameter points to linearize model about
    eta_L: (*, eta_dim) nuisance parameters points to linearize model about

    Returns
    -------
    AB: (*, Nx, y_dim, theta_dim+eta_dim) Jacobian matrices at (*, Nx) locations
    c: (*, Nx, y_dim) linear offsets at (*, Nx) locations
    """
    # Fix input shapes
    theta_L = np.atleast_1d(theta_L)
    if len(theta_L.shape) == 1:
        theta_L = np.expand_dims(theta_L, axis=0)
    eta_L = np.atleast_1d(eta_L)
    if len(eta_L.shape) == 1:
        eta_L = np.expand_dims(eta_L, axis=0)
    p_L = np.concatenate((theta_L, eta_L), axis=-1)     # (*, theta_dim + eta_dim)
    d = fix_input_shape(d)                              # (Nx, x_dim)
    AB = approx_jacobian(model, d, theta_L, eta_L)      # (*, Nx, y_dim, theta_dim + eta_dim)
    f_L = model(d, theta_L, eta_L)                      # (*, Nx, y_dim)
    delta = AB @ np.expand_dims(p_L, axis=-1)           # (*, Nx, y_dim, 1)
    c = f_L - np.squeeze(delta, axis=-1)                # (*, Nx, y_dim)
    return AB, c


def linear_eig(AB, theta_cov, eta_cov, gamma):
    """Computes the analytical expected information gain for a linear gaussian model
    Model: Y = [A|B]*[theta|eta]^T + c + xi,  where
           A -> system matrix
           theta -> model parameters, theta ~ N(theta_mean, theta_cov)
           eta -> nuisance parameters, eta ~ N(eta_mean, eta_cov)
           c -> constant offset
           xi -> experimental noise, xi ~ N(b, gamma)
    Parameters
    ----------
    AB: (*, y_dim, theta_dim+eta_dim) System matrices of length * and shape (y_dim, theta_dim+eta_dim)
    theta_cov: (*, theta_dim, theta_dim) Prior covariance matrix on model parameters
    eta_cov: (*, eta_dim, eta_dim) Prior covariance matrix on nuisance parameters
    gamma: (*, y_dim, y_dim) Experimental noise covariance

    Returns
    -------
    eig: (*,) The expected information gain for each system
    """
    AB = np.atleast_1d(AB)
    theta_cov = np.atleast_1d(theta_cov)
    theta_dim = theta_cov.shape[-1]
    eta_cov = np.atleast_1d(eta_cov)
    gamma = np.atleast_1d(gamma)
    if len(AB.shape) == 2:
        shape = (1,)
        AB = np.expand_dims(AB, axis=0)
        theta_cov = np.expand_dims(theta_cov, axis=0)
        eta_cov = np.expand_dims(eta_cov, axis=0)
        gamma = np.expand_dims(gamma, axis=0)
    else:
        shape = AB.shape[:-2]
    A = AB[..., 0:theta_dim]
    B = AB[..., theta_dim:]
    A_T = np.transpose(A, axes=tuple(np.arange(len(shape))) + (-1, -2))
    B_T = np.transpose(B, axes=tuple(np.arange(len(shape))) + (-1, -2))

    # Marginal posterior covariance
    C_inv = np.linalg.pinv(A @ theta_cov @ A_T + B @ eta_cov @ B_T + gamma)
    # C_inv = np.linalg.pinv(A @ theta_cov @ A_T + gamma)
    sigma_post = theta_cov - theta_cov @ A_T @ C_inv @ A @ theta_cov    # (*, theta_dim, theta_dim)

    # Compute expected information gain
    eig = (1/2) * np.log(np.linalg.det(theta_cov) / np.linalg.det(sigma_post))  # (*,)
    return eig


# Compute linear Gaussian estimate of EIG
def eig_lg(x_loc, model, theta_mean, theta_cov, eta_mean, eta_cov, noise_cov):
    # Get problem dimensions
    noise_cov = np.atleast_1d(noise_cov)
    theta_cov = np.atleast_1d(theta_cov)
    eta_cov = np.atleast_1d(eta_cov)
    y_dim = noise_cov.shape[-1]
    noise_cov = noise_cov.reshape((1, y_dim, y_dim))

    # Model parameters
    theta_L = np.atleast_1d(theta_mean)
    if len(theta_L.shape) == 1:
        theta_L = np.expand_dims(theta_L, axis=0)
    theta_dim = theta_L.shape[-1]
    theta_cov = theta_cov.reshape((1, theta_dim, theta_dim))

    # Nuisance parameters
    eta_L = np.atleast_1d(eta_mean)
    if len(eta_L.shape) == 1:
        eta_L = np.expand_dims(eta_L, axis=0)
    eta_dim = eta_L.shape[-1]
    eta_cov = eta_cov.reshape((1, eta_dim, eta_dim))

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)  # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Linearize model and compute analytical EIG
    AB, c = linearize_model(model, x_loc, theta_L, eta_L)     # AB: (Nx, y_dim, theta_dim+eta_dim), c: (Nx, y_dim)
    lin_eig = linear_eig(AB, theta_cov, eta_cov, noise_cov)   # (Nx, )

    return lin_eig
