import numpy as np

from src.utils import fix_input_shape, approx_jacobian, linear_eig


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


# Compute linear Gaussian estimate of EIG
def eig_lg(x_loc, model, prior_mean, prior_cov, eta=None, noise_cov=np.asarray(1.0)):
    # Get problem dimensions
    noise_cov = np.atleast_1d(noise_cov)
    prior_cov = np.atleast_1d(prior_cov)
    y_dim = noise_cov.shape[0]
    noise_cov = noise_cov.reshape((1, y_dim, y_dim))
    theta_L = np.atleast_1d(prior_mean)
    if len(theta_L.shape) == 1:
        theta_L = np.expand_dims(theta_L, axis=0)
    theta_dim = theta_L.shape[-1]
    prior_cov = prior_cov.reshape((1, theta_dim, theta_dim))
    if eta:
        eta = np.atleast_1d(eta)
        if len(eta.shape) == 1:
            eta = np.expand_dims(eta, axis=0)

    # Experimental operating conditions x
    x_loc = fix_input_shape(x_loc)  # (Nx, x_dim)
    Nx, x_dim = x_loc.shape

    # Linearize model and compute analytical EIG
    A, c = linearize_model(model, x_loc, theta_L, eta)     # A: (Nx, y_dim, theta_dim), c: (Nx, y_dim)
    lin_eig = linear_eig(A, prior_cov, noise_cov)           # (Nx, )

    return lin_eig
