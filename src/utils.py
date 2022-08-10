import numpy as np
import cupy as cp


def batch_normal_pdf(x, mu, cov, use_gpu=False):
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
    use_gpu: whether to execute on gpu
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

    # assert cov.shape[0] == cov.shape[1] == dim
    # assert x.shape[-1] == mu.shape[-1] == dim

    # Normalizing constant (scalar)
    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))

    # Can broadcast x - mu with x: (1, Nr, Nx, d) and mu: (Ns, Nr, Nx, d)
    diff = x - mu

    if use_gpu:
        # Move to GPU memory
        diff = cp.asarray(diff, dtype=cp.float32)
        cov = cp.asarray(cov, dtype=cp.float32)

        # In exponential
        diff_col = diff.reshape((*diff.shape, 1))                       # (*, d, 1)
        diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (*, 1, d)
        inexp = cp.squeeze(diff_row @ cp.linalg.inv(cov) @ diff_col, axis=(-1, -2))  # (*, 1, d) x (*, d, 1) = (*, 1, 1)

        # Compute pdf
        pdf = cp.asnumpy(preexp * cp.exp(-1 / 2 * inexp))
    else:
        # In exponential
        diff_col = diff.reshape((*diff.shape, 1))  # (*, d, 1)
        diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (*, 1, d)
        inexp = np.squeeze(diff_row @ np.linalg.inv(cov) @ diff_col, axis=(-1, -2))  # (*, 1, d) x (*, d, 1) = (*, 1, 1)

        # Compute pdf
        pdf = preexp * np.exp(-1 / 2 * inexp)

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


def model_1d_batch(x, theta, eta, model):
    """
    Predicts 1d model in batch experiments along x_dim
    Parameters
    ----------
    x: (Nx, x_dim) operating conditions
    theta: ((1 or Ns), Nx, theta_dim) model parameters
    eta: (Ns, Nx, eta_dim) nuisance parameters

    Returns
    -------
    model_eval: (Ns, Nx, y_dim) model output
    """
    shape = eta.shape[:-2]
    Nx, x_dim = x.shape
    y_dim = x_dim  # one output per x_dim

    model_eval = np.zeros((*shape, Nx, y_dim), dtype=np.float32)
    for i in range(x_dim):
        x_i = x[:, i, np.newaxis]  # (Nx, 1)

        # Evaluate model for the x_i experiment
        idx = tuple([slice(None)]*(len(shape) + 1)) + (i,)
        model_eval[idx] = np.squeeze(model(x_i, theta, eta), axis=-1)

    return model_eval  # (*, Nx, y_dim)

