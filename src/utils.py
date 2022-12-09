import numpy as np
import psutil
from sys import platform
import sys
import time
import logging
from numpy.linalg.linalg import LinAlgError
import scipy.optimize
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import matplotlib.cm

if platform != 'win32':
    import resource


def log_memory_usage(interval_sec):
    while True:
        # Log memory usage
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        remaining = mem.available * 100 / mem.total
        logging.info(f'  {remaining:.1f}% RAM available ({used_gb:.2f}/{total_gb:.2f} GB used)')

        # Sleep
        time.sleep(interval_sec)


def memory(percentage=1):
    def decorator(func):

        def wrapper(*args, **kwargs):
            # Limit memory usage
            logging.info(f'Memory percentage requested: {percentage * 100:.1f} %')
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            maxbytes = int(psutil.virtual_memory().available * percentage)
            resource.setrlimit(resource.RLIMIT_AS, (maxbytes, hard))
            logging.info(f'Setting memory limit to: {maxbytes / (1024 ** 3):.2f} GB')
            try:
                return func(*args, **kwargs)
            except MemoryError:
                mem = psutil.virtual_memory()
                used_gb = mem.used / (1024 ** 3)
                total_gb = mem.total / (1024 ** 3)
                remaining = mem.available * 100 / mem.total
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                logging.critical(f'Remaining RAM: {remaining:.1f} %  ({used_gb:.2f}/{total_gb:.2f} GB used)')
                sys.exit(1)
        return wrapper
    return decorator


def electrospray_samplers(Ne=576):
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
        # rpr = np.ones((*shape, 1)) * 8e-6
        kappa = np.random.randn(*shape, 1) * 6.04e-15 + 1.51e-13
        # subs = np.concatenate((rpr, kappa), axis=-1)
        return kappa  # (*, 1)

    def eta_sampler(shape):
        # Load emitter data
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

    return theta_sampler, eta_sampler


def approx_jacobian(func, d, theta, eta, pert=0.01):
    """Approximate Jacobian of the function at a specified theta/eta location
    Parameters
    ----------
    func: expects to be called as func(d, theta, eta), returns (*, Nx, y_dim)
    d: (Nx, x_dim) model input locations
    theta: (*, theta_dim) point to linearize model about
    eta: (*, eta_dim) nuisance point to linearize model about
    pert: Perturbation for approximate partial derivatives

    Returns
    -------
    J: (*, Nx, y_dim, theta_dim+eta_dim) The approximate Jacobian (y_dim, theta_dim+eta_dim) at locations (*, Nx)
    """
    f = func(d, theta, eta)         # (*, Nx, y_dim)
    shape = theta.shape[:-1]        # (*)
    theta_dim = theta.shape[-1]     # Number of model parameters
    eta_dim = eta.shape[-1]         # Number of nuisance parameters
    Nx, x_dim = d.shape             # Dimension of input
    y_dim = f.shape[-1]             # Dimension of output
    p = np.concatenate((theta, eta), axis=-1).astype(np.float32)   # (*, theta_dim + eta_dim)
    dp = pert * p
    dp[dp == 0] = pert  # arbitrary step size of pert if p=0

    # Return a Jacobian (y_dim, theta_dim+eta_dim) at locations (*, Nx)
    J = 0
    if len(shape) == 1:
        J = np.zeros((Nx, y_dim, theta_dim+eta_dim))
    elif len(shape) > 1:
        J = np.zeros((*(shape[:-1]), Nx, y_dim, theta_dim+eta_dim))
    ind = tuple([slice(None)] * len(shape))  # (*)

    for k in range(theta_dim+eta_dim):
        # Center difference scheme to approximate partial derivatives
        p_forward = np.copy(p)
        p_backward = np.copy(p)
        p_forward[(*ind, k)] += dp[(*ind, k)]
        p_backward[(*ind, k)] -= dp[(*ind, k)]
        f1 = func(d, p_forward[..., 0:theta_dim], p_forward[..., theta_dim:])       # (*, Nx, y_dim)
        f2 = func(d, p_backward[..., 0:theta_dim], p_backward[..., theta_dim:])     # (*, Nx, y_dim)
        J[(*ind, slice(None), k)] = (f1 - f2) / (2*np.expand_dims(dp[(*ind, k)], axis=-1))

    return J


def approx_hessian(func, d, theta, eta=None, pert=0.01):
    """Approximate Hessian of the function at a specified theta location
    Parameters
    ----------
    func: expects to be called as func(d, theta, eta), returns (*, Nx, y_dim)
    d: (Nx, x_dim) model input locations
    theta: (*, theta_dim) point to linearize model about
    eta: (*, Nx, eta_dim) nuisance parameters needed to run the model, or None
    pert: Perturbation for approximate partial derivatives

    Returns
    -------
    H: (*, Nx, theta_dim, theta_dim) The approximate Hessian (theta_dim, theta_dim) at locations (*, Nx)
    """
    f = np.atleast_1d(func(d, theta, eta))  # (*, Nx, y_dim)
    shape = theta.shape[:-1]                # (*)
    theta_dim = theta.shape[-1]             # Number of parameters
    Nx, x_dim = d.shape                     # Dimension of input
    y_dim = f.shape[-1]                     # Dimension of output
    dtheta = pert * theta

    # Return a Hessian (theta_dim, theta_dim) at locations (*, Nx)
    H = 0
    if len(shape) == 1:
        H = np.zeros((Nx, theta_dim, theta_dim))
    elif len(shape) > 1:
        H = np.zeros((*(shape[:-1]), Nx, theta_dim, theta_dim))
    else:
        H = np.zeros((theta_dim, theta_dim))
    ind = tuple([slice(None)] * len(shape))  # (*)

    for i in range(theta_dim):
        for j in range(i, theta_dim):
            # Allocate space at 4 grid points (n1=-1, p1=+1)
            theta_n1_n1 = np.copy(theta)
            theta_p1_p1 = np.copy(theta)
            theta_n1_p1 = np.copy(theta)
            theta_p1_n1 = np.copy(theta)

            # Perturbations to theta in each direction
            theta_n1_n1[(*ind, i)] -= dtheta[(*ind, i)]
            theta_n1_n1[(*ind, j)] -= dtheta[(*ind, j)]
            f_n1_n1 = func(d, theta_n1_n1, eta)

            theta_p1_p1[(*ind, i)] += dtheta[(*ind, i)]
            theta_p1_p1[(*ind, j)] += dtheta[(*ind, j)]
            f_p1_p1 = func(d, theta_p1_p1, eta)

            theta_n1_p1[(*ind, i)] -= dtheta[(*ind, i)]
            theta_n1_p1[(*ind, j)] += dtheta[(*ind, j)]
            f_n1_p1 = func(d, theta_n1_p1, eta)

            theta_p1_n1[(*ind, i)] += dtheta[(*ind, i)]
            theta_p1_n1[(*ind, j)] -= dtheta[(*ind, j)]
            f_p1_n1 = func(d, theta_p1_n1, eta)

            res = (f_n1_n1 + f_p1_p1 - f_n1_p1 - f_p1_n1) / np.expand_dims(4 * dtheta[(*ind, i)] * dtheta[(*ind, j)],
                                                                           axis=-1)

            # Hessian only computed for scalar functions, y_dim=1 on last axis
            H[(*ind, i, j)] = np.squeeze(res, axis=-1)
            H[(*ind, j, i)] = np.squeeze(res, axis=-1)

    return H


def hess(func, theta, pert=0.01):
    """Approximate Hessian of the function at a specified theta location
    Parameters
    ----------
    func: expects to be called as func(theta) -> (*, y_dim)
    theta: (*, theta_dim) point to linearize model about
    pert: Perturbation for approximate partial derivatives

    Returns
    -------
    H: (*, theta_dim, theta_dim) The approximate Hessian (theta_dim, theta_dim) at locations (*)
    """
    theta = np.atleast_1d(theta)
    shape = theta.shape[:-1]                # (*)
    theta_dim = theta.shape[-1]             # Number of parameters
    dtheta = pert * theta

    # Return a Hessian (theta_dim, theta_dim) at locations (*)
    H = np.zeros((*(shape[:-1]), theta_dim, theta_dim))

    for i in range(theta_dim):
        for j in range(i, theta_dim):
            # Allocate space at 4 grid points (n1=-1, p1=+1)
            theta_n1_n1 = np.copy(theta)
            theta_p1_p1 = np.copy(theta)
            theta_n1_p1 = np.copy(theta)
            theta_p1_n1 = np.copy(theta)

            # Perturbations to theta in each direction
            theta_n1_n1[..., i] -= dtheta[..., i]
            theta_n1_n1[..., j] -= dtheta[..., j]
            f_n1_n1 = func(theta_n1_n1)

            theta_p1_p1[..., i] += dtheta[..., i]
            theta_p1_p1[..., j] += dtheta[..., j]
            f_p1_p1 = func(theta_p1_p1)

            theta_n1_p1[..., i] -= dtheta[..., i]
            theta_n1_p1[..., j] += dtheta[..., j]
            f_n1_p1 = func(theta_n1_p1)

            theta_p1_n1[..., i] += dtheta[..., i]
            theta_p1_n1[..., j] -= dtheta[..., j]
            f_p1_n1 = func(theta_p1_n1)

            res = (f_n1_n1 + f_p1_p1 - f_n1_p1 - f_p1_n1) / np.expand_dims(4 * dtheta[..., i] * dtheta[..., j],
                                                                           axis=-1)

            # Hessian only computed for scalar functions, y_dim=1 on last axis
            H[..., i, j] = np.squeeze(res, axis=-1)
            H[..., j, i] = np.squeeze(res, axis=-1)

    return H


def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index = True
            else:
                use_index = False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, N))
        return cycler("color", colors)


def ax_default(ax, xlabel='', ylabel='', legend=True):
    plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    if legend:
        leg = plt.legend(fancybox=True)
        frame = leg.get_frame()
        frame.set_edgecolor('k')


def effective_sample_size(Nsamples, auto_corr):
    """Compute the effective MCMC sample size
    :param Nsamples: Number of MCMC samples
    :param auto_corr: (maxlag, ndim) Autocorrelation, R(l) for lag = 1 to inf, or R(l[-1]) ~ 0
    """
    IAC = 1 + 2*np.sum(auto_corr, axis=0)
    ESS = Nsamples / IAC
    return ESS


def autocorrelation(samples, maxlag=100, step=1):
    """Compute the correlation of a set of samples
    :param samples: (Nsamples, dim)
    :param maxlag: maximum distance to compute the correlation for
    :param step: step between distances from 0 to maxlag for which to compute the correlations
    """
    # Get the shapes
    nsamples, ndim = samples.shape

    # Compute the mean
    mean = np.mean(samples, axis=0)

    # Compute the variance for each parameter
    var = np.sum((samples - mean[np.newaxis, :]) ** 2, axis=0)

    lags = np.arange(0, maxlag, step)
    autos = np.zeros((len(lags), ndim))
    for zz, lag in enumerate(lags):
        autos[zz, :] = np.zeros((ndim))
        # compute the covariance between all samples *lag apart*
        for ii in range(nsamples - lag):
            autos[zz, :] = autos[zz, :] + (samples[ii, :] - mean) * (samples[ii + lag, :] - mean)
        autos[zz, :] = autos[zz, :] / var
    return lags, autos


def laplace_approx(x0, logpost, hess_custom=True):
    """Perform the laplace approximation, returning the MAP point and an approximation of the covariance
    :param x0: (nparam, ) array of initial parameters
    :param logpost: f(param) -> log posterior pdf

    :returns map_point: (nparam, ) MAP of the posterior
    :returns cov_approx: (nparam, nparam), covariance matrix for Gaussian fit at MAP
    """
    # Gradient free method to obtain optimum
    neg_logpost = lambda theta: -logpost(theta)
    hess_inv = lambda theta: np.linalg.pinv(hess(neg_logpost, theta))
    res = scipy.optimize.minimize(neg_logpost, x0, method='Nelder-Mead')
    map_point = res.x

    if hess_custom:
        cov_approx = hess_inv(map_point)
    else:
        # Gradient method which also approximates the inverse of the hessian
        res = scipy.optimize.minimize(neg_logpost, res.x*0.95)
        map_point = res.x
        cov_approx = res.hess_inv

    return map_point, cov_approx


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except LinAlgError:
        return False


def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def batch_normal_pdf(x, mu, cov, logpdf=False):
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
    cov: (*, d, d) covariance matrix
    Returns
    -------
    pdf: (*) the multivariate normal pdf at each x location
    """
    # Make some checks on input
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    cov = np.atleast_1d(cov)
    dim = cov.shape[-1]

    # 1-D case
    if len(cov.shape) == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(mu.shape) == 1:
        mu = mu[np.newaxis, :]

    assert cov.shape[-1] == cov.shape[-2] == dim
    assert x.shape[-1] == mu.shape[-1] == dim

    # Normalizing constant (scalar)
    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))

    # Can broadcast x - mu with x: (1, Nr, Nx, d) and mu: (Ns, Nr, Nx, d)
    diff = x - mu

    # In exponential
    diff_col = diff.reshape((*diff.shape, 1))  # (*, d, 1)
    diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (*, 1, d)
    inexp = np.squeeze(diff_row @ np.linalg.inv(cov) @ diff_col, axis=(-1, -2))  # (*, 1, d) x (*, d, 1) = (*, 1, 1)

    # Compute pdf
    pdf = np.log(preexp) + (-1/2)*inexp if logpdf else preexp * np.exp(-1 / 2 * inexp)

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
        mean = mean[np.newaxis, :]

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

