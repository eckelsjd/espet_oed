import numpy as np

from src.utils import laplace_approx, is_positive_definite, nearest_positive_definite


def dram(logpdf, x0, nsamples, prop_sampler, prop_logpdf, k0=100, gamma=0.5, eps=1e-7, burn_frac=0.2, adaptive=True,
         delayed=True, symmetric_prop=False, show_iter=False):
    """ Delayed adaptive metropolis-hastings MCMC
    :param logpdf: f(x) -> log PDF (potentially unnormalized) of target distribution
    :param nsamples: number of MCMC samples to collect
    :param x0: (dim, ) initial parameter sample
    :param prop_sampler: q(x, *params) -> y, gives the proposal sample from current location
    :param prop_logpdf: f(x | y, *params) -> log PDF of x under proposal distribution given y and params
    """
    # Get dimension of parameters
    x0 = np.squeeze(np.atleast_1d(x0))
    assert len(x0.shape) == 1
    dim = x0.shape[0]

    # Obtain laplace approximation to initialize MCMC
    x0, cov0 = laplace_approx(x0, logpdf)
    sd = (2.4**2/dim)
    cov0 = sd * cov0
    curr_cov = cov0
    curr_mean = x0
    num_accept = 0.0
    curr_loc_logpdf = logpdf(x0)

    # Allocate space
    samples = np.zeros((nsamples, dim))
    accept_ratio = np.zeros(nsamples)
    samples[0, :] = x0

    # Acceptance ratio functions
    if symmetric_prop:
        a1 = lambda x, y, x_log, y_log, *params: min(1, np.exp(y_log - x_log))

        def a2(x, y1, y2, x_log, y1_log, y2_log, *params):
            a1_num = a1(y2, y1, y2_log, y1_log, *params)
            if a1_num == 1:
                return 0
            a1_denom = a1(x, y1, x_log, y1_log, *params)
            if a1_denom == 1:
                return 1

            frac_1 = y2_log - x_log
            frac_2 = prop_logpdf(y1, y2, *params) - prop_logpdf(y1, x, *params)
            frac_3 = np.log(1 - a1_num) - np.log(1 - a1_denom)
            return min(1, np.exp(frac_1 + frac_2 + frac_3))

    else:
        a1 = lambda x, y, x_log, y_log, *params: min(1, np.exp(y_log + prop_logpdf(x, y, *params) - x_log -
                                                               prop_logpdf(y, x, *params)))

        def a2(x, y1, y2, x_log, y1_log, y2_log, *params):
            a1_num = a1(y2, y1, y2_log, y1_log, *params)
            if a1_num == 1:
                return 0
            a1_denom = a1(x, y1, x_log, y1_log, *params)
            if a1_denom == 1:
                return 1

            frac_1 = y2_log - x_log
            frac_2 = prop_logpdf(y1, y2, *params) - prop_logpdf(y1, x, *params)
            frac_3 = np.log(1 - a1_num) - np.log(1 - a1_denom)
            frac_4 = prop_logpdf(x, y2, y1, *params) - prop_logpdf(y2, y1, x, *params)
            return min(1, np.exp(frac_1 + frac_2 + frac_3 + frac_4))

    # Main sample loop
    for i in range(nsamples - 1):
        # Check covariance condition
        if not is_positive_definite(curr_cov):
            print(f'Caught non-positive definite matrix: {curr_cov}')
            curr_cov = nearest_positive_definite(curr_cov)

        # Propose sample
        x1 = samples[i, :]
        y1 = prop_sampler(x1, curr_cov)

        # Compute logpdf of first proposal
        x1_log = curr_loc_logpdf
        y1_log = logpdf(y1)

        # Compute first acceptance a1(x,y1)
        if np.random.rand() < a1(x1, y1, x1_log, y1_log, curr_cov):
            samples[i + 1, :] = y1
            curr_loc_logpdf = y1_log
            num_accept += 1
        else:
            if delayed:
                # Second level proposal
                C2 = curr_cov * gamma
                y2 = prop_sampler(x1, C2)
                y2_log = logpdf(y2)

                if np.random.rand() < a2(x1, y1, y2, x1_log, y1_log, y2_log, C2):
                    samples[i + 1, :] = y2
                    curr_loc_logpdf = y2_log
                    num_accept += 1
                else:
                    samples[i + 1, :] = x1
            else:
                samples[i + 1, :] = x1

        accept_ratio[i] = num_accept / (i + 1)

        if adaptive:
            # Update the sample mean every iteration
            xbar_last = curr_mean.copy()
            curr_mean = (1/(i+1)) * x1 + (i/(i+1))*xbar_last

            # Adapt covariance after k0 steps
            if i >= k0:
                k = i
                x_curr = x1[:, np.newaxis]
                xbar_curr = curr_mean[:, np.newaxis]
                xbar_last = xbar_last[:, np.newaxis]
                mult = np.eye(dim) * eps + k * xbar_last @ xbar_last.T - \
                       (k + 1) * xbar_curr @ xbar_curr.T + \
                       x_curr @ x_curr.T
                curr_cov = ((k - 1) / k) * curr_cov + (sd / k) * mult

        if show_iter and i % 500 == 0:
            print(f'Iteration: {i}. Sample: {samples[i, :]}')

    # Burn off starting samples
    start_idx = int(burn_frac * nsamples)

    return samples[start_idx:, :], accept_ratio[start_idx:-1]
