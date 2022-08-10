# -*- coding: utf-8 -*-
"""
@author: Collin
"""
import numpy as np
import scipy.optimize as opt
import scipy.linalg as la
import pickle as pkl
import time as t
# import winsound as ws
import matplotlib.pyplot as plt
import datetime as dt

VACUUM_PERMITTIVITY = 8.8542e-12

#%% pseudomarginal-compatible MH-MCMC DRAM code from AE 740 projects
#I guess I could call is PM MHDRAM-MCMC or some such
def mh_acceptance_prob(current_target_logpdf,proposed_target_logpdf,\
                       current_sample, proposed_sample, proposal_logpdf):
    """Compute the metropolis-hastings accept-reject probability
    
    I've modified Alex's code and format to also return the forward logpdf, as
    this is needed for the DR implementation, and I want to avoid evaluating
    it multiple times.
    
    Inputs
    ------
    current_target_logpdf : float
        logpdf at the current sample in the chain f_X(x^{(k)})
    proposed_target_logpdf : float
        logpdf at the proposed sample in the chain
    current_sample : vector (d, )
        current sample
    proposed_sample : vector (d, )
        proposed sample
    proposal_logpdf: f(x, y)
        callable that gives the log probability of y given x
    
    Returns
    -------
    a : float
        Acceptance probability.
    prop_forward_logpdf : float
        Forward proposal logpdf. If DR is implemented, used in DR calculation.
        
    """
    
    prop_reverse_logpdf = proposal_logpdf(proposed_sample, current_sample)
    prop_forward_logpdf = proposal_logpdf(current_sample, proposed_sample)
    check = proposed_target_logpdf - current_target_logpdf +\
        prop_reverse_logpdf - prop_forward_logpdf
    if check < 0:
        return np.exp(check), prop_forward_logpdf
    else:
        return 1  , prop_forward_logpdf
def DR_mh_acceptance_prob(current_target_logpdf, DR_proposed_target_logpdf,\
          prop_forward_logpdf, back_prop_prop_logpdf, proposal_acceptance,\
              back_prop_prop_acceptance, current_sample, proposed_sample,\
                  DR_proposed_sample, DR_proposal_logpdf):
    """
    This is an adaptation of Alex's code and format to the DR case.

    Parameters
    ----------
    current_target_logpdf : float
        Target logpdf of the current sample in the chain f_X(x^{(k)}).
    DR_proposed_target_logpdf : float
        Target logpdf of the delayed rejection proposed sample f_X(y_2).
    prop_forward_logpdf : float
        Forward proposal logpdf q_1(y_1 \mid x^{(k)}).
    back_prop_prop_logpdf : float
        Proposal logpdf backward between proposals q_1(y_1 \mid y_2).
    proposal_acceptance : float
        Original proposal acceptance probability a1(x^{(k)},y_1).
    back_prop_prop_acceptance : float
        Backward between proposals acceptance probability a1(y_2,y_1).
    current_sample : vector (d,)
        Current sample x^{(k)}.
    proposed_sample : vector (d,)
        Rejected proposal sample y_1.
    DR_proposed_sample : vector (d,)
        DR-proposed sample y_2.
    DR_proposal_logpdf : function (y_2,y_1,x)
        -> returns logpdf of y_2 \mid y_1, x.
    
    Returns
    -------
    a2 : float
        New acceptance probability.

    """
    DR_prop_reverse_logpdf = DR_proposal_logpdf(current_sample,\
                                        proposed_sample, DR_proposed_sample)
    DR_prop_forward_logpdf = DR_proposal_logpdf(DR_proposed_sample,\
                                            proposed_sample, current_sample)
    log_a1_y2_y1 = np.log(1-back_prop_prop_acceptance)
    log_a1 = np.log(1-proposal_acceptance)
    check = DR_proposed_target_logpdf - current_target_logpdf +\
        back_prop_prop_logpdf - prop_forward_logpdf + DR_prop_reverse_logpdf\
            - DR_prop_forward_logpdf + log_a1_y2_y1 - log_a1
    if check < 0:
        return np.exp(check)
    else:
        return 1
            
def mhmcmc_DRAM(proposed_starting_sample, num_samples, target_logpdf,\
            get_proposal, use_AM=False,AM_burnin=0,use_DR=False,get_DR=None,\
                report_interval=None):
    """Metropolis-Hastings MCMC
    
    I have modified Alex's mhmcmc function to support (optionally) DR/AM
    methods.
    
    Inputs
    ------
    proposed_starting_sample : (d, )
        the proposed initial sample, will be modified by initialization
    num_sample : positive integer, the number of total samples
    target_logpdf : function(x) 
        -> logpdf of the target distribution
    get_proposal : function (ii,samples[ii-1],*args)
        Returns a new proposal sampler and a tuple of args that will need to
        be passed back in at the next iteration. If the proposal update scheme
        is not used, this is only used once to fetch the sampler.
        If the update scheme is used, the sampler is recursively updated at
        each iteration, but the new sampler may not be used yet depending on
        ADAPTING burn-in (i.e., seeding the adapter with some initial samples
        first).
        Also returns new proposal logpdf function (x,y) which gives the logpdf
        of the proposal for y | x.
    use_AM : bool, optional
        Logical flag that tells mhmcmc_DRAM to use an adaptive metropolis
        scheme when True. The default is False.
    AM_burnin : int, optional
        When 'use_AM' is True, 'AM_burnin' determines after how many total
        samples (including the initial sample) to begin adapting.
    use_DR : bool, optional
        Logical flag that tells mhmcmc_DRAM to use a delayed rejection scheme
        when True. The default is False.
    get_DR_sampler : function(*args)
        Consistent with 'get_proposal', this function should take in
        the tuple of arguments 'args' and return a sampler and logpdf for the
        second stage of the DR scheme.
    report_interval : int or None
        The MCMC loop prints a progress report with the current sample,
        current sample number, and acceptance ratio so far every
        'report_interval' samples. If unspecified, no progress reports are
        issued.
        The default is None.
    
    Returns
    -------
    Samples: (num_samples, d) array of samples
    accept_ratio: ratio of proposed samples that were accepted
    """

    d = proposed_starting_sample.shape[0]
    samples = np.zeros((num_samples, d))
    #initialize samples
    
    
    proposal_sampler, proposal_logpdf, samples[0,:], args = get_proposal(0,\
                 proposed_starting_sample ,AM_burnin, None, None,\
                     target_logpdf)
    #initialize the proposal sampler
    #get_proposal has two syntaxes
    #the first for k=0 (initialization), where the target_logpdf must be
    #passed in to return an initial sampler (e.g. by Laplace approximation)
    #the second syntax (seen below for k>0) simply passes in the args previous
    #iterations passed out, which should be sufficient to update the sampler
    # (efficiently!)
    #also takes in the number of non-update iterations to control the behavior
    #of args (in this case, the inverse covariance and sqrt_cov args need to
    #reflect the base sampler instead, etc.)
    #also takes in the previous samplers, as it needs to discriminate whether
    #to update them based on the burnin; for initializing, can just pass in
    #None for these, since they're not used
    #in initialization syntax, must also return initial sample
    
    current_target_logpdf = target_logpdf(samples[0, :])
    #evaluate the logpdf of the starting sample
    
    num_accept = 0
    #start counting the number of accepted samples
    num_DR = 0
    #start counting the number of delayed rejection steps taken
    t0 = t.time()
    #pull a starting time (for progress reports)
    for ii in range(1, num_samples):
        # propose
        proposed_sample = proposal_sampler(samples[ii-1, :])
        proposed_target_logpdf = target_logpdf(proposed_sample)
        
        # determine acceptance probability
        a, prop_forward_logpdf = mh_acceptance_prob(current_target_logpdf,\
            proposed_target_logpdf, samples[ii-1,:], proposed_sample,\
                proposal_logpdf)
        
        # Accept or reject the sample
        if a == 1: #guaranteed to accept
            samples[ii, :] = proposed_sample
            current_target_logpdf = proposed_target_logpdf
            num_accept += 1
        else:
            u = np.random.rand()
            if u < a: # accept
                samples[ii, :] = proposed_sample
                current_target_logpdf = proposed_target_logpdf
                num_accept += 1
            elif use_DR: # reject first sample, and check to determine if DR
                # is being used
                num_DR += 1
                #increment counter for DR (used in time estimate)
                DR_proposal_sampler, DR_proposal_logpdf = get_DR(*args)
                #get second-level sampler and logpdf funcs
                DR_proposed_sample = DR_proposal_sampler(samples[ii-1,:],\
                                                         proposed_sample)
                #technically needs to take in proposed sample, thought the
                #RW we explore here won't use it I don't believe
                DR_proposed_target_logpdf = target_logpdf(DR_proposed_sample)
                #evaluate target logpdf at new proposed sample
                ay2_y1, y2_y1_prop_logpdf = mh_acceptance_prob(\
                   DR_proposed_target_logpdf, proposed_target_logpdf, \
                       DR_proposed_sample, proposed_sample, proposal_logpdf)
                a2 = DR_mh_acceptance_prob(current_target_logpdf,\
                   DR_proposed_target_logpdf, prop_forward_logpdf,\
                       y2_y1_prop_logpdf, a, ay2_y1, samples[ii-1,:],\
                           proposed_sample, DR_proposed_sample,\
                               DR_proposal_logpdf)                
                if a2 == 1: #guaranteed to accept
                    samples[ii,:] = DR_proposed_sample
                    current_target_logpdf = DR_proposed_target_logpdf
                    num_accept += 1
                else:
                    u2 = np.random.rand()
                    if u2 < a2: #accept
                        samples[ii,:] = DR_proposed_sample
                        current_target_logpdf = DR_proposed_target_logpdf
                        num_accept += 1
                    else: #reject
                        samples[ii,:] = samples[ii-1,:]
                        #don't update logpdf
                        #don't update samples
            else: #if not using DR, then we just take the new sample
                samples[ii, :] = samples[ii-1, :]
                
        if use_AM: #if adaptive metropolis scheme is active
            proposal_sampler, proposal_logpdf, args = get_proposal(ii+1,\
               samples[ii,:], AM_burnin, proposal_sampler,\
                   proposal_logpdf, *args)
            #update sampler
        
        if report_interval is not None:
            if ii % report_interval == 0: #if it's a multiple of the interval
                print('Iteration:', ii) #report iteration number
                print('Sample:', samples[ii,:]) #report current sample
                print('Acceptance Ratio', num_accept / ii)
                #report acceptance ratio
                print('Delayed Ratio',num_DR / ii)
                ElT = t.time() - t0 #compute elapsed time,
                print('Elapsed Time:',\
                      ElT // 3600, 'hr',\
                          (ElT % 3600) // 60, 'min',\
                              ((ElT % 3600) % 60) // 1, 'sec')
                ETtC = ElT/ ii * (num_samples-ii)
                #calculate estimated time to completion in seconds
                #from number of iterations so far
                print('Estimated Time Remaining:',\
                      ETtC // 3600, 'hr',\
                          (ETtC % 3600) // 60, 'min',\
                              ((ETtC % 3600) % 60) // 1, 'sec')
                #print out estimated time remaining
                #ws.Beep(263,250) #beep at middle C
                
    return samples, num_accept / float(num_samples-1)

def get_adaptive_GRW(k, x_k, burn_in, previous_proposal_sampler,\
                     previous_proposal_logpdf, *args,\
                          initializer=None, xi=0, lower=True):
    #was inadvertently using the upper Cholesky decomposition earlier, which I
    # don't believe is correct for what we're doing
    d = x_k.size
    sd = 2.4**2/d
    #scale factor
    if k==0: #if things need to be initialized

        target_logpdf, = args
        #only passed argument is the target_logpdf; comma tells it to unpack
        if initializer is not None:
            #if an initializer function is provided
            next_mean, next_covariance = initializer(x_k, target_logpdf)
            #call it with arguments the proposed starting sample and the
            #target logpdf to provide the next mean (starting sample) and
            #next covariance (starting covariance)
        else:
            #if an initializer is not provided
            next_mean, next_covariance = x_k, np.eye(d)
            #starting proposal is just a standard normal at the proposed
            #starting point
        starting_sample = next_mean
        #the intialization scheme returns also the starting sample, which is
        #syntactically different from future potential calls to this function            
        
        #the initial value for the covariance will not be used in the
        #adaptation, because k-1 will be zero
        next_sqrt_cov = la.cholesky(next_covariance,lower=lower)
        def next_proposal_sampler(current_sample):
            y = next_sqrt_cov @ np.random.randn(current_sample.size) +\
                current_sample
            return y
        #this covariance will be needed by the DR sampler to be scaled
        next_inv_cov = la.inv(next_covariance)
        def next_proposal_logpdf(current_sample,proposed_sample):
            dist = (proposed_sample - current_sample) @ next_inv_cov @\
                (proposed_sample - current_sample)
            return -1/2 * dist
        #also needed by potential DR sampler
        
        return next_proposal_sampler, next_proposal_logpdf, starting_sample,\
            (next_mean, next_covariance, next_sqrt_cov, next_inv_cov)
        #need to initialize these quantities
    else:
        previous_mean, previous_covariance, previous_sqrt_cov,\
            previous_inv_cov = args
        #otherwise, need to be input from previous iteration
        #unpack arguments
        #the inputs previous_sqrt_cov and previous_inv_cov are unused here,
        #but are used for the DR, so I pass them around in case they're needed
        # to avoid having to compute them again
        next_mean = 1/(k+1) * x_k + k/(k+1) * previous_mean
        #update mean of adaptive Gaussian
        
        next_covariance = (k-1)/k * previous_covariance + sd/k * (\
          xi*np.eye(d) + k*np.outer(previous_mean,previous_mean) - \
              (k+1)*np.outer(next_mean,next_mean) + np.outer(x_k,x_k))
        #calculate covariance for next sampler
        if k <= burn_in: #if we haven't finished burning into the AM scheme
            next_proposal_sampler, next_proposal_logpdf, next_sqrt_cov,\
                next_inv_cov = previous_proposal_sampler,\
                    previous_proposal_logpdf, previous_sqrt_cov,\
                        previous_inv_cov
        else:
            next_sqrt_cov = la.cholesky(next_covariance,lower=lower)
            def next_proposal_sampler(current_sample):
                y = next_sqrt_cov @ np.random.randn(current_sample.size)\
                    + current_sample
                return y
            #define a Guassian sampler
            next_inv_cov = la.inv(next_covariance)
            def next_proposal_logpdf(current_sample, proposed_sample):
                dist = (proposed_sample - current_sample) @ next_inv_cov @\
                    (proposed_sample - current_sample)
                return -1/2 * dist
            #define a Gaussian logpdf function
    return next_proposal_sampler, next_proposal_logpdf, (next_mean,\
                         next_covariance, next_sqrt_cov, next_inv_cov)
#return functions and args in the form required
def get_DR_GRW(*args, gamma=0.5):
    current_mean, current_covariance, current_sqrt_cov, current_inv_cov = args
    DR_sqrt_cov = np.sqrt(gamma)*current_sqrt_cov
    #since covariance is constant times existing covariance, don't have to
    #calculate decomposition again, can just scale
    def DR_proposal_sampler(current_sample, proposed_sample):
        y = DR_sqrt_cov @ np.random.randn(current_sample.size) +\
            current_sample
        return y
    #define a gaussian random walk sampler with a covariance reduced by a
    #factor gamma; #we ignore the previously proposed sample because our RW
    # does not use that information
    DR_inv_cov = current_inv_cov/gamma
    #since covariance is just a scale, don't have to calculate inverse again!
    def DR_proposal_logpdf(DR_proposed_sample, proposed_sample,\
                           current_sample):
        dist = (DR_proposed_sample - current_sample) @ DR_inv_cov @\
            (DR_proposed_sample - current_sample)
        return -1/2 * dist
    #for our Gaussian random walk, we don't use the information about where we
    #originally proposed to determine our next proposed point, so we ignore
    #this information in the calculation of the logpdf
    return DR_proposal_sampler, DR_proposal_logpdf

def init_GRW_opt(x_k, target_logpdf, nug=None, opt_bounds=None):
    
    d = x_k.size
    sd = 2.4**2/d
    
    def opt_fun(x): return -target_logpdf(x)
    #define a function as the negative of the logpdf
    #need to get hessian approximation after identifying the maximum
    #a posteriori point
    options = {'disp': True, 'fatol': 1e-2, 'bounds': opt_bounds}
    res = opt.minimize(opt_fun,x_k, method='Nelder-Mead', options=options)
    if opt_bounds is not None: #if bounds are found
        bounds = np.array(opt_bounds).T
        steps = np.min([res.x - bounds[0],bounds[1]-res.x],axis=0)
        #determine step sizes in comparison to bounds
    else:
        steps = res.x #else just use the plain value
    steps = np.abs(1e-2 * steps)
    neg_hess = approx_hessian(opt_fun, res.x, steps)
    #approximate negative hessian of logpdf using finite differences
    #at location of found optimum
    #step sizes are chosen to be 1e-1 times the magnitude of the function
    #limited by potential bounds of invalidity
    neg_hess_inv = la.inv(neg_hess)
    #minimizing the negative of the logpdf is maximizing the logpdf4
    #returns a hess_inv which is useful for defining cov
    #we have the inverse negative (since we worked with the - logpdf),
    #so we have the (base) covariance
    next_covariance = sd*neg_hess_inv
    if nug is not None:
        next_covariance += nug * np.eye(d)
    #starting sample is MAP point
    next_mean = res.x
    return next_mean, next_covariance

def init_GRW_opt_noCorr(x_k, target_logpdf, nug=None, opt_bounds=None):
    
    d = x_k.size
    sd = 2.4**2/d
    
    def opt_fun(x): return -target_logpdf(x)
    #define a function as the negative of the logpdf
    #need to get hessian approximation after identifying the maximum
    #a posteriori point
    # options = {'disp': True, 'maxiter': 100, 'finite_diff_rel_step': 1}
    # res = opt.minimize(opt_fun,x_k,method='BFGS', options=options)
    options = {'disp': True, 'fatol': 1e-2, 'bounds': opt_bounds}
    res = opt.minimize(opt_fun,x_k, method='Nelder-Mead', options=options)
    if opt_bounds is not None: #if bounds are found
        bounds = np.array(opt_bounds).T
        steps = np.min([res.x - bounds[0],bounds[1]-res.x],axis=0)
        #determine step sizes in comparison to bounds
    else:
        steps = res.x #else just use the plain value
    steps = np.abs(1e-3 * steps)
    neg_hess = approx_hessian(opt_fun, res.x, steps)
    #approximate negative hessian of logpdf using finite differences
    #at location of found optimum
    #step sizes are chosen to be 1e-1 times the magnitude of the function
    #limited by potential bounds of invalidity
    neg_hess = np.diag(np.diag(neg_hess))
    #ignore correlation from the hessian
    neg_hess_inv = la.inv(neg_hess)
    #minimizing the negative of the logpdf is maximizing the logpdf4
    #returns a hess_inv which is useful for defining cov
    #we have the inverse negative (since we worked with the - logpdf),
    #so we have the (base) covariance
    next_covariance = sd*neg_hess_inv
    #starting sample is MAP point
    next_mean = res.x
    if nug is not None:
        #add nugget to covariance for conditioning
        next_covariance += nug * np.eye(d)
    return next_mean, next_covariance

def init_GRW_preDef(x_k, target_logpdf, cov=None):
    #allows for a predefined starting covariance
    if cov is not None:
        return x_k, cov
    else:
        #this duplicates not passign in an initializer to the GRW
        return x_k, np.eye(x_k.size)
    

def approx_hessian(func,center_point,step_sizes):
    d = center_point.size #determine dimension
    if step_sizes.size == 1: #if a single step size is passed in for all dims
        step_sizes = step_sizes * np.ones(d) #make into vector
    hess = np.zeros((d,d)) #preallocate hessian
    center = func(center_point)
    #calculate function at point of interest
    
    #calculate diagonal entries using linear 3-pt stencils over each dimension
    for ii in range(d): #for each dimension
        plus = func(center_point + step_sizes * np.eye(1,d,ii).flatten())
        #one step forward in relevant dimension
        minus = func(center_point - step_sizes * np.eye(1,d,ii).flatten())
        #one step backward in relevant dimension
        hess[ii,ii] = (plus - 2*center + minus) / step_sizes[ii]**2
    
    #calculate mixed derivatives using 4-pt stencils over each subplane
    for jj in range(1,d): #go along off-diagonal columns
        for ii in range(jj): #go along off-diagonal rows
            plus_plus = func(center_point + step_sizes * (np.eye(1,d,ii)\
                                              + np.eye(1,d,jj)).flatten())
            #one step forward along each dimension of plane
            minus_minus = func(center_point - step_sizes * (np.eye(1,d,ii)\
                                                + np.eye(1,d,jj)).flatten())
            #one step backward along each dimension of plane
            plus_minus = func(center_point + step_sizes * (np.eye(1,d,ii)\
                                               - np.eye(1,d,jj)).flatten())
            #one step forward along first dimension, one step back along 2nd
            minus_plus = func(center_point - step_sizes * (np.eye(1,d,ii)\
                                               - np.eye(1,d,jj)).flatten())
            #step back in 1st dim, step forward in 2nd dim
            hess[ii,jj] = hess[jj,ii] = (plus_plus + minus_minus - plus_minus\
                                         - minus_plus) / step_sizes[ii] /\
                step_sizes[jj]
            #assign mixed derivative to either diagonal
    
    return hess
    #return the approximation of the hessian
    
def write2DArray(array,fname):
    with open(fname,'a') as fid:
        for line in array[:,:]:
            print(*line,file=fid,sep='\t')

# def read2DArray(fname):
#     matOut = [];
#     with open(fname,'r') as fid:
#         for line in fid:
#             pass
            
    
#%% Helper functions for simple distributions
def log_gamma(x,k,theta):
    """
    Calculates the logarithm of a gamma distribution within an additive
    constant. The shape parameter is 'k' and the scale parameter is 'theta'.
    Supports broadcasting as appropriate.

    Parameters
    ----------
    x : float or vector of floats (N,)
        Argument of the distribution.
    k : float or vector of floats (N,)
        Shape parameter of the gamma distribution.
    theta : float or vector of floats (N,)
        Scale parameter of the gamma distribution.

    Returns
    -------
    logpdf : float or vector of floats (N,)
        Log of the gamma distribution within an additive constant.

    """
    
    logpdf = (k-1) * np.log(x) - x / theta
    #calculate logpdf within an additive constant
    #note that this additive constant is constant with respect to the argument
    #only, not constant with respect to the parameters
    return logpdf
    #return it
    
def log_gauss(x,mu,sigma2):
    """
    Calculates the logarithm of a 1-D gaussian distribution within an additive
    constant. The mean is 'mu' and the variance is 'sigma2'. Supports
    broadcasting as appropriate.

    Parameters
    ----------
    x : float or vector of floats (N,)
        Argument of the distribution.
    mu : float or vector of floats (N,)
        Mean of the distribution.
    sigma : float or vector of floats (N,)
        Variance of the distribution.

    Returns
    -------
    logpdf : float or vector of floats (N,)
        Log of the gaussian within an additive constant.

    """
    
    logpdf = - 0.5 * np.log(sigma2) - 0.5 * (mu-x)**2 / sigma2
    return logpdf

def log_uniform(x, lb=-np.inf, ub= np.inf):
    """
    Calculates the logarithm of a uniform distribution within an additive
    constant. Essentially, just checks if the point is within bounds. Supports
    broadcasting and multidimensionality implicitly (you could, for example,
    feed in 'lb' and 'ub' as vectors and 'x' as a vector for a single-point
    estimate for a vector of quantities described by uniform bounds or even in
    that case pass 'x' in as an array to ccompute multiple at the same time).

    Parameters
    ----------
    x : float or vector of floats (N,)
        Argument of the distribution.
    lb : float, optional
        The lower bound of the uniform distribution. The default is -np.inf.
    ub : float, optional
        The upper bound of the uniform distribution. The default is np.inf.

    Returns
    -------
    logpdf : float or vector of floats (N,)
        The logpdf within an additive constant (essentially, just arbitrarily
        negative when not in bounds).

    """
    
    valid_mask = (x >= lb) & (x <= ub)
    #compute where the values are in bound
    logpdf = np.zeros(x.shape)
    #preallocate for the logpdf
    logpdf[~valid_mask] = -np.inf
    #assign negative infinity to where x is out of bounds, and zero elsewhere
    #valid within a multiplicative constant
    return logpdf
    
#%% Models

def current_model(params,voltage,subs,props,beams,geoms,es_models=None):
    """
    This function is a vectorized implementation that computes the predicted
    current given model parameters, voltage conditions, substrate properties,
    propellant properties, beam properties, and emitter geometries.
    
    It expects to compute its predictions for Nd voltage setpoint and for Nr
    realizations of all the parameters. The number of emitters in the array is
    implicitly specified by the size of geoms. The dimension of each input is
    important as noted, except that generally the dimension corresponding to
    Nr can be discarded for computing only a single realization.
    
    It includes an optional argument to prescribe an E/V for each realization
    of each emitter (e.g. as provided by electrostatic simulation of the
    geometry) rather than use a modified form of the hyperboloidal
    approximation.

    Parameters
    ----------
    params : vector of floats (4,)
        A vector containing the model parameters: the ionic emission offset
        \zeta_1, the ionic emission slope \zeta_2, the pooling radisu b_0, and
        the maximum number of sites N_{max}.
    voltage : vector of floats (Nd,)
        A vector of voltages at which the model is to be evaluted (e.g. all of
        the voltage data used in the inference).
    subs : array of floats (3,Nr) or (2,Nr)
        An array containing Nr samples/realizations of the substrate
        properties: EITHER the reservoir pore radius, the emitter bulk pore
        radius, and the emitter porosity if fed in as a (3,Nr) array OR the
        reservoir pore radius and substrate permeability if fed in as a (2,Nr)
        array. The difference will automatically be detected.
    props : array of floats (4,Nr)
        An array containing Nr samples/realizations of the propellant 
        properties: the surface tension, the conductivity, the density, and
        the viscosity.
    beams : array of floats (1,Nr)
        An array containing Nr samples/realizations of the beam properties:
        the charge-to-mass ratio. The first dimension must be retained.
    geoms : array of floats (6,Ne,Nr)
        An array containing Nr samples/ realizations of the geometries of each
        of the Ne emitters: the tip radius of curvature, the gap distance, the
        aperture radius, the cone half-angle, the emitter height, and the
        local pore radius. If Nr is greater than 1, the dimension of Ne must
        be retained, even for a single-emitter array.
    es_models : array of floats (1,Ne,Nr), optional
        An array containing the numerically-computed electrostatic mapping
        from emitter voltage to electric field magnitude at the emitter tip,
        for each realization (Nr) of each emitter (Ne) geometry.
        If provided, the model will substitute these values for those yielded
        by the ad-hoc modification to the Martinez-Sanchez approximation.
        The default is None (not used).        

    Returns
    -------
    current : array of floats (Nd,Nr)
        The total current predicted at each voltage 1...Nd for each sample/
        realization of the nuisance parameters 1...Nr.

    """
    ##convert inputs to ndarray if necessary
    if type(voltage) is not np.ndarray:
        voltage = np.array(voltage)
    if type(subs) is not np.ndarray:
        subs = np.array(subs)
    if type(props) is not np.ndarray:
        props = np.array(props)
    if type(beams) is not np.ndarray:
        beams = np.array(beams)
    if type(geoms) is not np.ndarray:
        geoms = np.array(geoms)
    
    ##extract model params from input vector
    ion_emis_offset = params[0]
    ion_emis_slope = params[1]
    pool_radius = params[2]
    
    ##determine dimensions for vectorization and broadcasting
    Nd = voltage.shape[0] #number of voltage set points/data to calculate at
    Ne = geoms[0].shape[0] #number of emitters in the array
    Nr = subs[0].shape[0] #number of samples/realizations
    
    ##reshape voltage
    voltage = voltage.reshape((Nd,1,1,1))
    
    ##extract individual variables from tuples and reshape as necessary
    if subs.shape[0] == 2: #if permeability specified instead of the thruple
        res_pore_radius = subs[0].reshape((1,1,Nr,1))
        permeability = subs[1].reshape((1,1,Nr,1))
        #extract substrate properties
    else:
        res_pore_radius = subs[0].reshape((1,1,Nr,1))
        bulk_pore_radius = subs[1].reshape((1,1,Nr,1))
        porosity = subs[2].reshape((1,1,Nr,1))
        #extract substrate properties (including reservoir pore radius)
        permeability = bulk_pore_radius**2 / 60 / (1-porosity)**2
        #calculate the permeability of the substrate
    #these properties are taken constant across voltages, emitters,and sites
    
    
    conductivity = props[0].reshape((1,1,Nr,1))
    surface_tension = props[1].reshape((1,1,Nr,1))
    density = props[2].reshape((1,1,Nr,1))
    viscosity = props[3].reshape((1,1,Nr,1))
    #extract propellant properties
    #these too are taken as constant across voltages, emitters, and sites
    
    charge_to_mass = beams[0].reshape((1,1,Nr,1))
    #extract beam properties (just charge to mass)
    #also taken as constant across voltages, emitters, and sites
    
    curvature_radius = geoms[0].reshape((1,Ne,Nr,1))
    gap_distance = geoms[1].reshape((1,Ne,Nr,1))
    aperture_radius = geoms[2].reshape((1,Ne,Nr,1))
    half_angle = geoms[3].reshape((1,Ne,Nr,1))
    emitter_height = geoms[4].reshape((1,Ne,Nr,1))
    loc_pore_radius = geoms[5].reshape((1,Ne,Nr,1))
    #extract geometries; #these are not a function of voltage nor do the vary
    #across sites, but do vary between emitters
    
    ##calculate some quantities common to all emitters and emission sites
    hydraulic_resistivity = viscosity / (2 * np.pi * permeability)
    #this is the quantity that appears in front of the geometric terms in the
    #expression for hydraulic resistance, and is analagous to a resistivity
    res_pressure = -2 * surface_tension / res_pore_radius
    #determine the (negative) reservoir pressure as a Laplace pressure from
    #the pore size
    
    ##compute electric field parameters
    if es_models is not None:
        #if numerical solutions for the field are provided
        if type(es_models) is not np.ndarray:
            es_models = np.array(es_models)
        #ensure they are numpy ndarrays
        es_models = es_models.reshape((1,Ne,Nr,1))
        #reshape the es_models so they match the other "geometric" parameters
        applied_field = es_models * voltage;
        #compute the applied field from the voltage...
        #it's that easy when you compute the E/V
    else:
        adj_gap_distance = np.sqrt(gap_distance**2 + aperture_radius**2)
        #compute adjusted gap distance
        emitter_hyperboloid = np.sqrt(1 / (1 + curvature_radius /\
                                           adj_gap_distance))
        #compute hyperboloidal coordinate of approximate emitter surface
        semiinterfocal_distance = adj_gap_distance / emitter_hyperboloid
        ##compute individual emitter currents
        applied_field = voltage / semiinterfocal_distance /\
            np.arctanh(emitter_hyperboloid) / (1 - emitter_hyperboloid**2)
        #compute applied field at the emitter tip
        #applied_field = applied_field * adj_gap_distance /\
        #    np.sqrt(aperture_radius**2 + adj_gap_distance**2)
    
    ##computing the number of sites for each emitter
    applied_electric_pressure = 1 / 2 * VACUUM_PERMITTIVITY * applied_field**2
    #compute the applied electric pressure
    min_cap_pressure = 2 * surface_tension / (loc_pore_radius + pool_radius)
    #compute the capillary pressure of the largest potential site
    max_cap_pressure = 2 * surface_tension / loc_pore_radius
    #compute the capillary pressure of the smallest potential site
    
    no_sites_mask = \
        applied_electric_pressure < min_cap_pressure - res_pressure
    #determine where the applied field is insufficient for any potential sites
    all_sites_mask = \
        applied_electric_pressure > max_cap_pressure - res_pressure
    #determine where the applied field is sufficient to saturate the sites
    
    num_sites = np.zeros((Nd,Ne,Nr,1))
    #preallocate for the number of sites
    if params.size == 4:
        #if max number of sites specified as model parameter
        max_sites = params[3] #pull from structure
    else: #otherwise, estimate on a per-emitter/realization basis
        A_emission = 2 * np.pi * curvature_radius**2 *\
            ( 1 - np.sin(half_angle))
        #estimate total area available for emission
        A_Taylor = np.pi * ((pool_radius + loc_pore_radius)/2)**2
        #estimate average area of Taylor cones
        #really an implicit solve type operation, but could probably implement
        #an invertable continuous relaxation to approximate it
        max_sites = A_emission / A_Taylor
        #estimate maximum number of sites
    
    num_sites[all_sites_mask] =\
        np.broadcast_to(np.floor(max_sites),(Nd,Ne,Nr,1))[all_sites_mask]
    #where sites are saturated, number of sites is N_max (floored)
    neither_mask = ~(no_sites_mask | all_sites_mask)
    #determine where neither condition is true and the number of sites must
    #be calculated
    
    min_site_radius =\
        2 * np.broadcast_to(surface_tension,(Nd,Ne,Nr,1))[neither_mask] /\
            (1/ 2 * VACUUM_PERMITTIVITY * applied_field[neither_mask]**2 +\
             np.broadcast_to(res_pressure,(Nd,Ne,Nr,1))[neither_mask])
    #for all sites in between, compute the minimum site size that onsets

    num_sites[neither_mask] =\
        np.floor(1 + (np.broadcast_to(max_sites,(Nd,Ne,Nr,1))[neither_mask]\
          - 1) * (1 - (min_site_radius -\
               np.broadcast_to(loc_pore_radius,(Nd,Ne,Nr,1))[neither_mask]) /\
                   pool_radius)**4)
    #use the minimum site size to compute the number of emission sites at
    #these conditions
    
    Ns_max = num_sites.flatten().max().astype(int)
    #calculate the maximum number of active sites amongst all emitters
    #across all voltages
    site_nums = np.arange(Ns_max).reshape((1,1,1,Ns_max)) + 1
    #each potential site denoted by an index (j in notes)
    active_mask = site_nums <= num_sites
    #determine mask of active sites where calculations must be performed
    #via broadcasting, this mask should now have full dimensions
    dims = (Nd,Ne,Nr,Ns_max)
    #create full dimension
    

    site_radius =\
        pool_radius *(1 - ((np.broadcast_to(site_nums,dims)[active_mask]- 1)/\
           (np.broadcast_to(max_sites,dims)[active_mask] - 1))**0.25) +\
              np.broadcast_to(loc_pore_radius,dims)[active_mask]
    #compute the site radius at only the active sites
    #note that this abandons the dimensioned structure for a masked structure
    #that is, the computations are computed over a flattened form at only the
    #active sites
    # if np.any(site_radius < 0):
    #     print('Negative Site Radius')
    char_pressure = 2 * np.broadcast_to(surface_tension,dims)[active_mask] /\
        site_radius
    #compute characteristic pressure
    char_elec_field = np.sqrt(2 * char_pressure / VACUUM_PERMITTIVITY)
    #compute characteristic electric field
    onset_field = np.sqrt((2 * (char_pressure -\
                           np.broadcast_to(res_pressure,dims)[active_mask]))/\
                          VACUUM_PERMITTIVITY)
    #calculate onset fields for each active emission site
    hydraulic_impedance = num_sites * hydraulic_resistivity /\
        (1 - np.cos(half_angle)) * (np.tan(half_angle) / curvature_radius -\
            np.cos(half_angle)/ emitter_height)
    #compute the hydraulic impedance associated with each site
    dimless_applied_field =\
        np.broadcast_to(applied_field,dims)[active_mask] / char_elec_field
    #compute the dimenionless applied field for active emission sites
    dimless_onset_field = onset_field / char_elec_field
    #compute the dimensionless onset field for active emission sites
    dimless_res_pressure = np.broadcast_to(res_pressure,dims)[active_mask] /\
        char_pressure
    #calculate dimenionless reservoir pressure
    dimless_hydraulic_impedance =\
        np.broadcast_to(conductivity,dims)[active_mask] * char_elec_field *\
        site_radius**2 *\
            np.broadcast_to(hydraulic_impedance,dims)[active_mask] /\
            (char_pressure * np.broadcast_to(density,dims)[active_mask] *\
             np.broadcast_to(charge_to_mass,dims)[active_mask])
    #calculate dimensionless hydraulic impedance C_R
    #should maybe change notation to \hat{R}_H?
    dimless_current =\
        (ion_emis_offset + ion_emis_slope * (dimless_applied_field -\
             dimless_onset_field) + dimless_res_pressure) /\
                dimless_hydraulic_impedance
    #compute dimensionless current emitted by each active site
    site_current = np.zeros(dims)
    #preallocate for the site current
    site_current[active_mask] = dimless_current *\
        np.broadcast_to(conductivity,dims)[active_mask] *\
        char_elec_field * site_radius**2
    #compute from each active site
    #all others don't emitter current and were preallocated as zero
    
    site_current[site_current < 0] = 0
    #this last step may be crucial
    #there are some cases where for sufficiently low field beyond the onset
    #field, the offset value isn't actually sufficient to overcome the
    #retardation by the reservoir pressure, and so negative current is an
    #unphysical case that should just be zero instead
    #it would be interesting to see how this compares with Coffman's results
    #with negative reservoir pressure
    #was there still current, if so, how much? was it retarded by the negative
    #reservoir pressure?
    
    current = np.sum(site_current,axis=(1,3))
    #hmm, now the ordering of dimensions seems meh, but I think it was easier
    #to add sites as the fourth dimension then to have inserted it as the
    #third
    return current
#%% calculating the log-likelihood and log prior

def pm_loglike(params, voltage, obs_current, noise_var, subs_sampler,\
               props_sampler, beams_sampler, geoms_sampler,\
                   block_samples=5000, num_blocks=10):
    """
    This function calls the current model and takes in the experimental data
    and uncertainties to compute the pseudomarginal estimation of the log
    likelihood.
    
    In addition to the model parameters for which to evaluate the model and
    the voltage/current data, it also takes in several function handles
    which it uses to sample from the nuisance parameters. These functions
    should be configured to sample consistent with the dimensional
    requirements of the vectorized current model.
    
    It is desirable for the current model to be written using vectorized
    operations to save on computational time. However, pracitcally I have
    run into memory limits, and so this function has also implemented a loop
    over vectorized model computations in order to support larger sample sizes
    while still vectorizing as much as possible. This works by having a block
    of samples of size 'block_samples' for which the computation is performed,
    then repeating this process 'num_blocks' times, resampling nuisance
    parameters each time. This effectively creates a number of realizations
    N_r = block_samples * num_blocks.

    Parameters
    ----------
    params : vector of floats (4,)
        A vector containing the model parameters: the ionic emission offset
        \zeta_1, the ionic emission slope \zeta_2, the pooling radius b_0, and
        the maximum number of sites N_{max}.
    voltage : vector of floats (Nd,)
        A vector of voltages at which the model is to be evaluted (e.g. all of
        the voltage data used in the inference).
    obs_current : vector of floats (Nd,)
        A vector of (experimental) current data corresponding to 'voltage'.
    noise_var : vector of floats (Nd,)
        Assuming a Gaussian likelihood in the data, the corresponding
        uncertainty (as a VARIANCE), in each datum.
    subs_sampler : function (block_samples)
        A function that returns 'block_samples' (Nr) samples of the substrate
        properties.
    props_sampler : function (block_samples)
        A function that returns 'block_samples' (Nr) samples of the propellant
        properties.
    beams_sampler : function (block_samples)
        A function that returns 'block_samples' (Nr) samples of the beam
        properties.
    geoms_sampler : function (block_samples)
        A function that returns 'block_samples' (Nr) samples of the geometries
        for each emitter.
    block_samples : int, optional
        The number of realizations/samples of the likelihood to compute per
        call of the function 'current_model'. This value is limited by
        available memory.
        The default is 5000.
    n_blocks : int, optional
        The number of blocks to compute. Not (strongly) limited by memory, so
        is used to achieve the practical desired sample size.
        The default is 10.

    Returns
    -------
    log_like : float
        A pseudomarginal estimation of the logarithm of the likelihood, within
        a constant offset.

    """
    
    Nd = voltage.size
    #determine number of voltage/current data
    model_current = np.empty((Nd,0))
    #create an empty array that we will horizontally concatenate as we compute
    #new blocks
    for ll in range(num_blocks): #for each block of realizations
        subs = subs_sampler(block_samples)
        props = props_sampler(block_samples)
        beams = beams_sampler(block_samples)
        geoms = geoms_sampler(block_samples)
        #sample the nuisance parameters for each realization
    
        block_current = current_model(params,voltage,subs,props,beams,geoms)
        #compute the model at each voltage for that block
        model_current = np.concatenate((model_current, block_current), axis=1)
        #concatenate along the dimension of realizations
    obs_current = obs_current.reshape(Nd,1)
    noise_var = noise_var.reshape(Nd,1)
    #reshape to ensure as (Nd,1)
    log_like = -1/2 * (model_current - obs_current)**2 / noise_var
    #compute the log likelihood of each realization of each datum
    #assuming independent noise and Gaussian likelihood
    log_like = np.sum(log_like,axis=0)
    #sum down the dimension of data to get Nr realizations of the likelihood
    #of all data
    #the log likelihood realizations are now flat
    max_like = np.max(log_like) #compute maximum likelihoo magnitude
    log_like = max_like + np.log(np.sum(np.exp(log_like - max_like)))
    #here I've omitted a factor -ln(Nr), since it will be constant between
    #evaluations
    return log_like
    #return the log_likelihood within a constant (provided the number of
    #samples used to compute the realizations does not change)
    
def ap_pm_loglike(params, voltage, obs_current, noise_var, subs, props,\
                  beams, geoms, es_models = None):
    """
    This function is similar to pm_loglike, except that it implements an
    approximately pseuodmarginal estimate by using a fixed set of samples from
    the nuisance parameters, rather than drawing them from the sampler.
    The number of realizations is enforced by the dimension of the inputs,
    consistent with the vectorized current model.
    
    It calls the current model and takes in the experimental data
    and uncertainties to compute the approximately pseudomarginal estimation
    of the log likelihood.
    
    In addition to the model parameters for which to evaluate the model and
    the voltage/current data.
    
    It is memory-limited for now.

    Parameters
    ----------
    params : vector of floats (3,) or (4,)
        A vector containing the model parameters: the ionic emission offset
        \zeta_1, the ionic emission slope \zeta_2, the pooling radius b_0[,
        and the maximum number of sites N_{max}.]
    voltage : vector of floats (Nd,)
        A vector of voltages at which the model is to be evaluted (e.g. all of
        the voltage data used in the inference).
    obs_current : vector of floats (Nd,)
        A vector of (experimental) current data corresponding to 'voltage'.
    noise_var : vector of floats (Nd,)
        Assuming a Gaussian likelihood in the data, the corresponding
        uncertainty (as a VARIANCE), in each datum.
    subs : array of floats (3,Nr) or (2,Nr)
        An array containing Nr samples/realizations of the substrate
        properties: EITHER the reservoir pore radius, the emitter bulk pore
        radius, and the emitter porosity if fed in as a (3,Nr) array OR the
        reservoir pore radius and substrate permeability if fed in as a (2,Nr)
        array. The difference will automatically be detected.
    props : array of floats (4,Nr)
        An array containing Nr samples/realizations of the propellant 
        properties: the surface tension, the conductivity, the density, and
        the viscosity.
    beams : array of floats (1,Nr)
        An array containing Nr samples/realizations of the beam properties:
        the charge-to-mass ratio. The first dimension must be retained.
    geoms : array of floats (6,Ne,Nr)
        An array containing Nr samples/ realizations of the geometries of each
        of the Ne emitters: the tip radius of curvature, the gap distance, the
        aperture radius, the cone half-angle, the emitter height, and the
        local pore radius. If Nr is greater than 1, the dimension of Ne must
        be retained, even for a single-emitter array.
    es_models : array of floats (1,Ne,Nr), optional
        An array containing the numerically-computed electrostatic mapping
        from emitter voltage to electric field magnitude at the emitter tip,
        for each realization (Nr) of each emitter (Ne) geometry.
        If provided, the model will substitute these values for those yielded
        by the ad-hoc modification to the Martinez-Sanchez approximation.
        The default is None (not used).    

    Returns
    -------
    log_like : float
        An approximately pseudomarginal estimation of the logarithm of the
        likelihood, within a constant offset.

    """
    
    Nd = voltage.size
    #determine number of voltage/current data
    if es_models is not None:
        #if electrostatic simulation results are provided
        model_current = current_model(params, voltage, subs, props, beams,\
                                      geoms, es_models)
        #model eval with es_models provided
    else:
        #if not supplied
        model_current = current_model(params,voltage,subs,props,beams,geoms)
        #compute other way
    #compute model predictions for all realizations, emitters, and voltages
    #for the given parameters
    obs_current = obs_current.reshape(Nd,1)
    noise_var = noise_var.reshape(Nd,1)
    #reshape to ensure as (Nd,1)
    log_like = -1/2 * (model_current - obs_current)**2 / noise_var
    #compute the log likelihood of each realization of each datum
    #assuming independent noise and Gaussian likelihood
    log_like = np.sum(log_like,axis=0)
    #sum down the dimension of data to get Nr realizations of the likelihood
    #of all data
    #the log likelihood realizations are now flat
    max_like = np.max(log_like) #compute maximum magnitude amongst likelihoods
    log_like = max_like + np.log(np.sum(np.exp(log_like - max_like)))
    #here I've omitted a factor -ln(Nr), since it will be constant between
    #evaluations
    return log_like
    #return the log_likelihood within a constant (provided the number of
    #samples used to compute the realizations does not change)

def gauss_sampler(num_samples,mu,cov,return_logpdf=False):
    d = mu.size #determine dimension of input
    mu = mu.reshape(d,1) #make sure input is in right shape
    L = la.cholesky(cov,lower=True)
    #perform a cholesky decomposition to get the "square root" of the cov
    samples = np.random.randn(d,num_samples)
    #generate num_samples samples from a standard normal of correct dimension
    samples = mu + L @ samples
    #scale them by the standard deviation and mean
    if return_logpdf: #if the whole shebang
        inv_cov = la.inv(cov) #invert the covariance
        dist = np.sum((samples - mu) * (inv_cov @ (samples-mu)),axis=0,\
                      keepdims=True)
        logpdf = -1/2 * dist
        #compute the logarithm of the pdf for a Gaussian for all points within
        #a constant offset
        return samples, logpdf
        #return both the samples and the logpdf
    else: #don't want/need the logpdf
        return samples
        #return just the samples
        
def delta_sampler(num_samples,x0,return_logpdf=False):
    d = x0.size #determine dimension of input
    x0 = x0.reshape(d,1) #ensure correct shape
    samples = x0 * np.ones(num_samples)
    #to sample from certainty, just reproduce the center
    if return_logpdf:
        return samples, np.zeros((1,num_samples))
        #return samples and zeros, since logpdf is constant/infinity/doesn't
        #matter
    else:
        return samples
        
def gauss_sampler_noCov(num_samples,mu,sigma,return_logpdf=False):
    d = mu.size #determine number of independent parameters to be sampled
    mu = mu.reshape(d,1)
    sigma = sigma.reshape(d,1)
    #reshape for broadcasting support
    samples = np.random.randn(d,num_samples)
    #generate num_samples samples from a standard normal of correct dimension
    samples = mu + sigma * samples
    #scale them by the standard deviation and mean
    if return_logpdf:#if the whole shebang
        logpdf = -1/2 * np.sum((samples-mu) / sigma**2 * (samples-mu),\
                               axis=0,keepdims=True)
        #compute the logarithm of the pdf for a Gaussian for all points within
        #a constant offset
        return samples, logpdf
        #return both the samples and the logpdf
    else: #don't want/need the logpdf
        return samples
        #return just the samples
        
def uniform_sampler(num_samples,lb,ub,return_logpdf=False):
    d = lb.size #determine dimension (vector) of input
    lb = lb.reshape(d,1)
    ub = ub.reshape(d,1)
    #reshape bounds to ensure proper dimension
    samples = np.random.rand(d,num_samples)
    #sample between 0 and unity for the interval
    samples = lb + (ub - lb) * samples
    #scale and offset samples over the bound of interest
    if return_logpdf:
        return samples, np.zeros((1,num_samples)) #constant logpdf in domain
    else:
        return samples


if __name__ == '__main__':
#%% some initialization and definition stuff

    #the order of parameters is:
        #emission_offset = sample[0] #(\zeta_1)
        #emission_slope = sample[1] #(\zeta_2)
        #pooling_radius = sample[2] #(b_0)
        #and, when included,
        #max_sites = sample[3] #(N_{max})
    
    lb_subs = np.array([1e-5/2])
    ub_subs = np.array([1.6e-5/2])
    mu_subs = np.array([1.51e-13])
    sigma_subs = np.array([6.04e-15])
    #substrate pore radius uniform on [1e-5,1.6e-5]/2
    #substrate permeability Gaussian with mean 1.51e-13 and stdv 6.04e-15
    def subs_sampler(num_samples):
        #define "prior" distribution over substrate parameters
        return np.concatenate((uniform_sampler(num_samples, lb_subs,\
                                               ub_subs),\
                        gauss_sampler_noCov(num_samples,mu_subs,sigma_subs)),\
                            axis = 0)
    #return uniform samples for the substrate pore radius and gaussian samples
    #for the permeability
    
    lb_props = np.array([1.146519,.050038,1279.782,.026119])
    ub_props = np.array([1.389983,.050452,1284.369,.034162])
    #conductivity uniform on [1.146519,1.389983]
    #surface tension uniform on [.050038,.050452]
    #density uniform on [1279.782,1284.369]
    #viscosity uniform on [.026119,.034162]
    def props_sampler(num_samples):
        #define "prior" distribution over propellant properties
        return uniform_sampler(num_samples, lb_props, ub_props)
        #return uniform samples over the bounds
    
    mu_beams = np.array([5.49932e5])
    sigma_beams = np.array([1.0034e4])
    def beams_sampler(num_samples):
        #define "prior" distribution over beam properties
        return gauss_sampler_noCov(num_samples, mu_beams, sigma_beams)
        pass
    
    num_emitters = 576
    #specify number of emitters
    lb_geoms = np.array([1e-5])
    ub_geoms = np.array([2e-5])
    mu_geoms = np.array([3.0e-6,4.9714e-04/2,0.26782,3.0181e-04,1.3e-06/2])
    sigma_geoms = np.array([5.2302e-6,7.1914e-06/2,0.0040,5.1302e-06,1.5e-07/2])
    #radius of curvature uniform on [1e-6,2e-6]
    #tip-to-extractor distance Gaussian with mean 3.0e-6 and stdv 5.2303e-06
    #aperture radius Gaussian with mean 4.9714e-04/2 and stdv 7.1914e-06/2
    #cone half-angle Gaussian with mean 0.26782 and stdv 0.0040
    #emitter height Gaussian with mean 3.0181e-04 and stdv 5.1302e-06
    #local pore radius Gaussian with mean 1.3e-06/2 and stdv 1.5e-07/2
    def geoms_sampler(num_samples):
        long_samples =\
            np.concatenate((uniform_sampler(num_samples*num_emitters,\
                                            lb_geoms, ub_geoms),\
                      gauss_sampler_noCov(num_samples*num_emitters, mu_geoms,\
                                          sigma_geoms)), axis=0)
        samples = long_samples.reshape((6,num_emitters,num_samples))
        return samples
    
    with open('training_data.pkl','rb') as fid:
        voltage, obs_current, noise_var = pkl.load(fid)
    #import training data from .pkl file
    
    
    pm_block_samples = 10
    pm_blocks = 1000
    #specify number of samples with which to evaluate the log likelihood
    
    def loglike(sample):
        #define a log likelihood over the sample
        #this needs to work with only 'sample' as an input
        return pm_loglike(sample, voltage, obs_current, noise_var,\
                          subs_sampler, props_sampler, beams_sampler,\
                              geoms_sampler,pm_block_samples,pm_blocks)
    
    def logprior(sample):
        #define a function that calculates the logarithm of the prior
        #for now, the prior over all 4 parameters is uniform, so they're
        #treated together here
        lb = np.array([0,0,0])#,1])
        ub = np.array([np.inf,np.inf,np.inf])#,np.inf])
        #zeta_1 uniform on 0 to infinity
        #zeta_2 uniform on 0 to infinity
        #b0 uniform on 0 to infinity
        #Nmax uniform on 1 to infinity
        return np.sum(log_uniform(sample,lb,ub))
    
    
    def logpdf(sample):
        #the logpdf must be the sum of the logprior over parameters and the
        #log likelihood
        check = logprior(sample)
        #calculate the log prior
        if check <= -np.inf:
            #if out of bounds of prior
            return check #don't actually need to evaluate the likelihood
        else:
            return loglike(sample) + check #otherwise, must and return sum
        
#%%  Perfect array deterministic sampling          
    
    
    # # opt_bounds = [(0,np.inf),(0,np.inf),(0,np.inf),(1,40)]
    
    # # def initializer(x_k,target_logpdf):
    # #     return init_GRW_preDef(x_k, target_logpdf,\
    # #                 cov=np.diag([1e-2, 1e-7,\
    # #                             1e-12]))
    # # pre-defined intializer for testing purposes
    # def initializer(x_k, target_logpdf):
    #     return init_GRW_opt(x_k,target_logpdf,nug=nug)
    
   
    # def get_proposal(k, x_k, burn_in, previous_proposal_sampler,\
    #                  previous_proposal_logpdf, *args):
    #     return get_adaptive_GRW(k, x_k, burn_in, previous_proposal_sampler,\
    #                             previous_proposal_logpdf, *args,\
    #                                 initializer=initializer, xi=xi)
    # #define a get proposal function
    # #this case, using a Gaussian random walk
    
    
    # def get_DR_prop(*args):
    #     return get_DR_GRW(*args,gamma=DR_gamma)
    # #define get proposal func for DR
    



    # subs_x0 = np.array([1.3e-5/2, 1.51e-13]).reshape((2,1))
    # # def det_subs_sampler(num_samples):
    # #     return delta_sampler(num_samples, subs_x0)
    
    # props_x0 =\
    #     np.array([1.2682510e+00, 5.0245000e-02, 1.2820755e+03, 3.0140500e-02]).reshape((4,1))
    # # def det_props_sampler(num_samples):
    # #     return delta_sampler(num_samples, props_x0)
    
    # beams_x0 = np.array([5.49932e5])
    # # def det_beams_sampler(num_samples):
    # #     return delta_sampler(num_samples, beams_x0)
    
    # geoms_x0 =\
    #     np.array([1.5e-5, 3.0e-6, 4.9714e-04/2, 0.26782, 3.0181e-04, 1.3e-06/2]).reshape((6,1,1))
    # # def det_geoms_sampler(num_samples):
    # #     long_samples = delta_sampler(num_samples*num_emitters, geoms_x0)
    # #     samples = long_samples.reshape((6,num_emitters,num_samples))
    # #     return samples
    
    # es_models_x0 = np.array([4.614e4]).reshape((1,1,1))
    # #computed from the geometries above
    
    # def det_loglike(sample):
    #     return ap_pm_loglike(sample, voltage, obs_current/num_emitters, noise_var/num_emitters, subs_x0, props_x0,\
    #               beams_x0, geoms_x0, es_models_x0)
    # def det_logpdf(sample):
    #     check = logprior(sample)
    #     if check <= -np.inf:
    #         return check
    #     else:
    #         return det_loglike(sample) + logprior(sample)
        
    # np.random.seed(1967) #set seed for reproducibility
    
    # #doing a test to try sort of manually fitting
    # #draw from the perfect geometry
    # test_point = np.array([1.17815344e+00, 7.34716087e-03, 7.00033136e-06])
    # #model parameters to try
    # test_voltage = np.linspace(800,2000,100)
    # test_current = current_model(test_point,test_voltage,subs_x0,props_x0,beams_x0,geoms_x0,es_models_x0)
    # #evaluate the model for those parameters
    # plt.figure()
    # plt.plot(test_voltage,test_current,'r-')
    # plt.plot(voltage,obs_current/num_emitters,'bo')
    
    # num_samples = 100000
    # DR_gamma = 1e-2 #covariance factor for delayed rejection
    # xi = 1e-20#[1e-7, 1e-3, 1e-8]#, 1e-4]
    # #xi = None
    # nug = np.array([1e-6,1e-2,1e-14])
    # prop_start = np.array([1.17815344e+00, 7.34716087e-03, 7.00033136e-06])
    #     # np.array([1.92059249e-04, 1.82133039e+01, 8.61573998e-05,\
    #     #           1.50573554e+00])
    # #proposed starting sample for the sampler
    # #will be changed by initialization scheme
    # samples_raw, acceptance_ratio = mhmcmc_DRAM(\
    #   prop_start, num_samples, det_logpdf, get_proposal,\
    #       use_AM=False, AM_burnin = 1000, use_DR=True, get_DR=get_DR_prop,\
    #           report_interval = 1000)
    # ws.Beep(262,1500)
    
    # dir_label = '..\\Data\\'
    
    # flabel = 'perfectArray_samples'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
    
    # fname = dir_label + flabel + '__' + dt_str    
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((samples_raw,subs_x0,props_x0,beams_x0,geoms_x0),fid)
    # with open(fname + '.txt','w') as fid:
    #     print(r'$\zeta_1$',r'$\zeta_2$',r'$b_0$',file=fid,sep='\t')
    # write2DArray(samples_raw, fname + '.txt')
    
        
    # #posterior predictions
    # #res = 1000
    # V_pre = voltage.copy()
    # burn = 10000
    # #ditch the first 100000 samples to burn-in
    # sub_sample = 10 #only sample every tenth sample
    # #should yield about 9000 subsamples
    # num_sub_samples = np.floor((num_samples - burn) / sub_sample).astype(int)
    # I_pre = np.zeros((num_sub_samples,voltage.size))#res))
    # for ii in range(num_sub_samples):
    #     I_noise = noise_var * np.random.randn(voltage.size)
    #     I_pre[ii] = \
    #         num_emitters * current_model(samples_raw[burn + ii * sub_sample],V_pre,\
    #           subs_x0,props_x0,beams_x0,geoms_x0,es_models_x0).flatten() + I_noise
    #     #make a prediction and corrupt it by noise
    #     print('Percent Complete:',ii/num_sub_samples)
    #     #print out progress
    
    # flabel = 'perfectArray_predicts'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
    # fname = dir_label + flabel + '__' + dt_str    
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((V_pre,I_pre),fid)
    # write2DArray(V_pre.reshape(1,voltage.size), fname + '.txt')
    # write2DArray(I_pre, fname + '.txt')
    
    
    
#%% Uncertain Array deterministic sampling
    #needs a new likelihood
    # np.random.seed(1967) #set seed for reproducibility
    # #easier than printing out the geometry that's sampled
    # subs = subs_sampler(1)
    # props = props_sampler(1)
    # beams = beams_sampler(1)
    # geoms = geoms_sampler(1)
    # def single_real_like(params):
    #     model_current = current_model(params,voltage,subs,props,beams,geoms)
    #     log_like = -1/2*(model_current.flatten() - obs_current)**2 / noise_var
    #     #compute the log likelihood of each realization of each datum
    #     #assuming independent noise and Gaussian likelihood
    #     return np.sum(log_like)
    # def sr_logpdf(sample):
    #     #the logpdf must be the sum of the logprior over parameters and the
    #     #log likelihood
    #     check = logprior(sample)
    #     #calculate the log prior
    #     if check <= -np.inf:
    #         #if out of bounds of prior
    #         return check #don't actually need to evaluate the likelihood
    #     else:
    #         return single_real_like(sample) + check #otherwise, must and return sum
        
    # DR_gamma = 1e-2 #covariance factor for delayed rejection
    # xi = 1e-20#[1e-7, 1e-3, 1e-8]#, 1e-4]
    # #xi = None
    # nug = np.array([1e-8,1e-8,1e-16])#,1e2])
    # prop_start = np.array([7.08e-2, 2.85e-2, .98e-6])
    # def sr_initializer(x_k, target_logpdf):
    #     return init_GRW_opt(x_k,target_logpdf,nug=nug)
    
   
    # def sr_get_proposal(k, x_k, burn_in, previous_proposal_sampler,\
    #                  previous_proposal_logpdf, *args):
    #     return get_adaptive_GRW(k, x_k, burn_in, previous_proposal_sampler,\
    #                             previous_proposal_logpdf, *args,\
    #                                 initializer=sr_initializer, xi=xi)
    # #define a get proposal function
    # #this case, using a Gaussian random walk
    
    # def sr_get_DR_prop(*args):
    #     return get_DR_GRW(*args,gamma=DR_gamma)
    # #define get proposal func for DR
    
    # dir_label = '..\\Data\\'
    
    # num_samples = 100000
    # samples_raw, acceptance_ratio = mhmcmc_DRAM(\
    #   prop_start, num_samples, sr_logpdf, sr_get_proposal,\
    #       use_AM=True, AM_burnin = 1000, use_DR=True, get_DR=sr_get_DR_prop,\
    #           report_interval = 1000)
    # ws.Beep(262,1500)
    
    # #write out samples to file
    
    # flabel = 'sr_deterministic_samples'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
    
    # fname = dir_label + flabel + '__' + dt_str    
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((samples_raw,subs,props,beams,geoms),fid)
    # with open(fname + '.txt','w') as fid:
    #     print(r'$\zeta_1$',r'$\zeta_2$',r'$b_0$',file=fid,sep='\t')
    # write2DArray(samples_raw, fname + '.txt')
    
        
    #posterior predictions
    # res = 1000
    # V_pre = np.linspace(800,1850,res)
    # burn = 10000
    # #ditch the first 100000 samples to burn-in
    # sub_sample = 10 #only sample every tenth sample
    # #should yield about 9000 subsamples
    # num_sub_samples = np.floor((num_samples - burn) / sub_sample).astype(int)
    # I_pre = np.zeros((num_sub_samples,res))
    # for ii in range(num_sub_samples):
    #     I_pre[ii] = \
    #         current_model(samples_raw[burn + ii * sub_sample],V_pre,\
    #           subs,props,beams, geoms).flatten()
    #     print('Percent Complete:',ii/num_sub_samples)
        
    # flabel = 'sr_deterministic_predicts'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
        
    
    # fname = dir_label + flabel + '__' + dt_str   
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((V_pre,I_pre),fid)
    # write2DArray(V_pre.reshape(1,res), fname + '.txt')
    # write2DArray(I_pre, fname + '.txt')
    
    # test_point = np.array([7.28856590e-02, 2.57199079e-02, 9.97411524e-07])
    # #model parameters to try
    # test_voltage = np.linspace(800,1850,1000)
    # test_current = current_model(test_point,test_voltage,subs,props,beams,geoms)
    # #evaluate the model for those parameters
    # plt.figure()
    # plt.plot(test_voltage,test_current,'r-')
    # plt.plot(voltage,obs_current,'bo')
    
#%% Inferring from Perez-Martinez Data and Predicting from there

    # with open('..\\Data\\PerezM_training_data.pkl','rb') as fid:
    #     voltage, obs_current, noise_var = pkl.load(fid)
        
    # subs_x0 = np.array([np.inf, 3e-14])
    # def det_subs_sampler(num_samples):
    #     return delta_sampler(num_samples, subs_x0)
    
    # props_x0 = props.flatten()
    # def det_props_sampler(num_samples):
    #     return delta_sampler(num_samples, props_x0)
    
    # beams_x0 = np.array([6.21e5])
    # def det_beams_sampler(num_samples):
    #     return delta_sampler(num_samples, beams_x0)
    
    # geoms_x0 =\
    #     np.array([1.5e-6, 3.0e-6, 1.6e-3, 0.26782, 3.0181e-04, 1.3e-06/2])
    # def det_geoms_sampler(num_samples):
    #     long_samples = delta_sampler(num_samples, geoms_x0)
    #     samples = long_samples.reshape((6,1,num_samples))
    #     return samples
    
    # def det_loglike(sample):
    #     return pm_loglike(sample, voltage, obs_current, noise_var, \
    #                       det_subs_sampler, det_props_sampler,\
    #                           det_beams_sampler, det_geoms_sampler,\
    #                               block_samples = 1, num_blocks = 1)
    # def det_logpdf(sample):
    #     check = logprior(sample)
    #     if check <= -np.inf:
    #         return check
    #     else:
    #         return det_loglike(sample) + logprior(sample)
        
    
    # num_samples = 100000
    # samples_raw, acceptance_ratio = mhmcmc_DRAM(\
    #   prop_start, num_samples, det_logpdf, get_proposal,\
    #       use_AM=False, AM_burnin = 1000, use_DR=True, get_DR=get_DR_prop,\
    #           report_interval = 1000)
    # ws.Beep(262,1500)
    
    # dir_label = '..\\Data\\'
    
    # flabel = 'PerM_deterministic_samples'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
    
    # fname = dir_label + flabel + '__' + dt_str    
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((samples_raw,subs_x0,props_x0,beams_x0,geoms_x0),fid)
    # with open(fname + '.txt','w') as fid:
    #     print(r'$\zeta_1$',r'$\zeta_2$',r'$b_0$',file=fid,sep='\t')
    # write2DArray(samples_raw, fname + '.txt')
    
        
    # #posterior predictions
    # res = 1000
    # V_pre = np.linspace(800,1850,res)
    # burn = 10000
    # #ditch the first 100000 samples to burn-in
    # sub_sample = 10 #only sample every tenth sample
    # #should yield about 9000 subsamples
    # num_sub_samples = np.floor((num_samples - burn) / sub_sample).astype(int)
    # I_pre = np.zeros((num_sub_samples,res))
    # for ii in range(num_sub_samples):
    #     I_pre[ii] = \
    #         current_model(samples_raw[burn + ii * sub_sample],V_pre,\
    #           subs,props,beams,geoms).flatten()
    
    # flabel = 'deterministic_predicts'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
    # fname = dir_label + flabel + '__' + dt_str    
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((V_pre,I_pre),fid)
    # write2DArray(V_pre.reshape(1,res), fname + '.txt')
    # write2DArray(I_pre, fname + '.txt')
    
    
    # #doing a test to try sort of manually fitting
    # subs = det_subs_sampler(1)
    # props = det_props_sampler(1)
    # beams = det_beams_sampler(1)
    # geoms = det_geoms_sampler(1)
    # #draw from the perfect geometry
    # test_point = np.array([1.09e1, 1, 1e-4])
    # #model parameters to try
    # test_voltage = np.linspace(1580,2020,1000)
    # test_current = current_model(test_point,test_voltage,subs,props,beams,geoms)
    # #evaluate the model for those parameters
    # plt.figure()
    # plt.plot(test_voltage,test_current,'r-')
    # plt.plot(voltage,obs_current,'bo')
    
        
#%% Likelihood expense test   
    # t0 = t.time()
    # a = loglike(prop_start)
    # at = t.time() - t0
    # ws.Beep(262,250)
    # Nt = 10
    # b = np.zeros(Nt)
    # bt = np.zeros(Nt)
    # for i in range(Nt):
    #     t0 = t.time()
    #     b[i] = loglike(prop_start)
    #     bt[i] = t.time() - t0
    #     #ws.Beep(262,250)
    # ws.Beep(262,1500)
    
#%% Uncertain Array Approximately Pseudomarginal approach
    # # the strategy here is to perform an approximately pseudomarginal approach
    # # by using the same samples from the nuisance parameters for each
    # # evaluation of the Monte Carlo estimator for the pseudomarginal
    # # likelihood
    # # it will introduce some bias, but may actually still provide some
    # # assessment of uncertainty
    # #
    # # the goal is to use the existing framework, but since it samples every
    # # time it needs to evaluate it this is not so good
    # # however, if we implement a deterministic sampler, the problem is solved
    # # 
    # # in the limit where the number of realizations is 1, this is equivalent
    # # to the single realization deterministic problem we had done earlier
    
    # np.random.seed(1967) #set seed for reproducibility
    
    # Nr = 100;
    # #number of realizations to bake into the approximately pseudomarginal
    # #likelihood
    
    # subs = subs_sampler(Nr)
    # props = props_sampler(Nr)
    # beams = beams_sampler(Nr)
    # geoms = geoms_sampler(Nr)
    
    # #write2DArray(geoms.reshape((6,num_emitters*Nr)).transpose(), '..\mr_geoms.dat')
    # es_models = np.loadtxt('..\mr_geoms_tipE.dat',dtype=float,delimiter='\t')
    # es_models = es_models.reshape((1,num_emitters,Nr))
    
    
    # def mr_loglike(sample):
    #     #define a log likelihood over the sample
    #     #this needs to work with only 'sample' as an input
    #     return ap_pm_loglike(sample, voltage, obs_current, noise_var,\
    #                       subs, props,\
    #                           beams, geoms,\
    #                               es_models)
    # def mr_logpdf(sample):
    #     #the logpdf must be the sum of the logprior over parameters and the
    #     #log likelihood
    #     check = logprior(sample)
    #     #calculate the log prior
    #     if check <= -np.inf:
    #         #if out of bounds of prior
    #         return check #don't actually need to evaluate the likelihood
    #     else:
    #         return mr_loglike(sample) + check #otherwise, must and return sum
        
        
    # #doing a test to try sort of manually fitting
    # test_point = np.array([2.718408696e+00, 2.19609878e-02, 1.82697901e-05])
    # #model parameters to try
    # test_subs = subs[:,2,None]
    # test_props = props[:,2,None]
    # test_beams = beams[:,2,None]
    # test_geoms = geoms[:,:,2,None]
    # test_es_models = es_models[:,:,2,None]
    # test_voltage = np.linspace(800,1900,100)
    # test_current = current_model(test_point,test_voltage,test_subs,test_props,test_beams,test_geoms,test_es_models)
    # #evaluate the model for those parameters
    # plt.figure()
    # plt.plot(test_voltage,test_current,'r-')
    # plt.plot(voltage,obs_current,'bo')
    
        
    # DR_gamma = 1e-2 #covariance factor for delayed rejection
    # xi = 1e-20#[1e-7, 1e-3, 1e-8]#, 1e-4]
    # #xi = None
    # nug = np.array([1e-9,1e-8,1e-16])#,1e2])
    # prop_start = np.array([2.71176449e+00, 1.93804068e-02, 1.92846788e-05])
    # def mr_initializer(x_k, target_logpdf):
    #     return init_GRW_opt(x_k,target_logpdf,nug=nug)
    
   
    # def mr_get_proposal(k, x_k, burn_in, previous_proposal_sampler,\
    #                   previous_proposal_logpdf, *args):
    #     return get_adaptive_GRW(k, x_k, burn_in, previous_proposal_sampler,\
    #                             previous_proposal_logpdf, *args,\
    #                                 initializer=mr_initializer, xi=xi)
    # #define a get proposal function
    # #this case, using a Gaussian random walk
    
    # def mr_get_DR_prop(*args):
    #     return get_DR_GRW(*args,gamma=DR_gamma)
    # #define get proposal func for DR
    
    # dir_label = '..\\Data\\'
    
    # num_samples = 100000
    # samples_raw, acceptance_ratio = mhmcmc_DRAM(\
    #   prop_start, num_samples, mr_logpdf, mr_get_proposal,\
    #       use_AM=True, AM_burnin = 1000, use_DR=True, get_DR=mr_get_DR_prop,\
    #           report_interval = 1000)
    # ws.Beep(262,1500)
    
    # #write out samples to file
    
    # flabel = 'Nr1_samples'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
    
    # fname = dir_label + flabel + '__' + dt_str    
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((samples_raw,subs,props,beams,geoms,es_models),fid)
    # with open(fname + '.txt','w') as fid:
    #     print(r'$\zeta_1$',r'$\zeta_2$',r'$b_0$',file=fid,sep='\t')
    # write2DArray(samples_raw, fname + '.txt')
    
        
    # #posterior predictions
    # #predicting at the data themselves, since we don't have noise data
    # #elsewhere
    # # res = 1000
    # V_pre = voltage.copy()#np.linspace(800,1850,res)
    # burn = 10000
    # #ditch the first 100000 samples to burn-in
    # sub_sample = 10 #only sample every tenth sample
    # #should yield about 9000 subsamples
    # num_sub_samples = np.floor((num_samples - burn) / sub_sample).astype(int)
    # I_pre = np.zeros((num_sub_samples,voltage.size))#res))
    # #will predict num_sub_samples times for each value of voltage
    # for ii in range(num_sub_samples):
    #     #for each sample we want to draw from the posterior predictive
    #     randReal = np.random.randint(0,Nr)
    #     #select one of the realizations at random
    #     I_noise = np.sqrt(noise_var) * np.random.randn(voltage.size)
    #     #sample from the experimental noise at each point
    #     I_pre[ii] = \
    #         current_model(samples_raw[burn + ii * sub_sample],V_pre,\
    #           subs[...,randReal,None],props[...,randReal,None],\
    #               beams[...,randReal,None], geoms[...,randReal,None],\
    #                   es_models[...,randReal,None]).flatten()\
    #             + I_noise
    #     #make a prediction and corrupt it by noise
    #     print('Percent Complete:',ii/num_sub_samples)
    #     #print out progress
        
    # flabel = 'Nr1_predicts'
    # dt_str = dt.datetime.today().isoformat('T','seconds')\
    #     .replace('-','_').replace(':','_')
        
    
    # fname = dir_label + flabel + '__' + dt_str   
    # with open(fname + '.pkl','wb') as fid:
    #     pkl.dump((V_pre,I_pre),fid)
    # write2DArray(V_pre.reshape(1,voltage.size), fname + '.txt')
    # write2DArray(I_pre, fname + '.txt')
    
#%% Uncertain Array Approximately Pseudomarginal approach with certain reservoir pressure
    
    np.random.seed(1967) #set seed for reproducibility
    
    Nr = 100;
    #number of realizations to bake into the approximately pseudomarginal
    #likelihood
    
    subs = subs_sampler(Nr)
    props = props_sampler(Nr)
    beams = beams_sampler(Nr)
    geoms = geoms_sampler(Nr)
    
    #write2DArray(geoms.reshape((6,num_emitters*Nr)).transpose(), '..\mr_geoms.dat')
    es_models = np.loadtxt('..\mr_geoms_tipE.dat',dtype=float,delimiter='\t')
    es_models = es_models.reshape((1,num_emitters,Nr))
    
    subs[0,:] = 8e-6
    #consider case where reservoir bulk pore radius is not uncertain, but is known exactly
    #take, for sake of argument, it to be 16 micron pore diameter
    
    
    def nrsv_mr_loglike(sample):
        #define a log likelihood over the sample
        #this needs to work with only 'sample' as an input
        return ap_pm_loglike(sample, voltage, obs_current, noise_var,\
                          subs, props,\
                              beams, geoms,\
                                  es_models)
    def nrsv_mr_logpdf(sample):
        #the logpdf must be the sum of the logprior over parameters and the
        #log likelihood
        check = logprior(sample)
        #calculate the log prior
        if check <= -np.inf:
            #if out of bounds of prior
            return check #don't actually need to evaluate the likelihood
        else:
            return nrsv_mr_loglike(sample) + check #otherwise, must and return sum
        
        
    #doing a test to try sort of manually fitting
    test_point = np.array([2.56727718e+00, 1.50167874e-02, 1.99673250e-05])
    #model parameters to try
    test_subs = subs[:,2,None]
    test_props = props[:,2,None]
    test_beams = beams[:,2,None]
    test_geoms = geoms[:,:,2,None]
    test_es_models = es_models[:,:,2,None]
    test_voltage = np.linspace(800,1900,100)
    test_current = current_model(test_point,test_voltage,test_subs,test_props,test_beams,test_geoms,test_es_models)
    #evaluate the model for those parameters
    plt.figure()
    plt.plot(test_voltage,test_current,'r-')
    plt.plot(voltage,obs_current,'bo')
    
        
    DR_gamma = 1e-2 #covariance factor for delayed rejection
    xi = 1e-20#[1e-7, 1e-3, 1e-8]#, 1e-4]
    #xi = None
    nug = np.array([1e-9,5e-8,1e-16])#,1e2])
    prop_start = np.array([2.56727718e+00, 1.50167874e-02, 1.99673250e-05])
    def nrsv_mr_initializer(x_k, target_logpdf):
        return init_GRW_opt(x_k,target_logpdf,nug=nug)
    
   
    def nrsv_mr_get_proposal(k, x_k, burn_in, previous_proposal_sampler,\
                      previous_proposal_logpdf, *args):
        return get_adaptive_GRW(k, x_k, burn_in, previous_proposal_sampler,\
                                previous_proposal_logpdf, *args,\
                                    initializer=nrsv_mr_initializer, xi=xi)
    #define a get proposal function
    #this case, using a Gaussian random walk
    
    def nrsv_mr_get_DR_prop(*args):
        return get_DR_GRW(*args,gamma=DR_gamma)
    #define get proposal func for DR
    
    dir_label = '..\\Data\\'
    
    num_samples = 100000
    samples_raw, acceptance_ratio = mhmcmc_DRAM(\
      prop_start, num_samples, nrsv_mr_logpdf, nrsv_mr_get_proposal,\
          use_AM=True, AM_burnin = 1000, use_DR=True, get_DR=nrsv_mr_get_DR_prop,\
              report_interval = 1000)
    ws.Beep(262,1500)
    
    #write out samples to file
    
    flabel = 'Nr100_noPr_samples'
    dt_str = dt.datetime.today().isoformat('T','seconds')\
        .replace('-','_').replace(':','_')
    
    fname = dir_label + flabel + '__' + dt_str    
    with open(fname + '.pkl','wb') as fid:
        pkl.dump((samples_raw,subs,props,beams,geoms,es_models),fid)
    with open(fname + '.txt','w') as fid:
        print(r'$\zeta_1$',r'$\zeta_2$',r'$b_0$',file=fid,sep='\t')
    write2DArray(samples_raw, fname + '.txt')
    
        
    #posterior predictions
    #predicting at the data themselves, since we don't have noise data
    #elsewhere
    # res = 1000
    V_pre = voltage.copy()#np.linspace(800,1850,res)
    burn = 10000
    #ditch the first 100000 samples to burn-in
    sub_sample = 10 #only sample every tenth sample
    #should yield about 9000 subsamples
    num_sub_samples = np.floor((num_samples - burn) / sub_sample).astype(int)
    I_pre = np.zeros((num_sub_samples,voltage.size))#res))
    #will predict num_sub_samples times for each value of voltage
    for ii in range(num_sub_samples):
        #for each sample we want to draw from the posterior predictive
        randReal = np.random.randint(0,Nr)
        #select one of the realizations at random
        I_noise = np.sqrt(noise_var) * np.random.randn(voltage.size)
        #sample from the experimental noise at each point
        I_pre[ii] = \
            current_model(samples_raw[burn + ii * sub_sample],V_pre,\
              subs[...,randReal,None],props[...,randReal,None],\
                  beams[...,randReal,None], geoms[...,randReal,None],\
                      es_models[...,randReal,None]).flatten()\
                + I_noise
        #make a prediction and corrupt it by noise
        print('Percent Complete:',ii/num_sub_samples)
        #print out progress
        
    flabel = 'Nr100_noPr_predicts'
    dt_str = dt.datetime.today().isoformat('T','seconds')\
        .replace('-','_').replace(':','_')
        
    
    fname = dir_label + flabel + '__' + dt_str   
    with open(fname + '.pkl','wb') as fid:
        pkl.dump((V_pre,I_pre),fid)
    write2DArray(V_pre.reshape(1,voltage.size), fname + '.txt')
    write2DArray(I_pre, fname + '.txt')
#%% Predicting from Nr=100 case for negative polarity
    with open('training_data2.pkl','rb') as fid:
        voltage, obs_current, noise_var = pkl.load(fid)
    #import training data from .pkl file
    #note that this is only used to sample experimental noise consistent with
    #'noise_var'
    with open('..\\Data\\Nr100_noPr_samples__2021_12_07T11_41_27.pkl','rb') as fid:
        samples_raw,subs,props,beams,geoms,es_models = pkl.load(fid)
    #load previous results for 100 realization
        
    mu_beams = np.array([6.2787e5])
    sigma_beams = np.array([1.4371e4])
    def beams_sampler(num_samples):
        #define "prior" distribution over beam properties
        return gauss_sampler_noCov(num_samples, mu_beams, sigma_beams)
    #redefine a new sampler over beam properties for the negative polarity mode
    
    np.random.seed(1967) #set seed for reproducibility
    
    Nr = 100;
    #number of realizations to bake into the approximately pseudomarginal
    #likelihood
    beams = beams_sampler(Nr)
    #sample new values for G to reflect this new case
    
    V_pre = voltage.copy()#np.linspace(800,1850,res)
    burn = 10000
    #ditch the first 100000 samples to burn-in
    sub_sample = 10 #only sample every tenth sample
    #should yield about 9000 subsamples
    num_samples = samples_raw.shape[0]
    num_sub_samples = np.floor((num_samples - burn) / sub_sample).astype(int)
    I_pre = np.zeros((num_sub_samples,voltage.size))#res))
    #will predict num_sub_samples times for each value of voltage
    for ii in range(num_sub_samples):
        #for each sample we want to draw from the posterior predictive
        randReal = np.random.randint(0,Nr)
        #select one of the realizations at random
        I_noise = np.sqrt(noise_var) * np.random.randn(voltage.size)
        #sample from the experimental noise at each point
        I_pre[ii] = \
            current_model(samples_raw[burn + ii * sub_sample],V_pre,\
              subs[...,randReal,None],props[...,randReal,None],\
                  beams[...,randReal,None], geoms[...,randReal,None],\
                      es_models[...,randReal,None]).flatten()\
                + I_noise
        #make a prediction and corrupt it by noise
        print('Percent Complete:',ii/num_sub_samples)
        #print out progress
    
    dir_label = '..\\Data\\'
    flabel = 'Nr100_noPr_negPolPredicts'
    dt_str = dt.datetime.today().isoformat('T','seconds')\
        .replace('-','_').replace(':','_')
        
    
    fname = dir_label + flabel + '__' + dt_str   
    with open(fname + '.pkl','wb') as fid:
        pkl.dump((V_pre,I_pre),fid)
    write2DArray(V_pre.reshape(1,voltage.size), fname + '.txt')
    write2DArray(I_pre, fname + '.txt')