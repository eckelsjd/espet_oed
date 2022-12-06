import numpy as np
# import cupy as cp
import sys
sys.path.extend('..')

from src.utils import fix_input_shape


# Test estimator on Linear-Gaussian model
def linear_gaussian_model(x, theta, eta=None):
    """
    Linear Gaussian model with analytical solution for eig:
    [y1, y2] = [d, 0; 0, 1-d] * [theta, eta]
    Parameters
    ----------
    x: (Nx, x_dim) input locations, or operating conditions
    theta: (*, theta_dim) model parameters
    eta: (*, eta_dim) nuisance parameters

    Nx: Number of input locations
    x_dim: Dimension of operating conditions
    y_dim: Dimension of output
    theta_dim: Dimension of model parameters
    eta_dim: Dimension of nuisance parameters

    Returns
    -------
    g_theta: (*, Nx, y_dim) model output
    """
    x = fix_input_shape(x)
    Nx, x_dim = x.shape
    theta = np.atleast_1d(theta)
    if len(theta.shape) == 1:
        theta = np.expand_dims(theta, axis=0)

    if eta is None:
        # Eta is passed in as second theta dimension (for joint EIG)
        eta = np.expand_dims(theta[..., 1], axis=-1)
        theta = np.expand_dims(theta[..., 0], axis=-1)
    else:
        eta = np.atleast_1d(eta)
        if len(eta.shape) == 1:
            eta = np.expand_dims(eta, axis=0)
    theta_dim = theta.shape[-1]
    shape = eta.shape[:-1]
    eta_dim = eta.shape[-1]
    y_dim = 2

    x = x.reshape((1,) * (len(shape)-1) + (Nx, x_dim))  # (...1, Nx, x_dim)
    model_eval = np.zeros((*eta.shape[:-2], Nx, y_dim), dtype=np.float32)

    # 1 model param, 1 nuisance param and 1 input location
    x = np.squeeze(x, axis=-1)              # (...1, Nx)
    theta = np.squeeze(theta, axis=-1)      # (*, Nx)
    eta = np.squeeze(eta, axis=-1)          # (*, Nx)

    y1 = x * theta                          # (*, Nx)
    y2 = (1 - x) * eta                      # (*, Nx)

    y = np.concatenate((np.expand_dims(y1, axis=-1), np.expand_dims(y2, axis=-1)), axis=-1)  # (*, Nx, 2)
    return y


# Analytical solution for linear gaussian model
def linear_gaussian_eig(d, var):
    return 0.5 * np.log(1 + d ** 2 / var)


# Simple nonlinear model example
def nonlinear_model(x, theta, eta=None):
    """Compute nonlinear model paper from Marzouk (2011)
    Parameters
    ----------
    x: (Nx, x_dim) Input locations
    theta: (*, Nx, theta_dim) Model parameters
    eta: (*, Nx, eta_dim) No nuisance parameters for this model (UNUSED)

    Returns
    -------
    y: (*, Nx, y_dim) The model output, where y_dim = x_dim
    """
    # Fix input shapes
    theta = np.atleast_1d(theta)
    if len(theta.shape) == 1:
        theta = np.expand_dims(theta, axis=0)
    x = fix_input_shape(x)
    shape = theta.shape[:-1]
    theta_dim = theta.shape[-1]
    Nx, x_dim = x.shape
    y_dim = x_dim  # one output per x_dim
    assert theta_dim == 1
    assert (theta.shape[-2] == Nx or theta.shape[-2] == 1)
    theta = np.squeeze(theta, axis=-1)                # (*, Nx)
    x = x.reshape((1,)*(len(shape)-1) + (Nx, x_dim))  # (...1, Nx, x_dim)

    if len(shape) == 1:
        model_eval = np.zeros((Nx, y_dim), dtype=np.float32)
    elif len(shape) > 1:
        model_eval = np.zeros((*(shape[:-1]), Nx, y_dim), dtype=np.float32)

    # Compute model along each x_dim
    for i in range(x_dim):
        # Select all x for each x_dim
        ind = tuple([slice(None)]*(len(shape)) + [i])  # slice(None) == : indexing
        x_i = x[ind]  # (...1, Nx)

        # Evaluate model for the x_i dimension
        model_eval[ind] = np.square(x_i) * np.power(theta, 3) + np.exp(-abs(0.2 - x_i)) * theta  # (*, Nx)

    return model_eval  # (*, Nx, y_dim)


def custom_nonlinear(x, theta, eta=None, env_var=0.1**2, wavelength=0.5, wave_amp=0.1, tanh_amp=0.5, tanh_shift=4):
    """Custom nonlinear model
    Parameters
    ----------
    x: (Nx, x_dim) Input locations
    theta: (*, theta_dim) Model parameters
    eta: (*, eta_dim) Nuisance parameters
    env_var: Variance of Gaussian envelope
    wavelength: Sinusoidal perturbation wavelength
    wave_amp: Amplitude of perturbation
    tanh_amp: Amplitude of tanh(x)
    tanh_shift: Shift factor of tanh(2*shift*x - shift)

    Returns
    -------
    y: (*, Nx, y_dim) The model output, where y_dim = x_dim
    """
    x = fix_input_shape(x)
    Nx, x_dim = x.shape
    theta = np.atleast_1d(theta)
    if len(theta.shape) == 1:
        theta = np.expand_dims(theta, axis=0)

    if eta is None:
        # Eta is passed in as second theta dimension (for joint EIG)
        eta = np.expand_dims(theta[..., 1], axis=-1)
        theta = np.expand_dims(theta[..., 0], axis=-1)
    else:
        eta = np.atleast_1d(eta)
        if len(eta.shape) == 1:
            eta = np.expand_dims(eta, axis=0)
    theta_dim = theta.shape[-1]
    shape = eta.shape[:-1]
    eta_dim = eta.shape[-1]
    y_dim = x_dim
    assert y_dim == x_dim == 1

    x = x.reshape((1,) * (len(shape)-1) + (Nx, x_dim))  # (...1, Nx, x_dim)
    model_eval = np.zeros((*eta.shape[:-2], Nx, y_dim), dtype=np.float32)

    # 1 model param, 1 nuisance param and 1 input location
    x = np.squeeze(x, axis=-1)  # (...1, Nx)
    theta = np.squeeze(theta, axis=-1)  # (*, Nx)
    eta = np.squeeze(eta, axis=-1)  # (*, Nx)

    # Traveling sinusoid with moving Gaussian envelope
    env_range = [0.2, 0.6]
    mu = env_range[0] + theta * (env_range[1] - env_range[0])
    theta_env = 1 / (np.sqrt(2 * np.pi * env_var)) * np.exp(-0.5 * (x - mu) ** 2 / env_var)
    ftheta = wave_amp * np.sin((2*np.pi/wavelength) * theta) * theta_env

    # Underlying tanh dependence on x
    fd = tanh_amp * np.tanh(2*tanh_shift*x - tanh_shift) + tanh_amp

    # Perturbations at x=1 from nuisance variable
    eta_env = 1 / (np.sqrt(2 * np.pi * env_var)) * np.exp(-0.5 * (x - 1) ** 2 / env_var)
    feta = wave_amp * np.sin((2*np.pi/(0.25*wavelength)) * eta) * eta_env

    # Compute model = f(theta, d) + f(d)
    y = np.expand_dims(ftheta + fd + feta, axis=-1)
    return y  # (*, Nx, y_dim)


def electrospray_current_model(x, theta, eta, gpu=False):
    """ Predicts total array current for an electrospray thruster
    Parameters
    ----------
    x: (Nx, x_dim) voltage operating conditions [V]
    theta: (*, Nx, theta_dim) model parameters
    eta: (*, Nx, eta_dim) nuisance parameters

    Returns
    -------
    current: (*, Nx, y_dim) model output [A]
    """
    lp = cp if gpu else np
    # Fix shapes
    x = lp.atleast_1d(x).astype(lp.float32)
    if len(x.shape) == 1:
        # Assume one x dimension
        x = x[:, lp.newaxis]
    Nx, x_dim = x.shape
    theta = lp.atleast_1d(theta).astype(lp.float32)
    if len(theta.shape) == 1:
        theta = lp.expand_dims(theta, axis=0)
    eta = lp.atleast_1d(eta).astype(lp.float32)
    if len(eta.shape) == 1:
        eta = lp.expand_dims(eta, axis=0)
    theta_dim = theta.shape[-1]
    eta_shape = eta.shape[:-1]
    eta_dim = eta.shape[-1]
    y_dim = 1
    voltage = x.reshape((1,) * (len(eta_shape)-1) + (Nx, x_dim))  # (...1, Nx, x_dim)

    # Problem setup
    VACUUM_PERMITTIVITY = 8.8542e-12
    Ne = int(lp.floor((eta_dim - 7) / 7))    # number of emitters
    assert theta_dim == 3

    # Extract model parameters
    ion_emis_offset = theta[..., 0, lp.newaxis]  # (*, 1)
    ion_emis_slope = theta[..., 1, lp.newaxis]
    pool_radius = theta[..., 2, lp.newaxis]

    # Extract substrate properties
    res_pore_radius = eta[..., 0, lp.newaxis]
    permeability = eta[..., 1, lp.newaxis]

    # Extract material properties
    conductivity = eta[..., 2, lp.newaxis]
    surface_tension = eta[..., 3, lp.newaxis]
    density = eta[..., 4, lp.newaxis]
    viscosity = eta[..., 5, lp.newaxis]

    # Extract beam properties
    charge_to_mass = eta[..., 6, lp.newaxis]

    # Extract emitter geometries
    geo_data = eta[..., 7:].reshape((*eta_shape, Ne, 7))
    curvature_radius = geo_data[..., 0]  # (..., Nx, Ne)
    gap_distance = geo_data[..., 1]
    aperture_radius = geo_data[..., 2]
    half_angle = geo_data[..., 3]
    emitter_height = geo_data[..., 4]
    loc_pore_radius = geo_data[..., 5]
    emax_sim = geo_data[..., 6]
    # del geo_data

    # Calculate some quantities common to all emitters and emission sites   # (Nx, 1)
    hydraulic_resistivity = viscosity / (2 * lp.pi * permeability)
    res_pressure = -2 * surface_tension / res_pore_radius

    # Compute electric field
    applied_field = emax_sim * voltage                                      # (..., Nx, Ne)

    # Compute the number of sites for each emitter                          # (..., Nx, Ne)
    applied_electric_pressure = 1 / 2 * VACUUM_PERMITTIVITY * applied_field ** 2
    min_cap_pressure = 2 * surface_tension / (loc_pore_radius + pool_radius)
    max_cap_pressure = 2 * surface_tension / loc_pore_radius

    no_sites_mask = applied_electric_pressure < min_cap_pressure - res_pressure
    all_sites_mask = applied_electric_pressure > max_cap_pressure - res_pressure

    A_emission = 2 * lp.pi * curvature_radius ** 2 * (1 - lp.sin(half_angle))
    A_Taylor = lp.pi * ((pool_radius + loc_pore_radius) / 2) ** 2
    max_sites = A_emission / A_Taylor

    num_sites = lp.where(all_sites_mask, lp.floor(max_sites), 0)            # (..., Nx, Ne)
    neither_mask = ~(no_sites_mask | all_sites_mask)

    # Ignore errors from bad sites that we don't need
    with np.errstate(divide='ignore', invalid='ignore'):
        min_site_radius = 2 * surface_tension / (1 / 2 * VACUUM_PERMITTIVITY * applied_field ** 2 + res_pressure)
        arr = lp.floor(1 + (max_sites - 1) * (1 - (min_site_radius - loc_pore_radius) / pool_radius) ** 4)
        num_sites = num_sites + lp.where(neither_mask, arr, 0)

        Ns_max = int(lp.max(num_sites).astype(int))
        site_nums = lp.arange(Ns_max).reshape(((1,)*int(len(num_sites.shape)) + (Ns_max,))) + 1      # (...1, Ns_max)
        active_mask = site_nums <= num_sites[..., lp.newaxis]                                   # (..., Ns_max)

        # Compute site radius
        site_radius = pool_radius[..., lp.newaxis] * (1 - ((site_nums-1) / (max_sites[..., lp.newaxis]-1)) ** 0.25) + \
                      loc_pore_radius[..., lp.newaxis]

        # Compute characteristic pressure
        char_pressure = 2 * surface_tension[..., lp.newaxis] / site_radius

        # Compute characteristic electric field and onset field
        char_elec_field = lp.sqrt(2 * char_pressure / VACUUM_PERMITTIVITY)
        onset_field = lp.sqrt((2 * (char_pressure - res_pressure[..., lp.newaxis])) / VACUUM_PERMITTIVITY)
        dimless_applied_field = applied_field[..., lp.newaxis] / char_elec_field
        dimless_onset_field = onset_field / char_elec_field

        # Compute hydraulic impedance
        hydraulic_impedance = num_sites * hydraulic_resistivity / (1 - lp.cos(half_angle)) * \
                              (lp.tan(half_angle) / curvature_radius - lp.cos(half_angle) / emitter_height)
        dimless_res_pressure = res_pressure[..., lp.newaxis] / char_pressure
        dimless_hydraulic_impedance = conductivity[..., lp.newaxis] * char_elec_field * site_radius ** 2 * \
                                      hydraulic_impedance[..., lp.newaxis] / (char_pressure * density[..., lp.newaxis] *
                                                                     charge_to_mass[..., lp.newaxis])

        # Compute current emitted by each active site
        dimless_current = (ion_emis_offset[..., lp.newaxis] + ion_emis_slope[..., lp.newaxis] * (dimless_applied_field -
                            dimless_onset_field) + dimless_res_pressure) / dimless_hydraulic_impedance

        arr = dimless_current * conductivity[..., lp.newaxis] * char_elec_field * site_radius ** 2

    site_current = lp.where(active_mask, arr, 0)
    site_current[site_current < 0] = 0

    # Sum current over all sites and emitters
    current = lp.sum(site_current, axis=(-1, -2))[..., lp.newaxis]   # (..., 1)

    if gpu:
        current = lp.asnumpy(current)

    return current
