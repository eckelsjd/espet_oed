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
    theta: (*, Nx, theta_dim) model parameters
    eta: (*, Nx, eta_dim) nuisance parameters

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
    theta_shape = theta.shape[:-2]
    theta_dim = theta.shape[-1]
    eta_shape = eta.shape[:-2]
    eta_dim = eta.shape[-1]
    y_dim = 2

    x = x.reshape((1,) * len(eta_shape) + (Nx, x_dim))  # (...1, Nx, x_dim)
    model_eval = np.zeros((*eta_shape, Nx, y_dim), dtype=np.float32)

    # 1 model param, 1 nuisance param and 1 input location
    x = np.squeeze(x, axis=-1)              # (...1, Nx)
    theta = np.squeeze(theta, axis=-1)      # (*, Nx)
    eta = np.squeeze(eta, axis=-1)          # (*, Nx)

    y1 = x * theta                          # (*, Nx) or (1, Nx)
    if y1.shape[0] == 1:
        # If reusing samples of theta
        tile_axis = (eta_shape[0],) + (1,)*(len(eta_shape)-1) + (1,)  # (*, 1..., 1), only tile on first axis of eta
        y1 = np.tile(y1, tile_axis)         # (*, Nx)

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
    theta: (*, Nx, theta_dim) Model parameters
    eta: (*, Nx, eta_dim) Nuisance parameters
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
    theta_shape = theta.shape[:-2]
    theta_dim = theta.shape[-1]
    eta_shape = eta.shape[:-2]
    eta_dim = eta.shape[-1]
    y_dim = x_dim
    assert y_dim == x_dim == 1

    x = x.reshape((1,) * len(eta_shape) + (Nx, x_dim))  # (...1, Nx, x_dim)
    model_eval = np.zeros((*eta_shape, Nx, y_dim), dtype=np.float32)

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


def electrospray_current_model_gpu(x, theta, eta):
    """ MEMORY-HEAVY O(1) runtime, O(5GB/process) memory
    Predicts total array current for an electrospray thruster
    Parameters
    ----------
    x: (Nx, x_dim) voltage operating conditions [V]
    theta: (*, Nx, theta_dim) model parameters
    eta: (*, Nx, eta_dim) nuisance parameters

    Returns
    -------
    current: (*, Nx, y_dim) model output [A]
    """
    # Transfer to GPU memory
    voltage = cp.asarray(x)
    theta = cp.asarray(theta)
    eta = cp.asarray(eta)

    # Constants
    eta_dim = eta.shape[-1]
    theta_dim = theta.shape[-1]
    VACUUM_PERMITTIVITY = 8.8542e-12
    Nx, x_dim = x.shape
    Ne = int(np.floor((eta_dim - 7) / 7))   # number of emitters

    # Extract model params
    ind = tuple([slice(None)] * (len(theta.shape) - 1))
    ion_emis_offset = theta[(*ind, 0)]                          # (*, Nx)
    ion_emis_slope = theta[(*ind, 1)]                           # (*, Nx)
    pool_radius = theta[(*ind, 2)]                              # (*, Nx)

    # Extract substrate properties
    res_pore_radius = eta[(*ind, 0)]                            # (*, Nx)
    permeability = eta[(*ind, 1)]

    # Extract material properties
    conductivity = eta[(*ind, 2)]                               # (*, Nx)
    surface_tension = eta[(*ind, 3)]
    density = eta[(*ind, 4)]
    viscosity = eta[(*ind, 5)]

    # Extract beam properties
    charge_to_mass = eta[(*ind, 6)]                             # (*, Nx)

    # Extract emitter geometries
    geo_data = eta[(*ind, slice(7, None))].reshape((*eta.shape[:-1], Ne, 7))
    geoms = geo_data[(*ind, slice(None), slice(6))]             # (*, Nx, Ne, 6)
    emax_sim = geo_data[(*ind, slice(None), 6)]                 # (*, Nx, Ne)
    curvature_radius = geoms[(*ind, slice(None), 0)]            # (*, Nx, Ne)
    # gap_distance = geoms[(*ind, slice(None), 1)]
    # aperture_radius = geoms[(*ind, slice(None), 2)]
    half_angle = geoms[(*ind, slice(None), 3)]
    emitter_height = geoms[(*ind, slice(None), 4)]
    loc_pore_radius = geoms[(*ind, slice(None), 5)]

    del geoms
    del geo_data

    # Define some useful slice indices
    Nx_idx = (*ind, cp.newaxis)                                 # (*, Nx, 1)
    Nxx_idx = (*ind, cp.newaxis, cp.newaxis)                    # (*, Nx, 1, 1)
    Ne_idx = (*ind, slice(None), cp.newaxis)                    # (*, Nx, Ne, 1)

    # Calculate some quantities common to all emitters and emission sites
    hydraulic_resistivity = viscosity / (2 * cp.pi * permeability)          # (*, Nx)
    res_pressure = -2 * surface_tension / res_pore_radius                   # (*, Nx)

    # Compute electric field
    voltage = voltage.reshape((1,) * len(emax_sim.shape[:-2]) + (Nx, 1))    # (...1, Nx, 1)
    applied_field = emax_sim * voltage                                      # (*, Nx, Ne)

    # Compute the number of sites for each emitter                          # (*, Nx, Ne)
    applied_electric_pressure = 1 / 2 * VACUUM_PERMITTIVITY * applied_field ** 2
    min_cap_pressure = 2 * surface_tension[Nx_idx] / (loc_pore_radius + pool_radius[Nx_idx])
    max_cap_pressure = 2 * surface_tension[Nx_idx] / loc_pore_radius

    no_sites_mask = applied_electric_pressure < min_cap_pressure - res_pressure[Nx_idx]
    all_sites_mask = applied_electric_pressure > max_cap_pressure - res_pressure[Nx_idx]

    mask_shape = all_sites_mask.shape                                       # (*, Nx, Ne)
    A_emission = 2 * cp.pi * curvature_radius ** 2 * (1 - cp.sin(half_angle))
    A_Taylor = cp.pi * ((pool_radius[Nx_idx] + loc_pore_radius) / 2) ** 2
    max_sites = A_emission / A_Taylor

    num_sites = cp.where(all_sites_mask, cp.floor(max_sites), 0)            # (*, Nx, Ne)
    neither_mask = ~(no_sites_mask | all_sites_mask)

    # Ignore errors from bad sites that we don't need
    with np.errstate(divide='ignore', invalid='ignore'):
        min_site_radius = 2 * surface_tension[Nx_idx] / (1 / 2 * VACUUM_PERMITTIVITY * applied_field ** 2 +
                                                         res_pressure[Nx_idx])
        arr = cp.floor(1 + (max_sites - 1) * (1 - (min_site_radius - loc_pore_radius) / pool_radius[Nx_idx]) ** 4)
        num_sites = num_sites + cp.where(neither_mask, arr, 0)

        Ns_max = int(cp.max(num_sites))
        site_nums = cp.arange(Ns_max).reshape((1,)*len(mask_shape) + (Ns_max, )) + 1    # (...1     , Ns_max)
        active_mask = site_nums <= num_sites[Ne_idx]                                    # (*, Nx, Ne, Ns_max)

        # Compute site radius
        site_radius = pool_radius[Nxx_idx] * (1 - ((site_nums-1) / (max_sites[Ne_idx]-1)) ** 0.25) + \
                      loc_pore_radius[Ne_idx]

        # Compute characteristic pressure
        char_pressure = 2 * surface_tension[Nxx_idx] / site_radius

        # Compute characteristic electric field and onset field
        char_elec_field = cp.sqrt(2 * char_pressure / VACUUM_PERMITTIVITY)
        onset_field = cp.sqrt((2 * (char_pressure - res_pressure[Nxx_idx])) / VACUUM_PERMITTIVITY)
        dimless_applied_field = applied_field[Ne_idx] / char_elec_field
        dimless_onset_field = onset_field / char_elec_field

        # Compute hydraulic impedance
        hydraulic_impedance = num_sites * hydraulic_resistivity[Nx_idx] / (1 - cp.cos(half_angle)) * \
                              (cp.tan(half_angle) / curvature_radius - cp.cos(half_angle) / emitter_height)
        dimless_res_pressure = res_pressure[Nxx_idx] / char_pressure
        dimless_hydraulic_impedance = conductivity[Nxx_idx] * char_elec_field * site_radius ** 2 * \
                                      hydraulic_impedance[Ne_idx] / (char_pressure * density[Nxx_idx] *
                                                                     charge_to_mass[Nxx_idx])

        # Compute current emitted by each active site
        dimless_current = (ion_emis_offset[Nxx_idx]+ion_emis_slope[Nxx_idx]*(dimless_applied_field-dimless_onset_field)
                           + dimless_res_pressure) / dimless_hydraulic_impedance

        arr = dimless_current * conductivity[Nxx_idx] * char_elec_field * site_radius ** 2

    site_current = cp.where(active_mask, arr, 0)
    site_current[site_current < 0] = 0

    # Sum current over all sites and emitters
    current = cp.sum(site_current, axis=(-1, -2))[Nx_idx]  # (*, Nx, 1)

    return cp.asnumpy(current)


def electrospray_current_model_cpu(x, theta, eta):
    """ CPU-HEAVY O(N) runtime, O(5MB/process) memory
    Predicts total array current for an electrospray thruster
    Parameters
    ----------
    x: (Nx, x_dim) voltage operating conditions [V]
    theta: (*, Nx, theta_dim) model parameters
    eta: (*, Nx, eta_dim) nuisance parameters

    Returns
    -------
    current: (*, Nx, y_dim) model output [A]
    """
    # Fix shapes
    x = fix_input_shape(x)
    Nx, x_dim = x.shape
    theta = np.atleast_1d(theta)
    if len(theta.shape) == 1:
        theta = np.expand_dims(theta, axis=0)
    eta = np.atleast_1d(eta)
    if len(eta.shape) == 1:
        eta = np.expand_dims(eta, axis=0)
    theta_shape = theta.shape[:-2]
    theta_dim = theta.shape[-1]
    eta_shape = eta.shape[:-2]
    eta_dim = eta.shape[-1]
    y_dim = 1
    voltage = x.reshape((1,) * len(eta_shape) + (Nx, x_dim))  # (...1, Nx, x_dim)

    # Problem setup
    VACUUM_PERMITTIVITY = 8.8542e-12
    Ne = int(np.floor((eta_dim - 7) / 7))    # number of emitters
    assert theta_dim == 3

    # Extract model parameters
    ion_emis_offset = theta[..., 0, np.newaxis]  # (*, 1)
    ion_emis_slope = theta[..., 1, np.newaxis]
    pool_radius = theta[..., 2, np.newaxis]

    # Extract substrate properties
    res_pore_radius = eta[..., 0, np.newaxis]
    permeability = eta[..., 1, np.newaxis]

    # Extract material properties
    conductivity = eta[..., 2, np.newaxis]
    surface_tension = eta[..., 3, np.newaxis]
    density = eta[..., 4, np.newaxis]
    viscosity = eta[..., 5, np.newaxis]

    # Extract beam properties
    charge_to_mass = eta[..., 6, np.newaxis]

    # Extract emitter geometries
    geo_data = eta[..., 7:].reshape((*eta_shape, Nx, Ne, 7))
    curvature_radius = geo_data[..., 0]  # (..., Nx, Ne)
    gap_distance = geo_data[..., 1]
    aperture_radius = geo_data[..., 2]
    half_angle = geo_data[..., 3]
    emitter_height = geo_data[..., 4]
    loc_pore_radius = geo_data[..., 5]
    emax_sim = geo_data[..., 6]
    # del geo_data

    # Calculate some quantities common to all emitters and emission sites   # (Nx, 1)
    hydraulic_resistivity = viscosity / (2 * np.pi * permeability)
    res_pressure = -2 * surface_tension / res_pore_radius

    # Compute electric field
    applied_field = emax_sim * voltage                                      # (..., Nx, Ne)

    # Compute the number of sites for each emitter                          # (..., Nx, Ne)
    applied_electric_pressure = 1 / 2 * VACUUM_PERMITTIVITY * applied_field ** 2
    min_cap_pressure = 2 * surface_tension / (loc_pore_radius + pool_radius)
    max_cap_pressure = 2 * surface_tension / loc_pore_radius

    no_sites_mask = applied_electric_pressure < min_cap_pressure - res_pressure
    all_sites_mask = applied_electric_pressure > max_cap_pressure - res_pressure

    A_emission = 2 * np.pi * curvature_radius ** 2 * (1 - np.sin(half_angle))
    A_Taylor = np.pi * ((pool_radius + loc_pore_radius) / 2) ** 2
    max_sites = A_emission / A_Taylor

    num_sites = np.where(all_sites_mask, np.floor(max_sites), 0)            # (..., Nx, Ne)
    neither_mask = ~(no_sites_mask | all_sites_mask)

    # Ignore errors from bad sites that we don't need
    with np.errstate(divide='ignore', invalid='ignore'):
        min_site_radius = 2 * surface_tension / (1 / 2 * VACUUM_PERMITTIVITY * applied_field ** 2 + res_pressure)
        arr = np.floor(1 + (max_sites - 1) * (1 - (min_site_radius - loc_pore_radius) / pool_radius) ** 4)
        num_sites = num_sites + np.where(neither_mask, arr, 0)

        Ns_max = np.max(num_sites).astype(int)
        site_nums = np.arange(Ns_max).reshape(((1,)*len(num_sites.shape) + (Ns_max,))) + 1      # (...1, Ns_max)
        active_mask = site_nums <= num_sites[..., np.newaxis]                                   # (..., Ns_max)

        # Compute site radius
        site_radius = pool_radius[..., np.newaxis] * (1 - ((site_nums-1) / (max_sites[..., np.newaxis]-1)) ** 0.25) + \
                      loc_pore_radius[..., np.newaxis]

        # Compute characteristic pressure
        char_pressure = 2 * surface_tension[..., np.newaxis] / site_radius

        # Compute characteristic electric field and onset field
        char_elec_field = np.sqrt(2 * char_pressure / VACUUM_PERMITTIVITY)
        onset_field = np.sqrt((2 * (char_pressure - res_pressure[..., np.newaxis])) / VACUUM_PERMITTIVITY)
        dimless_applied_field = applied_field[..., np.newaxis] / char_elec_field
        dimless_onset_field = onset_field / char_elec_field

        # Compute hydraulic impedance
        hydraulic_impedance = num_sites * hydraulic_resistivity / (1 - np.cos(half_angle)) * \
                              (np.tan(half_angle) / curvature_radius - np.cos(half_angle) / emitter_height)
        dimless_res_pressure = res_pressure[..., np.newaxis] / char_pressure
        dimless_hydraulic_impedance = conductivity[..., np.newaxis] * char_elec_field * site_radius ** 2 * \
                                      hydraulic_impedance[..., np.newaxis] / (char_pressure * density[..., np.newaxis] *
                                                                     charge_to_mass[..., np.newaxis])

        # Compute current emitted by each active site
        dimless_current = (ion_emis_offset[..., np.newaxis] + ion_emis_slope[..., np.newaxis] * (dimless_applied_field -
                            dimless_onset_field) + dimless_res_pressure) / dimless_hydraulic_impedance

        arr = dimless_current * conductivity[..., np.newaxis] * char_elec_field * site_radius ** 2

    site_current = np.where(active_mask, arr, 0)
    site_current[site_current < 0] = 0

    # Sum current over all sites and emitters
    current = np.sum(site_current, axis=(-1, -2))[..., np.newaxis]   # (..., 1)

    return current
