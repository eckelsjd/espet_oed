# Run surrogate inference on input emitter geometry -> max electric field

# EXAMPLE USAGE:
# as script: python esi_surrogate.py
# as import: from esi_surrogate import forward

import numpy as np
import scipy.io  # to load .mat data
import onnx      # to load trained neural network
import onnxruntime as ort  # to run inference
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

# Hardcoded normalization settings from training data
# Network inputs
XS_DEFAULT = {'xmin': np.array([[0.001931616642030], [0.007460017614712], [0.174549678663273], [0.002644155216795]]),
              'xmax': np.array([[1], [1], [1.221728349454381], [1]]),
              'ymin': np.array(-1), 'ymax': np.array(1),
              'lambda': np.array([[0.142427825812150], [6.640356835835101], [0], [0.222484472518862]]),
              'd_offset': np.array(0.001)
              }

# Network outputs
YS_DEFAULT = {'xmin': np.array([[0.009316156987571]]),
              'xmax': np.array([[1]]),
              'ymin': np.array(-1), 'ymax': np.array(1),
              'lambda': np.array(0.254636526066730)
              }

# Bias Voltage
V0_DEFAULT = 1000  # V

# Bounds on geometry parameters
H_BOUNDS = [50*1e-6, 1000*1e-6]       # m
ALPHA_BOUNDS = [10*np.pi/180, 70*np.pi/180]     # rad
RC_BOUNDS = [1*1e-6, 100*1e-6]        # m
D_BOUNDS = [-1000*1e-6, 3000*1e-6]    # m
RA_BOUNDS = [10*1e-6, 3000*1e-6]      # m
L_BOUNDS = [D_BOUNDS[0], RC_BOUNDS[0], ALPHA_BOUNDS[0], H_BOUNDS[0], RA_BOUNDS[0]]
U_BOUNDS = [D_BOUNDS[1], RC_BOUNDS[1], ALPHA_BOUNDS[1], H_BOUNDS[1], RA_BOUNDS[1]]


def parse_norm_settings(filename='../../data/geometry/models/norm_data.mat'):
    """Parse normalization settings from Matlab .mat file
    :param filename: path to norm_data.mat file generated from network training
    --Returns--
        xs: input normalization settings (dict)
        ys: output normalization settings (dict)
    """
    # Load data
    data = scipy.io.loadmat(filename)

    # Parse input norm settings
    xs = data['xs'][0][0]
    xmax_input = xs[2]
    xmin_input = xs[3]
    ymax_input = np.squeeze(xs[6])
    ymin_input = np.squeeze(xs[7])
    lambda_x = data['lambda_x'].T
    xoffset = np.squeeze(data['xoffset'])
    xs = {'xmin': xmin_input, 'xmax': xmax_input, 'ymin': ymin_input,
          'ymax': ymax_input, 'lambda': lambda_x, 'd_offset': xoffset}

    # Parse output norm settings
    ys = data['ys'][0][0]
    xmax_output = ys[2]
    xmin_output = ys[3]
    ymax_output = np.squeeze(ys[6])
    ymin_output = np.squeeze(ys[7])
    lambda_y = np.squeeze(data['lambda_y'])
    ys = {'xmin': xmin_output, 'xmax': xmax_output, 'ymin': ymin_output, 'ymax': ymax_output, 'lambda': lambda_y}

    return xs, ys


def mapminmax(x, ps, direction='Forward'):
    """
    Pre and post process normalization for FNN. Replicates Matlab's mapminmax function.
    Ni: number of inputs
    No: number of outputs
    Nd: number of data samples
    :param x: (Ni,Nd) if 'Forward', (No,Nd) if 'Reverse', data that is to be normalized
    :param ps: Normalization process settings (dict)
        :xmin: (Ni,1) if 'Forward', (No,1) if 'Reverse': minimum x value from NN training
        :xmax: (Ni,1) if 'Forward', (No,1) if 'Reverse': maximum x value from NN training
        :ymin: (1,1) mapping minimum for both NN inputs/outputs (assume they use the same mapping)
        :ymax: (1,1) mapping maximum for both NN inputs/outputs (assume they use the same mapping)
    :param direction: 'Forward' for network inputs, 'Reverse' for network outputs
    --returns--
        xnorm: Normalized inputs if 'Forward'
        y: Un-normalized outputs if 'Reverse'
    """
    # Unpack process settings
    xmin = ps['xmin']
    xmax = ps['xmax']
    ymin = ps['ymin']
    ymax = ps['ymax']

    if direction == 'Forward':
        # Apply the mapping x -> xnorm
        Ni, Nd = x.shape
        xmin_tile = np.tile(xmin, (1, Nd))
        xmax_tile = np.tile(xmax, (1, Nd))
        xnorm = np.divide((x - xmin_tile), (xmax_tile-xmin_tile))*(ymax-ymin) + ymin
        return xnorm
    elif direction == 'Reverse':
        # Undo the mapping y -> ynorm (equivalently, apply the mapping ynorm -> y)
        No, Nd = x.shape
        ynorm = x  # raw network output is already normalized
        ynorm_min = ymin
        ynorm_max = ymax
        ymin_tile = np.tile(xmin, (1, Nd))
        ymax_tile = np.tile(xmax, (1, Nd))
        ratio = (ynorm - ynorm_min)/(ynorm_max-ynorm_min)
        y = np.multiply(ratio, (ymax_tile - ymin_tile)) + ymin_tile
        return y
    else:
        print(f'Incorrect direction: {direction}')


def exp_cdf(x, lambda_coeff):
    """Exponential distribution cumulative distribution function (CDF), used for normalization"""
    return 1 - np.exp(-lambda_coeff*x)


def exp_cdf_inv(x, lambda_coeff):
    """Inverse exponential distribution cumulative distribution function (CDF), used for normalization"""
    return (-1/lambda_coeff)*np.log(np.abs(1-x))


def check_geom_constraints(x):
    """
    Check that all 3 practical geometry constraints are met (1. within limits, 2. height lower bound, 3. non-contact)
    :param x: (5, Nsamples) np array where each column is an emitter geometry [d, Rc, alpha, h, Ra]^T
                in units of [m, m, rad, m, m]
    :return (1, Nsamples) boolean array to indicate the idx of samples that did not meet all design constraints
    """
    d = x[0, :]  # Tip-to-extractor distance [m]
    rc = x[1, :]  # Radius of curvature [m]
    alpha = x[2, :]  # Cone half-angle [rad]
    h = x[3, :]  # Emitter height [m]
    ra = x[4, :]  # Radius of aperture [m]
    Nsamples = x.shape[1]

    # Check that all samples are in design space bounds
    lb = np.array(L_BOUNDS)[:, np.newaxis]
    lb = np.tile(lb, (1, Nsamples))
    ub = np.array(U_BOUNDS)[:, np.newaxis]
    ub = np.tile(ub, (1, Nsamples))
    ds_constraint = (x < lb) + (x > ub)

    # Constraints are set to true when they are violated
    ds_constraint = np.any(ds_constraint, axis=0)

    # Check lower height bound (5% for numerical safety)
    hl = np.multiply(rc, (1 - np.sin(alpha)))
    h_constraint = h < 1.05*hl

    # Check non-contact constraint
    # Cone base radius
    rb = np.multiply(rc, np.cos(alpha)) + np.multiply((h - hl), np.tan(alpha))

    # Cone height
    h0 = np.divide(rb, np.tan(alpha))

    # Get emitter cross-sectional radius at r(y=y_crit)
    y_crit = h - np.abs(d)

    # Mask for piecewise parameterization of emitter cone radius (first piece is linear region)
    b1 = 0 < y_crit
    b2 = y_crit < h - hl
    y_mask = b1 * b2
    r_crit_1 = (rb - np.multiply(np.divide(rb, h0), y_crit)) * y_mask

    # Second piece is circular arc
    b1 = h - hl <= y_crit
    b2 = y_crit <= h
    y_mask = b1 * b2
    yc = h - rc
    r_crit_2 = np.sqrt(np.abs(np.square(rc) - np.square(y_crit - yc))) * y_mask

    r_crit = r_crit_1 + r_crit_2

    # 5% margin for numerical safety
    d_constraint = (r_crit >= ra * 0.95) + (y_crit <= 0)
    d_sign = d < 0
    d_constraint = d_constraint * d_sign

    return ds_constraint + h_constraint + d_constraint


def forward(x, net_file='esi_surrogate.onnx', data_file=None, V0=None):
    """Run a forward inference pass of the network saved in net_file on the inputs x.
    :param x: (5, Nsamples) np array where each column is an emitter geometry [d, Rc, alpha, h, Ra]^T
                in units of [m, m, rad, m, m]
    :param net_file: Path to the .onnx trained network file
    :param data_file: Path to norm_data.mat file containing normalization settings
    :param V0: (Nsamples,) or (1,) bias voltage for each geometry. If size is (1,), same bias applied to all geometries
    :return (1, Nsamples) E_max [V/m] predictions for each of the Nsamples geometry inputs
    """
    # Check dimensions of input x
    x = np.atleast_1d(np.squeeze(x))
    if len(x.shape) == 1:
        Ninputs = x.shape[0]
        x = x[:, np.newaxis]  # Add new axis for single geometry sample
    else:
        Ninputs, Nsamples = x.shape
    assert Ninputs == 5

    # Check if all inputs meet practical geometry constraints
    constraints_failed = check_geom_constraints(x)

    if np.any(constraints_failed):
        warnings.warn('Forward network pass failed! Some of the inputs do not lie within the training bounds of the '
                      'surrogate. The bad inputs are flagged by index in the return value of this function.')
        return constraints_failed

    # Set normalization settings
    if data_file:
        xs, ys = parse_norm_settings(data_file)
    else:
        xs = XS_DEFAULT
        ys = YS_DEFAULT

    # NONDIMENSIONALIZE and APPLY CDF TRANSFORM
    d = x[0, :]       # Tip-to-extractor distance [m]
    rc = x[1, :]      # Radius of curvature [m]
    alpha = x[2, :]   # Cone half-angle [rad]
    h = x[3, :]       # Emitter height [m]
    ra = x[4, :]      # Radius of aperture [m]

    d_tilde = exp_cdf(np.divide(d + xs['d_offset'], h), xs['lambda'][0])
    rc_tilde = exp_cdf(np.divide(rc, h), xs['lambda'][1])
    ra_tilde = exp_cdf(np.divide(ra, h), xs['lambda'][3])
    x_nondim = np.vstack((d_tilde, rc_tilde, alpha, ra_tilde))

    # NORMALIZE INPUTS
    xnorm = mapminmax(x_nondim, xs, direction='Forward')

    # Load network weights from onnx file
    onnx_model = onnx.load(net_file)
    onnx.checker.check_model(onnx_model)

    # RUN NETWORK FORWARD PASS
    ort_sess = ort.InferenceSession(net_file)
    input_name = ort_sess.get_inputs()[0].name

    # Need to pass inputs in as shape (Nd, Ni)
    outputs = ort_sess.run(None, {input_name: xnorm.T.astype(np.float32)})
    ynorm = outputs[0].T  # (No, Nd)

    # REVERSE NORMALIZATION
    y = mapminmax(ynorm, ys, direction='Reverse')

    # Undo the CDF transform from training
    y = exp_cdf_inv(y, ys['lambda'])

    # Get Bias voltage
    if V0:
        V0 = np.atleast_1d(np.squeeze(V0))
        assert len(V0.shape) == 1
        if V0.shape[0] > 1:
            assert V0.shape[0] == y.shape[1]  # Must be same size as number of samples
    else:
        V0 = V0_DEFAULT

    # RE-DIMENSIONALIZE
    E_max = np.multiply(y, np.divide(V0, h))

    return E_max.flatten()


def main():
    # EXAMPLE USAGE
    data_dir = Path('../../data/geometry')
    test_data = np.loadtxt(str(data_dir/'test'/'samples'/'samples.txt'), skiprows=1)
    test_data[:, 2] = test_data[:, 2] * (np.pi / 180)

    E_max = forward(test_data.T, net_file=str(data_dir/'models'/'esi_surrogate.onnx'),
                    data_file=str(data_dir/'models'/'norm_data.mat'))
    plt.hist(E_max, density=True, bins=50, edgecolor='black')
    plt.xlabel('Electric field magnitude [V/m]')
    plt.ylabel('Empirical PDF over test set')
    plt.show()


if __name__ == '__main__':
    main()
