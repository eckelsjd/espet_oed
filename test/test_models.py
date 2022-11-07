import numpy as np
import matplotlib.pyplot as plt

from src.models import linear_gaussian_model, nonlinear_model
from src.utils import get_cycle


def test_linear_gaussian_model():
    theta = np.array([0.25]).reshape(1, 1)
    eta = np.array([-0.6]).reshape(1, 1)
    Nx = 100
    d = np.linspace(0, 1, Nx).reshape((Nx, 1))
    y = linear_gaussian_model(d, theta, eta)

    plt.figure()
    plt.plot(d, y[:, 0], '-r', label=r'$y_1$')
    plt.plot(d, y[:, 1], '-b', label=r'$y_2$')
    plt.xlabel(r'Operating condition $d$')
    plt.ylabel(r'Model output $y$')
    plt.legend()
    plt.show()


def test_nonlinear_model():
    d = np.array([0, 0.1, 0.2, 0.5, 1])
    Nx = d.shape[0]
    Ntheta = 100
    theta = np.linspace(0, 1, Ntheta).reshape((Ntheta, 1, 1))
    eta = np.array([1]).reshape((1, 1, 1))
    y = nonlinear_model(d, theta, eta)  # (Ntheta, Nx, 1)

    plt.figure()
    plt.rcParams["axes.prop_cycle"] = get_cycle("tab10")
    ax = plt.gca()
    for i in range(Nx):
        plt.plot(theta[:, 0, 0], y[:, i, 0], label=f'd={d[i]}')

    plt.xlabel(r'Model parameter $\theta$')
    plt.ylabel(r'Model output $y$')
    plt.xlim([0, 1])
    plt.ylim([0, 1.2*y.max()])
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    leg = plt.legend()
    frame = leg.get_frame()
    frame.set_edgecolor('k')
    frame.set_facecolor([0.9, 0.9, 0.9, 0.6])
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')

    # test_linear_gaussian_model()
    test_nonlinear_model()
