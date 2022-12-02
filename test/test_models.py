import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import linear_gaussian_model, custom_nonlinear
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


def test_custom_nonlinear():
    N = 100
    d = np.linspace(0, 1, N)
    theta = np.linspace(0, 1, N)
    pt_grids = np.meshgrid(d, theta)
    x_loc = np.vstack([grid.ravel() for grid in pt_grids]).T  # (np.prod(Nx), x_dim)
    x = x_loc[:, 0, np.newaxis]
    t = x_loc[:, 1, np.newaxis]
    eta = np.ones(t.shape)*(1/8)*0

    y = custom_nonlinear(x, t, eta)

    # Reform grids
    dg, tg = [x_loc[:, i].reshape((N, N)) for i in range(2)]  # reform grids
    yg = y.reshape((N, N))

    fig, ax = plt.subplots()
    c = ax.contourf(dg, tg, yg, 60, cmap='jet')
    plt.colorbar(c, label=r'Forward model $G(\theta, \phi, d)$')
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.set_xlabel(r'Operating condition $d$')
    ax.set_ylabel(r'Model parameter $\theta$')
    fig.set_size_inches(4.8, 3.6)
    plt.tight_layout()
    fig.savefig(str(Path('../results/figs') / 'nonlinear_model_contour.png'), dpi=300, format='png')
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
    # test_nonlinear_model()
    test_custom_nonlinear()
