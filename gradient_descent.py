import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from autograd import grad
from functools import partial
import autograd.numpy as np


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X - mux
    Ymu = Y - muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return 100 * np.exp(-z/(2*(1-rho**2))) / denom


def func(pt):
    x, y = pt
    z1 = bivariate_normal(x, y, 1.0, 1.0, 0.0, 0.0)
    z2 = bivariate_normal(x, y, 1.5, 0.5, 1, 1)
    return z2 - z1


def plot_func(f):

    y = np.linspace(-2, 2, 100)
    x = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, y)

    z = f([x, y])
    plt.pcolor(x, y, z, cmap="plasma")
    plt.plot(-0.02, -0.26, "ro")
    plt.tick_params(axis='both',labelbottom=False, labelleft=False)

def descent(f, x, lr=0.01, maxiter=1000, eps=0.01):
    i = 1
    delta = 1
    f_grad = grad(f)

    path = [x]
    while i < maxiter and delta > eps:

        step = f_grad(x) * lr
        i += 1
        delta = np.abs(f_grad(x - step)).sum()
        x = x - step
        path.append(x)

    return path


def momentum(f, x, lr=0.01, mom=0.5, maxiter=1000, eps=0.01):

    v = np.asarray((0, 0))

    i = 1
    delta = 1
    f_grad = grad(f)
    path = [x]
    while i < maxiter and delta > eps:
        v = mom * v + lr * f_grad(x)
        i += 1
        delta = np.abs(f_grad(x - v)).sum()
        x = x - v
        path.append(x)

    return path


def frames(path, n):

    plot_func(func)
    plt.plot(path[n][0], path[n][1], "k.")
    n_path = path[: n + 1]
    plt.plot([i[0] for i in n_path], [i[1] for i in n_path], "-k")


def make_gif(path):

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    frm = partial(frames, path[::2])
    anim = animation.FuncAnimation(
        fig, frm, frames=np.arange(0, len(path) // 2), interval=200
    )

    anim.save("grad.gif", dpi=80, writer="imagemagick")


if __name__ == "__main__":

    X = np.array([1.5, 0.73])
    path = momentum(func, X)
    make_gif(path)
