import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from autograd import grad
from functools import partial

def f(X):
        x, y = X
        return (x-1)**2 + (y-1)**2

def plot_func(f):

        y = np.linspace(-1, 3, 100)
        x = np.linspace(-2, 2, 100)
        x, y = np.meshgrid(x, y)

        z = f([x, y])
        plt.pcolor(x, y, z, cmap='plasma')
        plt.plot(1, 1, 'ro')


def gradient_descent(f, lr=0.1, maxiter=100, eps=0.01):

        X = np.array([-1., -1.])

        i = 1
        delta = 1
        f_grad = grad(f)

        path = [X]
        while i < maxiter and delta > eps:

                step = f_grad(X) * lr
                i += 1
                delta = np.abs(f_grad(X-step)).sum()
                X = X - step
                path.append(X)

        return path


def frames(path, n):
        
        plot_func(f)
        plt.plot(path[n][0], path[n][1], 'k.')
        n_path = path[:n+1]
        plt.plot([i[0] for i in n_path], [i[1] for i in n_path], '-k')


def make_gif(path):

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        frm = partial(frames, path[::2])
        anim = animation.FuncAnimation(
                     fig, frm, frames=np.arange(0, len(path)//2), interval=200)

        anim.save('grad.gif', dpi=80, writer='imagemagick')

if __name__ == '__main__':

        a, b = 1, 100

        path = gradient_descent(f)
        make_gif(path)
