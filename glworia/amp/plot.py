import jax
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax.numpy as jnp

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.minor.size'] = 2

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.labeltop'] = plt.rcParams['ytick.labelright'] = False
mpl.rcParams['axes.unicode_minus'] = False

params = {'axes.labelsize': 18,
          'font.family': 'serif',
          'font.size': 9,
          'legend.fontsize': 12,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'savefig.dpi': 200,
          'lines.markersize': 6,
          'axes.formatter.limits': (-3, 3)}

mpl.rcParams.update(params)

def contour_plot_T(T, xlim = (-1, 1), ylim = (-1, 1), levels = 20):
    fig, ax = plt.subplots()

    x = jnp.linspace(*xlim, num = 100)
    y = jnp.linspace(*ylim, num = 100)
    X, Y = jnp.meshgrid(x, y)
    XX = jnp.dstack((X, Y))
    Z = T(XX)
    ax.contour(X, Y, Z, levels = levels)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return fig, ax

# def find_root_newt(x_init, hesT, dT, tol = 1e-8, imax = 100):
#     x = x_init
#     i = 0
#     x_list = jnp.array([x])
#     d = jnp.inf
#     init_val = [x_init, 0, d]
#     cond_fun = make_cond_fun(tol, imax)
#     body_fun = make_body_fun(hesT, dT)
#     x_iter = jax.lax.while_loop(cond_fun, body_fun, init_val)
#     return x_iter