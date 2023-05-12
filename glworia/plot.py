import matplotlib.pyplot as plt
import jax.numpy as jnp

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

def find_root_newt(x_init, hesT, dT, tol = 1e-8, imax = 100):
    x = x_init
    i = 0
    x_list = jnp.array([x])
    d = jnp.inf
    init_val = [x_init, 0, d]
    cond_fun = make_cond_fun(tol, imax)
    body_fun = make_body_fun(hesT, dT)
    x_iter = jax.lax.while_loop(cond_fun, body_fun, init_val)
    return x_iter