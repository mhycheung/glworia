import jax.numpy as jnp
import numpy as np

def make_dstack(xlim = (-1, 1), ylim = (-1, 1), xnum = 100, ynum = 100):
    x = jnp.linspace(*xlim, num = xnum)
    y = jnp.linspace(*ylim, num = ynum)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.dstack((X, Y))
    return X, Y, Z

def local_minima(array2d):
    return ((array2d < jnp.roll(array2d,  1, 0)) &
            (array2d < jnp.roll(array2d, -1, 0)) &
            (array2d < jnp.roll(array2d,  1, 1)) &
            (array2d < jnp.roll(array2d, -1, 1)))

def find_root(ddT2, x_init, tol = 1e-5, a = 1e-3):
    x = x_init
    d = jnp.linalg.norm(ddT2(x))
    i = 0
    x_list = np.array([x])
    d_list = np.array([d])
    while d > tol and i < 100:
        d_vec = ddT2(x)
        x_new = x - a*d_vec
        d = jnp.linalg.norm(x_new - x)
        x_list = np.append(x_list, [x], axis = 0)
        d_list = np.append(d_list, [d])
        i += 1
        x = x_new
    return x, x_list, d_list, i

def turn_1D_to_2D(xs, x0):
    return jnp.column_stack((xs, x0*jnp.ones(len(xs))))

def pass_fun(*args, **kwargs):
    return True

def raise_jax_error(*args, **kwargs):
    return False

def nan_to_const(x, const = 0):
    return jnp.where(jnp.isnan(x), const, x)

def const_to_nan(x, const = 0):
    return jnp.where(x == const, jnp.nan, x)

def pad_to_len_3(x, const):
    return jnp.pad(x, (3 - len(x), 0), 'constant', constant_values = const)

def make_points_arr_mesh(x, y):
    X, Y = jnp.meshgrid(x, y)
    return jnp.vstack([X.ravel(), Y.ravel()]).T