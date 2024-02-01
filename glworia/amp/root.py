import jax
import jax.numpy as jnp
from jax import vmap, grad, jit
from functools import partial
from .utils import *
from jax.experimental.host_callback import call

def newton_1D_cond_fun_full(x_iter, tol):
    r = x_iter[1]
    return jnp.abs(r) > tol

def return_nan(x):
    return jnp.nan

def return_self(x):
    return x

def newton_1D_step_fun_full(x_iter, f, df, max_iter):
    x = x_iter[0]
    args = x_iter[2]
    iter = x_iter[3]
    r = f(x, *args)/df(x, *args)
    r = jax.lax.cond(iter + 1 > max_iter, return_nan, return_self, r)
    return [x - r, r, args, iter + 1]

def make_newton_1D_cond_fun(tol):
    return lambda x_iter: newton_1D_cond_fun_full(x_iter, tol)

def make_newton_1D_step_fun(f, df, max_iter):
    return lambda x_iter: newton_1D_step_fun_full(x_iter, f, df, max_iter)

@partial(jnp.vectorize, excluded={1, 2, 3})
@partial(jit, static_argnums = (1, 2))
def newton_1D(x_init, cond_fun, step_fun, args):
    init_val = [x_init, jnp.inf, args, 0]
    x_iter = jax.lax.while_loop(cond_fun, step_fun, init_val)
    return x_iter[0]

def bisection_1D_cond_fun_full(x_iter, tol):
    x_low = x_iter[1]
    x_hi = x_iter[2]
    return jnp.abs(x_hi - x_low) > tol

def make_bisection_1D_cond_fun(tol):
    return lambda x_iter: bisection_1D_cond_fun_full(x_iter, tol)

def update_x_hi(x_T_arr):
    x_low = x_T_arr[0]
    T_low = x_T_arr[1]
    x_mid = x_T_arr[2]
    T_mid = x_T_arr[3]
    return x_low, T_low, x_mid, T_mid

def update_x_low(x_T_arr):
    x_mid = x_T_arr[2]
    T_mid = x_T_arr[3]
    x_hi = x_T_arr[4]
    T_hi = x_T_arr[5]
    return x_mid, T_mid, x_hi, T_hi

def bisection_1D_step_fun_full(x_iter, T):

    x_low = x_iter[1]
    x_hi = x_iter[2]
    T_low = x_iter[3]
    T_hi = x_iter[4]
    T0 = x_iter[5]
    args = x_iter[6]
    x_mid = (x_low + x_hi)/2
    T_mid = T(x_mid, *args) - T0

    x_low, T_low, x_hi, T_hi = jax.lax.cond(
        T_low*T_mid < 0, update_x_hi, update_x_low,
        jnp.array([x_low, T_low, x_mid, T_mid, x_hi, T_hi]))

    return [x_mid, x_low, x_hi, T_low, T_hi, T0, args]

def make_bisection_1D_step_fun(T):
    return lambda x_iter: bisection_1D_step_fun_full(x_iter, T)

@partial(jit, static_argnums = (0, 4, 5))
def bisection_1D(T, T0, x_low, x_hi, cond_fun, step_fun, args):
    
    T_low = T(x_low, *args) - T0
    T_hi = T(x_hi, *args) - T0
    x_mid = (x_low + x_hi)/2

    bisection_sign_opposite = jnp.where(T_low*T_hi < 0, 1, jnp.nan)
    
    init_val = [x_mid, x_low, x_hi, T_low, T_hi, T0, args]
    
    x_iter = jax.lax.while_loop(cond_fun, step_fun, init_val)

    return x_iter[0]

def make_bisection_1D_v():
    return jnp.vectorize(bisection_1D, excluded={0, 2, 3, 4, 5, 6})

# def bisection_1D_var_arg(F, F0, arg, x_low, x_hi, tol = 1e-7):
#     T = lambda x: F(x, arg)
#     return bisection_1D(T, F0, x_low, x_hi, tol = tol)

def make_bisection_1D_var_arg_v():
    return jnp.vectorize(bisection_1D, excluded={0, 1, 2, 3, 4, 5}, signature = '(1,1)->()')

def make_bisection_1D_var_2D():
    bisection_1D_var_2D = lambda T, T0, x_low, x_hi, cond_fun, step_fun, y, lp: bisection_1D(T, T0, x_low, x_hi, cond_fun, step_fun, [y, jnp.atleast_1d(lp)])
    return jnp.vectorize(bisection_1D_var_2D, excluded={0, 1, 2, 3, 4, 5}, signature = '(),()->()')

def get_crit_points_1D(x_init_arr, cond_fun, step_fun, y, lens_params, round_decimal = 8):
    args = (y, jnp.atleast_1d(lens_params))
    crit_points_full = (newton_1D(x_init_arr, cond_fun, step_fun, args))
    unique, count = jnp.unique(jnp.round(crit_points_full, round_decimal), return_counts = True, size = 20, fill_value = jnp.nan)
    count_adj = jnp.where(jnp.abs(unique) > 0.1, 100*count, count)
    # print(unique, count, count_adj)
    indices_sorted = jnp.argsort(-count_adj)
    # uniques_sorted = unique[indices_sorted]    
    # third_best_count = -jnp.sort(-count)[2]
    crit_points_screened = unique[indices_sorted[:3]]
    # crit_points_screened = jnp.unique(unique_top_three, size = 3, fill_value = jnp.nan)
    crit_points_screened = nan_to_const(crit_points_screened, 0.)
    crit_points_screened = jnp.sort(crit_points_screened)
    # crit_sad_max = -jnp.sort(-crit_points_screened[:2])
    # crit_points_screened = crit_points_screened.at[:2].set(crit_sad_max)
    return const_to_nan(crit_points_screened, 0.)

# @partial(jnp.vectorize, excluded={0,1,2,5}, signature = '(),()->(3)')
# # @partial(jit, static_argnums = (1, 2, 5))
# def get_crit_points_vec(x_init_arr, cond_fun, step_fun, y, lens_params, round_decimal = 8):
#     args = (y, jnp.atleast_1d(lens_params))
#     crit_points_full = (newton_1D(x_init_arr, cond_fun, step_fun, args))
#     # crit_points_screened = -jnp.unique(jnp.round(crit_points_full, round_decimal), size = 3, fill_value = 0.)
#     unique, count = jnp.unique(jnp.round(crit_points_full, round_decimal), return_counts = True, size = 20, fill_value = jnp.nan)
#     third_best_count = -jnp.sort(-count)[2]
#     unique_top_three = jnp.where(count >= third_best_count, unique, jnp.nan)
#     crit_points_screened = jnp.unique(unique_top_three, size = 3, fill_value = jnp.nan)
#     crit_points_screened = nan_to_const(crit_points_screened, 0.)
#     crit_points_screened = jnp.sort(crit_points_screened)
#     return const_to_nan(crit_points_screened, 0.)

def make_crit_curve_helper_func(T_funcs, crit_bisect_tol = 1e-9):
    ddPsi_1D = T_funcs['ddPsi_1D']

    bisection_1D_var_arg_v = make_bisection_1D_var_arg_v()
    bisect_cond_fun_crit = make_bisection_1D_cond_fun(crit_bisect_tol)
    bisect_step_fun_ddPsi_1D = make_bisection_1D_step_fun(ddPsi_1D)

    crit_curve_helper_funcs = {'bisection_1D_var_arg_v': bisection_1D_var_arg_v,
                               'bisect_cond_fun_crit': bisect_cond_fun_crit,
                               'bisect_step_fun_ddPsi_1D': bisect_step_fun_ddPsi_1D}
    
    return crit_curve_helper_funcs

def get_crit_curve_1D_raw(dPsi_1D_param_free, ddPsi_1D, param_arr, bisection_1D_var_arg_v, cond_fun, step_fun, x_low = 1e-8, x_hi = 10.):
    param_arr_2D = jnp.atleast_2d(param_arr).T
    param_arr_3D = jnp.atleast_3d(param_arr_2D)
    x_crit = bisection_1D_var_arg_v(ddPsi_1D, 1, x_low, x_hi, cond_fun, step_fun, param_arr_3D)
    y_crit = dPsi_1D_param_free(x_crit, param_arr_2D) - x_crit
    return x_crit, y_crit

def get_crit_curve_1D(param_arr, T_funcs, crit_helper_funcs, x_low = 1e-8, x_hi = 10.):
    dPsi_1D_param_free = T_funcs['dPsi_1D_param_free']
    ddPsi_1D = T_funcs['ddPsi_1D']
    bisection_1D_var_arg_v = crit_helper_funcs['bisection_1D_var_arg_v']
    cond_fun = crit_helper_funcs['bisect_cond_fun_crit']
    step_fun = crit_helper_funcs['bisect_step_fun_ddPsi_1D']
    return get_crit_curve_1D_raw(dPsi_1D_param_free, ddPsi_1D, param_arr,
                                 bisection_1D_var_arg_v, cond_fun, step_fun,
                                 x_low = x_low, x_hi = x_hi)

def linear_interp(x, y):
    return lambda x_interp: jnp.interp(x_interp, x, y)
