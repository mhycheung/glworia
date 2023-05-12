import jax
from jax import jit
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

from glworia.lens_functions import *
from glworia.utils import *
from glworia.root import *
from glworia.contour import *

def amplification_computation_prep(Psi, **kwargs):

    settings = {'h': 0.01, 'newt_tol': 1e-9, 'bisect_tol': 1e-9}
    settings.update(kwargs)

    h = settings['h']
    newt_tol = settings['newt_tol']
    bisect_tol = settings['bisect_tol']

    T_funcs = make_T_funcs(Psi)

    T = T_funcs['T']
    dT = T_funcs['dT']
    dT_norm = T_funcs['dT_norm']
    f = T_funcs['f']
    T_hess_det = T_funcs['T_hess_det']
    T_1D = T_funcs['T_1D']
    dT_1D = T_funcs['dT_1D']
    ddT_1D = T_funcs['ddT_1D']
    ddPsi_1D = T_funcs['ddPsi_1D']
    
    
    newt_cond_fun = make_newton_1D_cond_fun(newt_tol)
    newt_step_fun = make_newton_1D_step_fun(dT_1D, ddT_1D)

    bisection_1D_var_arg_v = make_bisection_1D_var_arg_v()
    bisection_1D_v = make_bisection_1D_v()
    bisect_cond_fun = make_bisection_1D_cond_fun(bisect_tol)
    bisect_step_fun_ddPsi_1D = make_bisection_1D_step_fun(ddPsi_1D)
    bisect_step_fun_T_1D = make_bisection_1D_step_fun(T_1D)

    contour_cond_func = contour_int_cond_func_full
    contour_step_func = make_contour_int_step_func(T, dT, dT_norm, f, T_hess_det, h)

    x_init_sad_max_routine = make_x_init_sad_max_routine(T_1D, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D)

    compute_contour_ints_sad_max = make_compute_contour_ints_sad_max(contour_cond_func, contour_step_func)

    helper_funcs = {'newt_cond_fun': newt_cond_fun, 'newt_step_fun': newt_step_fun,
                    'bisection_1D_var_arg_v': bisection_1D_var_arg_v, 'bisection_1D_v': bisection_1D_v,
                    'bisect_cond_fun': bisect_cond_fun, 'bisect_step_fun_ddPsi_1D': bisect_step_fun_ddPsi_1D,
                    'bisect_step_fun_T_1D': bisect_step_fun_T_1D, 'x_init_sad_max_routine': x_init_sad_max_routine,
                    'compute_contour_ints_sad_max': compute_contour_ints_sad_max,
                    'contour_cond_func': contour_cond_func, 'contour_step_func': contour_step_func}
    
    return T_funcs, helper_funcs

def amplification_computation(T_funcs, helper_funcs, y, lens_params, **kwargs):
    
    settings = {'crit_bisect_x_low': -10, 'crit_bisect_x_high': 10, 'crit_bisect_x_num': 100,
                'crit_screen_round_decimal': 8, 'T0_max': 10, 'long_num': 10000, 'short_num':1000,
                'dense_width': 0.01}
    settings.update(kwargs)

    crit_bisect_x_low = settings['crit_bisect_x_low']
    crit_bisect_x_high = settings['crit_bisect_x_high']
    crit_bisect_x_num = settings['crit_bisect_x_num']
    crit_screen_round_decimal = settings['crit_screen_round_decimal']
    T0_max = settings['T0_max']
    long_num = settings['long_num']
    short_num = settings['short_num']
    dense_width = settings['dense_width']

    crit_x_init_arr = jnp.linspace(crit_bisect_x_low, crit_bisect_x_high, crit_bisect_x_num)

    T = T_funcs['T']
    dT = T_funcs['dT']
    dT_norm = T_funcs['dT_norm']
    f = T_funcs['f']
    T_hess_det = T_funcs['T_hess_det']
    T_1D = T_funcs['T_1D']
    mu = T_funcs['mu']

    newt_cond_fun = helper_funcs['newt_cond_fun']
    newt_step_fun = helper_funcs['newt_step_fun']
    bisection_1D_v = helper_funcs['bisection_1D_v']
    bisect_cond_fun = helper_funcs['bisect_cond_fun']
    bisect_step_fun_T_1D = helper_funcs['bisect_step_fun_T_1D']
    x_init_sad_max_routine = helper_funcs['x_init_sad_max_routine']
    compute_contour_ints_sad_max = helper_funcs['compute_contour_ints_sad_max']
    contour_cond_func = helper_funcs['contour_cond_func']
    contour_step_func = helper_funcs['contour_step_func']

    y0 = y[0]

    crit_points_screened = get_crit_points_1D(crit_x_init_arr, newt_cond_fun, newt_step_fun, y0, lens_params, round_decimal = crit_screen_round_decimal)
    T_images_raw = T_1D(crit_points_screened, y0, lens_params)
    T_images = T_images_raw - T_images_raw[2]   
    T_images = pad_to_len_3(T_images, jnp.nan)

    T0_arr = make_adaptive_T0_arr(T_images, T0_max, long_num, short_num, dense_width = dense_width)

    contour_integral = contour_init_1D(crit_points_screened, T0_arr, T_1D, T,
                                 dT, dT_norm, f, T_hess_det, mu, y, lens_params,
                                 long_num, short_num, dense_width)
    contour_integral.find_outer(bisect_cond_fun, bisect_step_fun_T_1D)
    contour_integral.get_T0_sad_max()
    contour_integral.make_x_inits(bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D, x_init_sad_max_routine)
    contour_integral.computer_contour_ints(contour_cond_func, contour_step_func, compute_contour_ints_sad_max)
    contour_integral.sum_results()

    return contour_integral

def amplification_fft(contour_integral, fft_len, t_min = 0, t_max = None):
    
    T_min = contour_integral.min.T
    if t_max is None:
        t_max = contour_integral.T0_arr[-1] - T_min
    t_fft = jnp.linspace(t_min, t_max, num = fft_len)
    dt = t_fft[1] - t_fft[0]
    Ft_fft = jnp.interp(t_fft, contour_integral.T0_arr - T_min, contour_integral.u_sum)

    w_arr = jnp.linspace(0, 2*jnp.pi/dt, num = fft_len)
    Fw_raw = w_arr*jnp.fft.fft(Ft_fft)*dt
    Fw = -jnp.imag(Fw_raw) - 1.j*jnp.real(Fw_raw) + Ft_fft[-1]

    return w_arr, Fw

def F_geom(ws, crit_points_screened, T_1D, mu, y0, lens_params):
    Ts = T_1D(crit_points_screened, y0, lens_params)
    Ts = nan_to_const(Ts, 1)
    T_sad = Ts[0] - Ts[2]
    T_max = Ts[1] - Ts[2]
    mus = mu(crit_points_screened, lens_params)
    mus = nan_to_const(mus, 0)
    sqrt_mus = jnp.sqrt(jnp.abs(mus))
    F = sqrt_mus[2] + sqrt_mus[0]*jnp.exp(1.j*(ws*T_sad - jnp.pi*0.5)) + sqrt_mus[1]*jnp.exp(1.j*(ws*T_max - jnp.pi*1.0))
    return F

def F_geom_from_contour(ws, contour_integral, T_funcs):
    crit_points_screened = contour_integral.crit_points_screened
    crit_points_pad = pad_to_len_3(crit_points_screened, jnp.nan)
    T_1D = T_funcs['T_1D']
    mu = T_funcs['mu']
    y0 = contour_integral.y0
    lens_params = contour_integral.lens_params
    return F_geom(ws, crit_points_pad, T_1D, mu, y0, lens_params)