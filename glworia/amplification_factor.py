import jax
from jax import jit
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

from glworia.lens_functions import *
from glworia.utils import *
from glworia.root import *
from glworia.contour import *

def chev_points(a, b, n):
    chev = -jnp.cos(jnp.pi*(jnp.arange(n)+0.5)/n)
    chev_inner_width = chev[-1]-chev[0]
    return (a+b)/2 + (b-a)/chev_inner_width * chev

def chev_first_half(a, b, n):
    chev = -jnp.cos(jnp.pi*(jnp.arange(n)+0.5)/n)
    chev_inner_width = chev[-1]-chev[0]
    reg = jnp.linspace(-1, 1, n)
    chev_half = jnp.where(reg < 0, 2*chev/chev_inner_width, reg)
    return (a+b)/2 + (b-a)/2 * chev_half

# def chev_points_np(a, b, n):
#     chev = -np.cos(np.pi*(np.arange(n)+0.5)/n)
#     chev_inner_width = chev[-1]-chev[0]
#     return (a+b)/2 + (b-a)/chev_inner_width * chev

# def chev_first_half_np(a, b, n):
#     chev = -np.cos(np.pi*(np.arange(n)+0.5)/n)
#     chev_inner_width = chev[-1]-chev[0]
#     reg = np.linspace(-1, 1, n)
#     chev_half = np.where(reg < 0, 2*chev/chev_inner_width, reg)
#     return (a+b)/2 + (b-a)/2 * chev_half

def make_T0_arr_multiple_chev(N, T_images, T0_max):
    T_im_sad = T_images[0]
    T_im_max = T_images[1]
    dt_around_image = T_im_sad / N**3
    T0_arr_low = chev_points(dt_around_image,
                              T_im_sad - dt_around_image,
                              N)
    T0_arr_mid_1 = chev_first_half(T_im_sad + dt_around_image,
                              2*T_im_sad,
                              N)
    T0_arr_mid_2 = jnp.linspace(2*T_im_sad + dt_around_image,
                              10*T_im_sad,
                              N)
    T0_arr_high = jnp.logspace(jnp.log10(10*T_im_sad + dt_around_image),
                               jnp.log10(T0_max),
                               N)
    T0_arr_sad_max = chev_points(T_im_sad + dt_around_image,
                                  T_im_max - dt_around_image,
                                  N)
    return jnp.array([T0_arr_low, T0_arr_mid_1,
                            T0_arr_mid_2, T0_arr_high]), T0_arr_sad_max

def make_T0_arr_multiple(N, T_images, T0_max):
    T_im_sad = T_images[0]
    T_im_max = T_images[1]
    dt_around_image = T_im_sad / N**3
    T0_arr_low = jnp.linspace(dt_around_image,
                              T_im_sad - dt_around_image,
                              N)
    T0_arr_mid_1 = jnp.linspace(T_im_sad + dt_around_image,
                              2*T_im_sad,
                              N)
    T0_arr_mid_2 = jnp.linspace(2*T_im_sad + dt_around_image,
                              10*T_im_sad,
                              N)
    T0_arr_high = jnp.logspace(jnp.log10(10*T_im_sad + dt_around_image),
                               jnp.log10(T0_max),
                               N)
    T0_arr_sad_max = jnp.linspace(T_im_sad + dt_around_image,
                                  T_im_max - dt_around_image,
                                  N)
    return jnp.array([T0_arr_low, T0_arr_mid_1,
                            T0_arr_mid_2, T0_arr_high]), T0_arr_sad_max


def amplification_computation_prep(Psi, **kwargs):

    settings = {'h': 0.01, 'newt_tol': 1e-8, 'newt_max_iter': 500, 'bisect_tol': 1e-9}
    settings.update(kwargs)

    h = settings['h']
    newt_tol = settings['newt_tol']
    newt_max_iter = settings['newt_max_iter']
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
    newt_step_fun = make_newton_1D_step_fun(dT_1D, ddT_1D, newt_max_iter)

    bisection_1D_var_arg_v = make_bisection_1D_var_arg_v()
    bisection_1D_v = make_bisection_1D_v()
    bisect_cond_fun = make_bisection_1D_cond_fun(bisect_tol)
    bisect_step_fun_ddPsi_1D = make_bisection_1D_step_fun(ddPsi_1D)
    bisect_step_fun_T_1D = make_bisection_1D_step_fun(T_1D)
    bisect_step_fun_ddT_1D = make_bisection_1D_step_fun(ddT_1D)

    contour_cond_func = contour_int_cond_func_full
    contour_step_func = make_contour_int_step_func(T, dT, dT_norm, f, T_hess_det, h)

    x_init_sad_max_routine = make_x_init_sad_max_routine(T_1D, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D)

    compute_contour_ints_sad_max = make_compute_contour_ints_sad_max(contour_cond_func, contour_step_func)

    helper_funcs = {'newt_cond_fun': newt_cond_fun, 'newt_step_fun': newt_step_fun,
                    'bisection_1D_var_arg_v': bisection_1D_var_arg_v, 'bisection_1D_v': bisection_1D_v,
                    'bisect_cond_fun': bisect_cond_fun, 'bisect_step_fun_ddPsi_1D': bisect_step_fun_ddPsi_1D,
                    'bisect_step_fun_T_1D': bisect_step_fun_T_1D, 'bisect_step_fun_ddT_1D': bisect_step_fun_ddT_1D,
                    'x_init_sad_max_routine': x_init_sad_max_routine,
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

def amplification_computation_for_interpolation(T_funcs, helper_funcs, crit_funcs, y, lens_params, **kwargs):
    
    settings = {'crit_bisect_x_low': -20, 'crit_bisect_x_high': 20, 'crit_bisect_x_num': 1000,
                'crit_screen_round_decimal': 7, 'T0_max': 1000, 'long_num': 10000, 'short_num':1000,
                'dense_width': 0.01, 'crit_run': False, 'N': 100}
    settings.update(kwargs)

    crit_bisect_x_low = settings['crit_bisect_x_low']
    crit_bisect_x_high = settings['crit_bisect_x_high']
    crit_bisect_x_num = settings['crit_bisect_x_num']
    crit_screen_round_decimal = settings['crit_screen_round_decimal']
    T0_max = settings['T0_max']
    long_num = settings['long_num']
    short_num = settings['short_num']
    dense_width = settings['dense_width']
    crit_run = settings['crit_run']
    N = settings['N']

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

    x_im = get_crit_points_1D(
                        crit_x_init_arr,
                        newt_cond_fun,
                        newt_step_fun,
                        y0,
                        lens_params,
                        crit_screen_round_decimal)
    T_images_raw = T_1D(x_im, y0, lens_params)
    T_min = T_images_raw[2]
    T_images = T_images_raw - T_min  

    lens_param_to_x_crit = crit_funcs['lens_param_to_x_crit']
    lens_param_to_y_crit = crit_funcs['lens_param_to_y_crit']
    x_crit_val = lens_param_to_x_crit(lens_params[0])
    y_crit_val = lens_param_to_y_crit(lens_params[0])
    T_vir = T_1D(x_crit_val, y0, lens_params) - T_min

    strong_lensing = y0 < y_crit_val

    if crit_run:
        T_images = T_images.at[0:2].set(T_vir)
        x_im = x_im.at[0:2].set(x_crit_val)
        multi_image = True
    elif strong_lensing:
        multi_image = True
    else:
        multi_image = False

    contour_int = contour_integral(x_im, multi_image,
                               T_funcs, y, lens_params, critical = crit_run,
                               )

    if not multi_image:
        T0_find_max = jnp.linspace(T_vir*0.5, T_vir*1.5, N)
        contour_int.set_T_max(jnp.max(T0_find_max))
        contour_int.find_T_outer(bisect_cond_fun, bisect_step_fun_T_1D)
        T_val_max = contour_int.find_T_for_max_u(T0_find_max,bisection_1D_v, bisect_cond_fun, 
                                    bisect_step_fun_T_1D, 
                        contour_cond_func, contour_step_func)
        T0_min_out_segs, T0_arr_sad_max = make_T0_arr_multiple_chev(N, jnp.array([T_val_max, jnp.nan, 0.]), T0_max)
    else:
        T0_min_out_segs, T0_arr_sad_max = make_T0_arr_multiple_chev(N, T_images, T0_max)

    contour_int.set_T0_segments(T0_min_out_segs, T0_arr_sad_max)
    contour_int.get_and_set_T_max()

    contour_int.find_T_outer(bisect_cond_fun, bisect_step_fun_T_1D)
    contour_int.make_x_inits(bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D)
    contour_int.contour_integrate(contour_cond_func, contour_step_func)

    if (not strong_lensing) or crit_run:
        contour_int.compute_mus()

    return contour_int

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

def crtical_curve_interpolants(param_arr, T_funcs, crit_curve_helper_funcs):
    x_crit, y_crit = get_crit_curve_1D(param_arr, T_funcs, crit_curve_helper_funcs, x_hi = 100.)
    lens_param_to_x_crit = lambda param: jnp.interp(param, param_arr, -x_crit)
    lens_param_to_y_crit = lambda param: jnp.interp(param, param_arr, y_crit)
    x_crit_to_lens_param = lambda x: jnp.interp(x, -x_crit, param_arr)
    y_crit_to_lens_param = lambda y: jnp.interp(y, y_crit, param_arr, left = jnp.nan, right = jnp.nan)
    y_crit_to_x_crit = lambda y: jnp.interp(y, y_crit, -x_crit)

    crit_funcs = {'lens_param_to_x_crit': lens_param_to_x_crit, 
                  'lens_param_to_y_crit': lens_param_to_y_crit,
                  'x_crit_to_lens_param': x_crit_to_lens_param,
                  'y_crit_to_lens_param': y_crit_to_lens_param,
                  'y_crit_to_x_crit': y_crit_to_x_crit}
    
    return crit_funcs