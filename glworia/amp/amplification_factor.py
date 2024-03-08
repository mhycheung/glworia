from typing import List, Tuple, Union, Optional, Dict, Any, Callable
from scipy.interpolate import interp1d
from .frequency_domain import *
from .contour import *
from .root import *
from .utils import *
from .lens_functions import *
import jax
from jax import jit
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)


def chev_points(a: float, b: float, n: int) -> jnp.ndarray:
    """
    Returns n Chebyshev points in the interval [a, b].
    """
    chev = -jnp.cos(jnp.pi*(jnp.arange(n)+0.5)/n)
    chev_inner_width = chev[-1]-chev[0]
    return (a+b)/2 + (b-a)/chev_inner_width * chev


def chev_first_half(a: float, b: float, n: int) -> jnp.ndarray:
    """
    Returns the first n points out of 2*n Chebyshev points, with the n points covering the interval [a, b].
    """
    chev = -jnp.cos(jnp.pi*(jnp.arange(n)+0.5)/n)
    chev_inner_width = chev[-1]-chev[0]
    reg = jnp.linspace(-1, 1, n)
    chev_half = jnp.where(reg < 0, 2*chev/chev_inner_width, reg)
    return (a+b)/2 + (b-a)/2 * chev_half


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
    T_im_max_eff = jnp.min(jnp.array([T_im_max - dt_around_image, T0_max]))
    T0_arr_sad_max = chev_points(T_im_sad + dt_around_image,
                                 T_im_max_eff,
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
    T_im_max_eff = jnp.min(jnp.array([T_im_max - dt_around_image, T0_max]))
    T0_arr_sad_max = jnp.linspace(T_im_sad + dt_around_image,
                                  T_im_max_eff,
                                  N)
    return jnp.array([T0_arr_low, T0_arr_mid_1,
                      T0_arr_mid_2, T0_arr_high]), T0_arr_sad_max


def amplification_computation_prep(Psi: Callable, **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
    """
    Constructs the functions needed for computing the amplification factor.

    Parameters:
        Psi: The Fermat potential of the lens model.

    Keyword Args:
        h (float): The step size for numerical differentiation.
        newt_tol (float): The tolerance for the Newton's method.
        newt_max_iter (int): The maximum number of iterations for the Newton's method.
        bisect_tol (float): The tolerance for the bisection method.

    Returns:
        T_funcs: A dictionary of functions needed for computing the amplification factor.
        helper_funcs: A dictionary of helper functions needed for computing the amplification factor.
    """

    settings = {'h': 0.01, 'newt_tol': 1e-8,
                'newt_max_iter': 500, 'bisect_tol': 1e-9}
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
    contour_step_func = make_contour_int_step_func(
        T, dT, dT_norm, f, T_hess_det, h)

    x_init_sad_max_routine = make_x_init_sad_max_routine(
        T_1D, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D)

    compute_contour_ints_sad_max = make_compute_contour_ints_sad_max(
        contour_cond_func, contour_step_func)

    helper_funcs = {'newt_cond_fun': newt_cond_fun, 'newt_step_fun': newt_step_fun,
                    'bisection_1D_var_arg_v': bisection_1D_var_arg_v, 'bisection_1D_v': bisection_1D_v,
                    'bisect_cond_fun': bisect_cond_fun, 'bisect_step_fun_ddPsi_1D': bisect_step_fun_ddPsi_1D,
                    'bisect_step_fun_T_1D': bisect_step_fun_T_1D, 'bisect_step_fun_ddT_1D': bisect_step_fun_ddT_1D,
                    'x_init_sad_max_routine': x_init_sad_max_routine,
                    'compute_contour_ints_sad_max': compute_contour_ints_sad_max,
                    'contour_cond_func': contour_cond_func, 'contour_step_func': contour_step_func}

    return T_funcs, helper_funcs


@partial(jnp.vectorize, signature='(),()->()')
def y_crit_override_default(y_crit, lens_params):
    y_crit = jax.lax.cond(y_crit < 0, lambda y_crit: jnp.zeros_like(
        y_crit), lambda y_crit: y_crit, y_crit)
    return y_crit


def x_im_nan_sub_default(x_im, y0, lens_params):
    return x_im


def origin_type_default(lens_params):
    return 'regular'


@partial(jnp.vectorize, signature='(n,2)->(n)')
def add_to_strong_default(point):
    return jnp.full(point.shape[0], False)


def amplification_computation_for_interpolation(T_funcs: Dict[str, Callable], helper_funcs: Dict[str, Callable], crit_funcs: Dict[str, Callable], y: jnp.ndarray, lens_params: jnp.ndarray, **kwargs) -> Union[contour_integral, tuple]:
    """
    Compute the contour integral for constructing interpolation tables of the time domain amplification factor.

    Parameters:
        T_funcs: A dictionary of functions derived from the Fermat potential needed for computing the amplification factor.
        helper_funcs: A dictionary of helper functions needed for the numerical implementation.
        crit_funcs: A dictionary of functions for computing the caustic curve.
        y: The 2D impact parameter. For now the second component is not used.
        lens_params: The lens model parameter. For now only one parameter is supported.

    Keyword Args:
        im_x_init_low (float): The lower bound for the initial guess of the image position.
        im_x_init_high (float): The upper bound for the initial guess of the image position.
        im_x_init_num (int): The number of initial guesses of the image position.
        im_screen_round_decimal (int): The number of decimal places to round the image position.
        T0_max (float): The maximum value of the time delay to compute the time domain amplification up to.
        crit_run (bool): Whether the run corresponds to a run on the caustic.
        N (int): The number of interpolation node in time in each segment.
        return_all (bool): Whether to return all the computed values for debugging.
        singular (bool): Whether the center of the lens is singular, i.e. a cusp or a pole in the time delay function.
        origin_type (Callable): A function that returns the origin type of the lens model (cusp, pole or regular).
        y_crit_override (Callable): A function that overrides the computed values of the caustic curve.
        x_im_nan_sub (Callable): A function that substitutes `nan` values for the image position.
        add_to_strong (Callable): A function decides whether to add certain points to the strong lensing regime.
        T_vir_low_bound (float): The lower bound in time delay when searching for the maximum point of the time domain amplification.

    Returns:
        contour_int (contour_integral): The contour integral object.
        T0_min_out_segs (list[jnp.ndarray]): The corresponding time delays of the interpolation nodes for contour lines around the minimum. Each element of the list corresponds to a segment.
        T0_arr_sad_max (jnp.ndarray): The corresponding time delays of the interpolation nodes for contour lines around the maximum point.
        x_im (jnp.ndarray): The image positions.
        T_val_max (float): The time delay at the maximum value of the time domain amplification.
    """

    settings = {'im_x_init_low': -20, 'im_x_init_high': 20, 'im_x_init_num': 1000,
                'im_screen_round_decimal': 7, 'T0_max': 1000,
                'crit_run': False, 'N': 100, 'return_all': False, 'singular': False,
                'origin_type': origin_type_default, 'y_crit_override': y_crit_override_default,
                'x_im_nan_sub': x_im_nan_sub_default, 'add_to_strong': add_to_strong_default,
                'T_vir_low_bound': 1e-2}
    settings.update(kwargs)

    return_all = settings['return_all']

    im_x_init_low = settings['im_x_init_low']
    im_x_init_high = settings['im_x_init_high']
    im_x_init_num = settings['im_x_init_num']
    im_screen_round_decimal = settings['im_screen_round_decimal']
    T0_max = settings['T0_max']
    crit_run = settings['crit_run']
    N = settings['N']
    singular = settings['singular']

    origin_type = settings['origin_type']
    origin = origin_type(lens_params)

    if origin not in ['regular', 'im', 'cusp', 'pole']:
        raise ValueError('origin must be one of regular, im, cusp, or pole')

    im_x_init_arr = jnp.linspace(
        im_x_init_low, im_x_init_high, im_x_init_num)

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

    y_crit_override = settings['y_crit_override']
    x_im_nan_sub = settings['x_im_nan_sub']
    add_to_strong = settings['add_to_strong']

    y0 = y[0]

    x_im_raw = get_crit_points_1D(
        im_x_init_arr,
        newt_cond_fun,
        newt_step_fun,
        y0,
        lens_params,
        im_screen_round_decimal)
    lens_params = jnp.atleast_1d(lens_params)
    x_im = x_im_nan_sub(x_im_raw, y0, lens_params[0])
    if origin != 'regular':
        x_im = x_im.at[1].set(0.)
    T_images_raw = T_1D(x_im, y0, lens_params)
    T_min = T_images_raw[2]
    T_images = T_images_raw - T_min

    lens_param_to_x_crit = crit_funcs['lens_param_to_x_crit']
    lens_param_to_y_crit = crit_funcs['lens_param_to_y_crit']
    x_crit_val = lens_param_to_x_crit(lens_params[0])
    y_crit_val = lens_param_to_y_crit(lens_params[0])
    T_vir = T_1D(x_crit_val, y0, lens_params) - T_min

    y_crit_val_overriden = y_crit_override(y_crit_val, lens_params)
    strong_lensing = (y0 < y_crit_val_overriden)
    overrode = (y_crit_val_overriden != y_crit_val)

    if crit_run:
        T_images = T_images.at[0:2].set(T_vir)
        x_im = x_im.at[0:2].set(x_crit_val)
        multi_image = True
    elif strong_lensing or add_to_strong(jnp.array([[y0, lens_params[0]]]))[0]:
        multi_image = True
    else:
        multi_image = False

    contour_int = contour_integral(x_im, multi_image,
                                   T_funcs, y, lens_params, critical=crit_run,
                                   )

    if not multi_image:
        if overrode:
            T_vir_low_bound = settings['T_vir_low_bound']
            # FIXME: 100. is hardcoded
            T0_find_max = jnp.linspace(T_vir_low_bound, 100., N)
            iters = 5
        else:
            T0_find_max = jnp.linspace(T_vir*1e-3, T_vir*1.5, N)
            iters = 2
        contour_int.set_T_max(jnp.max(T0_find_max))
        contour_int.find_T_outer(bisect_cond_fun, bisect_step_fun_T_1D)
        T_val_max = contour_int.find_T_for_max_u(T0_find_max, bisection_1D_v, bisect_cond_fun,
                                                 bisect_step_fun_T_1D,
                                                 contour_cond_func, contour_step_func, iters)
        T0_min_out_segs, T0_arr_sad_max = make_T0_arr_multiple_chev(
            N, jnp.array([T_val_max, jnp.nan, 0.]), T0_max)
    else:
        T_val_max = jnp.nan
        T0_min_out_segs, T0_arr_sad_max = make_T0_arr_multiple_chev(
            N, T_images, T0_max)
    contour_int.set_T0_segments(T0_min_out_segs, T0_arr_sad_max)
    contour_int.get_and_set_T_max()
    contour_int.find_T_outer(bisect_cond_fun, bisect_step_fun_T_1D)
    contour_int.make_x_inits(
        bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D)
    contour_int.contour_integrate(contour_cond_func, contour_step_func)

    if (not strong_lensing) or crit_run:
        contour_int.compute_mus()

    if return_all:
        return contour_int, T0_min_out_segs, T0_arr_sad_max, x_im, T_val_max
    else:
        return contour_int


def compute_F(w_interp: jnp.ndarray, y: jnp.ndarray, lens_params: jnp.ndarray, T_funcs: Dict[str, Callable], helper_funcs: Dict[str, Callable], crit_funcs: Dict[str, Callable],
              N: int, T0_max: float, crit_run: bool = False, singular: bool = False, origin: str = 'regular', **kwargs) -> Tuple[jnp.ndarray, List[jnp.ndarray], contour_integral, jnp.ndarray]:
    """
    Compute the amplification factor in the frequency domain.

    Parameters:
        w_interp: The frequencies to compute the amplification factor.
        y: The 2D impact parameter. For now the second component is not used.
        lens_params: The lens model parameter. For now only one parameter is supported.
        T_funcs: A dictionary of functions derived from the Fermat potential needed for computing the amplification factor.
        helper_funcs: A dictionary of helper functions needed for the numerical implementation.
        crit_funcs: A dictionary of functions for computing the caustic curve.
        N: The number of interpolation node in time in each segment.
        T0_max: The maximum value of the time delay to compute the time domain amplification up to.
        crit_run: Whether the run corresponds to a run on the caustic.
        singular: Whether the center of the lens is singular, i.e. a cusp or a pole in the time delay function.
        origin: The origin type of the lens model (cusp, pole or regular).

    Keyword Args:
        N_fft (int): The number of points in the Fast Fourier Transform.
        t_fft_short_max_fac (float): The factor determining the maximum value of the time delay to compute the amplification factor in the short time regime.
        t_fft_long_max_fac (float): The factor determining the maximum value of the time delay to compute the amplification factor in the long time regime.
        w_trans_weak_fac (List[float]): The transition frequencies between different FFT regimes and geometrical optics for the weak lensing regime.
        w_trans_strong_fac (List[float]): The transition frequencies between different FFT regimes and geometrical optics for the strong lensing regime.
        sig_fac (float): The width of the transition region.
        **kwargs: Additional keyword arguments passed to `amplification_computation_for_interpolation`.

    Returns:
        F_interp (jnp.ndarray): The interpolated amplification factor.
        F_interp_raw (List[jnp.ndarray]): A list of the amplification factor from the different FFT regimes, for debugging.
        contour_obj (contour_integral): The contour integral object.
        partitions (jnp.ndarray): The transition frequencies.
    """

    (contour_obj, T0_min_out_segs, T0_arr_sad_max,
     x_im, T_val_max) = amplification_computation_for_interpolation(
        T_funcs,
        helper_funcs,
        crit_funcs,
        y,
        lens_params,
        crit_run=crit_run,
        N=N,
        T0_max=T0_max,
        return_all=True,
        singular=singular,
        origin=origin,
        **kwargs)

    freq_settings = {
        'N_fft': 2**16,
        't_fft_short_max_fac': 20,
        't_fft_long_max_fac': 2000,
        'w_trans_weak_fac': [2.5, 50],
        'w_trans_strong_fac': [2.5, 250],
        'sig_fac': 10,
    }

    freq_settings.update(kwargs)

    N_fft = freq_settings['N_fft']
    t_fft_short_max_fac = freq_settings['t_fft_short_max_fac']
    t_fft_long_max_fac = freq_settings['t_fft_long_max_fac']

    strongly_lensed = jnp.isnan(T_val_max)

    mu = T_funcs['mu']
    T_1D = T_funcs['T_1D']

    y0 = y[0]

    T_im = T_1D(x_im, y0, lens_params)
    T_im -= T_im[2]
    mu_im = mu(x_im, lens_params)

    T0_min_out_full = jnp.concatenate(T0_min_out_segs)
    u_min_out_full = jnp.concatenate(contour_obj.u_min_out)

    # if origin in ['regular', 'im']:
    T_im_hi = jnp.nanmax(jnp.array([T_im[1], T_val_max]))
    # else:
    #     T_im_hi = T_val_max
    t_fft_short_max = T_im_hi*t_fft_short_max_fac
    t_fft_long_max = jnp.min(jnp.array([T_im_hi*t_fft_long_max_fac, T0_max]))
    t_fft_short = jnp.linspace(0, t_fft_short_max, N_fft)
    t_fft_long = jnp.linspace(0, t_fft_long_max, N_fft)

    if strongly_lensed:
        F_fft_short = interp_F_fft_strong_jnp(t_fft_short,
                                              T0_min_out_full,
                                              u_min_out_full,
                                              T0_arr_sad_max,
                                              contour_obj.u_sad_max)
        F_fft_long = interp_F_fft_strong_jnp(t_fft_long,
                                             T0_min_out_full,
                                             u_min_out_full,
                                             T0_arr_sad_max,
                                             contour_obj.u_sad_max)
    else:
        F_fft_short = interp_F_fft_weak_jnp(t_fft_short,
                                            T0_min_out_full,
                                            u_min_out_full)
        F_fft_long = interp_F_fft_weak_jnp(t_fft_long,
                                           T0_min_out_full,
                                           u_min_out_full)

    w_arr_high, Fw_high = amplification_fft_jnp(t_fft_short, F_fft_short)
    w_arr_low, Fw_low = amplification_fft_jnp(t_fft_long, F_fft_long)

    if not strongly_lensed:
        mu_im = mu_im.at[:2].set(jnp.array([0., 0.]))
        T_im = jnp.zeros_like(T_im)

    w_trans_weak_fac = freq_settings['w_trans_weak_fac']
    w_trans_strong_fac = freq_settings['w_trans_strong_fac']
    w_trans_1 = w_trans_strong_fac[0]/T_im_hi if strongly_lensed else w_trans_weak_fac[0]/T_im_hi
    w_trans_2 = w_trans_strong_fac[1]/T_im_hi if strongly_lensed else w_trans_weak_fac[1]/T_im_hi

    w_list = [w_arr_low[1:], w_arr_high]
    F_list = [Fw_low[1:], Fw_high]
    w_low_trans = w_arr_low[1]
    partitions = jnp.array([w_low_trans, w_trans_1, w_trans_2])
    sigs = partitions/freq_settings['sig_fac']

    F_interp, F_interp_raw = interp_partitions_jnp(
        w_interp, w_list, F_list, partitions, sigs, T_im, mu_im, origin=origin)

    return F_interp, F_interp_raw, contour_obj, partitions


def crtical_curve_interpolants(param_arr, T_funcs, crit_curve_helper_funcs, add_y=jnp.array([]), add_x=jnp.array([]),
                               add_param=jnp.array([]), add_indxs=jnp.array([-1])):
    x_crit, y_crit = get_crit_curve_1D(
        param_arr, T_funcs, crit_curve_helper_funcs, x_hi=100.)
    x_crit = jnp.insert(x_crit, add_indxs, add_x)
    y_crit = jnp.insert(y_crit, add_indxs, add_y)
    param_arr = jnp.insert(param_arr, add_indxs, add_param)

    def lens_param_to_x_crit(param): return jnp.interp(
        param, param_arr, -x_crit)
    def lens_param_to_y_crit(param): return jnp.interp(
        param, param_arr, y_crit)

    def x_crit_to_lens_param(x): return jnp.interp(x, -x_crit, param_arr)
    y_crit_ordered = jax.lax.cond(
        y_crit[1] > y_crit[0], lambda y: y, lambda y: jnp.flip(y), y_crit)
    param_arr_ordered = jax.lax.cond(
        y_crit[1] > y_crit[0], lambda y: y, lambda y: jnp.flip(y), param_arr)
    x_crit_ordered = jax.lax.cond(
        y_crit[1] > y_crit[0], lambda y: y, lambda y: jnp.flip(y), -x_crit)

    def y_crit_to_lens_param(y): return jnp.interp(
        y, y_crit_ordered, param_arr_ordered, left=jnp.nan, right=jnp.nan)

    def y_crit_to_x_crit(y): return jnp.interp(
        y, y_crit_ordered, -x_crit_ordered)

    crit_funcs = {'lens_param_to_x_crit': lens_param_to_x_crit,
                  'lens_param_to_y_crit': lens_param_to_y_crit,
                  'x_crit_to_lens_param': x_crit_to_lens_param,
                  'y_crit_to_lens_param': y_crit_to_lens_param,
                  'y_crit_to_x_crit': y_crit_to_x_crit}

    return crit_funcs


def crtical_curve_interpolants_np(param_arr, T_funcs, crit_curve_helper_funcs, add_y=np.array([]), add_x=np.array([]),
                                  add_param=np.array([]), add_indxs=np.array([-1])):
    x_crit, y_crit = get_crit_curve_1D(
        param_arr, T_funcs, crit_curve_helper_funcs, x_hi=100.)
    x_crit = np.insert(x_crit, add_indxs, add_x)
    y_crit = np.insert(y_crit, add_indxs, add_y)
    param_arr = np.insert(param_arr, add_indxs, add_param)
    lens_param_to_x_crit = interp1d(param_arr, -x_crit)
    lens_param_to_y_crit = interp1d(param_arr, y_crit)
    x_crit_to_lens_param = interp1d(-x_crit, param_arr)
    if y_crit[1] > y_crit[0]:
        y_crit_ordered = y_crit
        param_arr_ordered = param_arr
        x_crit_ordered = -x_crit
    else:
        y_crit_ordered = np.flip(y_crit)
        param_arr_ordered = np.flip(param_arr)
        x_crit_ordered = np.flip(-x_crit)
    y_crit_to_lens_param = interp1d(
        y_crit_ordered, param_arr_ordered, fill_value=np.nan, bounds_error=False)
    y_crit_to_x_crit = interp1d(y_crit_ordered, -x_crit_ordered)

    crit_funcs = {'lens_param_to_x_crit': lens_param_to_x_crit,
                  'lens_param_to_y_crit': lens_param_to_y_crit,
                  'x_crit_to_lens_param': x_crit_to_lens_param,
                  'y_crit_to_lens_param': y_crit_to_lens_param,
                  'y_crit_to_x_crit': y_crit_to_x_crit}

    return crit_funcs


def low_w_approximation(w, Psi):
    return 1 - 1.j*w*Psi(jnp.exp(1.j*jnp.pi/4)/jnp.sqrt(w)) - w**2/2*Psi(jnp.exp(1.j*jnp.pi/4)/jnp.sqrt(w))**2
