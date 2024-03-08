from .amplification_factor import *
from .lens_functions import *
from .contour import *
from .utils import *
from .root import *
from .plot import *
from .lens_model import *

from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from tqdm import tqdm
import os, pickle

from scipy.interpolate import LinearNDInterpolator

from typing import List, Tuple, Union, Optional, Dict, Any, Callable

def make_grid_points(settings: Dict[str, Union[str, float, int]], functions_dict: Optional[Dict[str,Dict[str, Callable]]] = None, mid_point: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Callable]:
    """
    Make the interpolation node grid points of the amplification factor.

    Parameters:
        settings: a dictionary containing the settings for the interpolation.
        functions_dict: a dictionary of dictionaries containing relevant functions for computing the amplification factor.
        mid_point: whether to use the mid-points of the grid instead of the grid points themselves. Useful for estimating the error of the interpolation.

    Returns:
        weak_points: the interpolation nodes in the weak-lensing regime.
        strong_points: the interpolation nodes in the strong-lensing regime.
        crit_points_in_bound: the interpolation nodes on the caustic curve.
        crit_T_vir: the time delay of the saddle/maximum image on the caustic curve. The saddle and maximum images merge on the caustic.
        lens_param_to_y_crit: a function that maps the lens parameter to y on the caustic curve.
    """

    lp_low = settings['lp_low']
    lp_high = settings['lp_high']
    crit_lp_N = settings['crit_lp_N']
    y_low = settings['y_low']
    y_high = settings['y_high']
    N_grid = settings['N_grid']
    N_grid_strong = settings['N_grid_strong']
    N_crit = settings['N_crit']
    T0_max = settings['T0_max']
    N = settings['N']
    lens_model_name = settings['lens_model_name']

    im_x_init_low = settings['im_x_init_low']
    im_x_init_high = settings['im_x_init_high']
    im_x_init_num = settings['im_x_init_num']
    im_screen_round_decimal = settings['im_screen_round_decimal']

    im_x_init_arr = jnp.linspace(
    im_x_init_low, 
    im_x_init_high, 
    im_x_init_num)

    param_arr = jnp.linspace(lp_low, lp_high, crit_lp_N)

    lm = get_lens_model(lens_model_name)
    Psi = lm.get_Psi()

    if functions_dict is None:
        T_funcs, helper_funcs = amplification_computation_prep(Psi)
        crit_curve_helper_funcs = make_crit_curve_helper_func(T_funcs)
        crit_funcs = crtical_curve_interpolants(param_arr, T_funcs, crit_curve_helper_funcs)
    else:
        T_funcs = functions_dict['T_funcs']
        helper_funcs = functions_dict['helper_funcs']
        crit_funcs = functions_dict['crit_funcs']
        crit_curve_helper_funcs = functions_dict['crit_curve_helper_funcs']

    irregular_crit_points_dict = lm.get_irregular_crit_points_dict()
    y_crit_override = lm.get_y_crit_override()
    add_to_strong = lm.get_add_to_strong()

    if irregular_crit_points_dict is not None:
        irregular_crit_points = irregular_crit_points_dict['irregular_crit_points']
        irregular_y_crit_points = irregular_crit_points_dict['irregular_y_crit_points']
        irregular_x_crit_points = irregular_crit_points_dict['irregular_x_crit_points']
        irregular_lp_crit_points = irregular_crit_points_dict['irregular_lp_crit_points']

    newt_cond_fun = helper_funcs['newt_cond_fun']
    newt_step_fun = helper_funcs['newt_step_fun']

    lens_param_to_y_crit = crit_funcs['lens_param_to_y_crit']
    y_crit_to_lens_param = crit_funcs['y_crit_to_lens_param']
    y_crit_to_x_crit = crit_funcs['y_crit_to_x_crit']

    lp_arr = jnp.linspace(lp_low, lp_high, num = N_grid)
    y_arr = jnp.linspace(y_low, y_high, num = N_grid)
    lp_arr_strong = jnp.linspace(lp_low, lp_high, num = N_grid_strong)
    y_arr_strong = jnp.linspace(y_low, y_high, num = N_grid_strong)

    if mid_point:
        y_arr = (y_arr[:-1] + y_arr[1:])/2
        lp_arr = (lp_arr[:-1] + lp_arr[1:])/2
        y_arr_strong = (y_arr_strong[:-1] + y_arr_strong[1:])/2
        lp_arr_strong = (lp_arr_strong[:-1] + lp_arr_strong[1:])/2

    grid_points = make_points_arr_mesh(y_arr, lp_arr)
    grid_points_strong = make_points_arr_mesh(y_arr_strong, lp_arr_strong)

    @partial(jnp.vectorize, signature = '(n,2)->(n)')
    def is_strong(point):
        return y_crit_override(lens_param_to_y_crit(point[:,1]), point[:,1]) > point[:,0]

    weak_points = grid_points[~is_strong(grid_points)]
    if add_to_strong is None:
        strong_points = grid_points_strong[is_strong(grid_points_strong)]
    else:
        strong_points = grid_points_strong[is_strong(grid_points_strong) | add_to_strong(grid_points_strong)]

    lp_crit_points = jnp.linspace(lp_low, lp_high, num = N_crit)
    x_crit_points, y_crit_points = get_crit_curve_1D(lp_crit_points, T_funcs, crit_curve_helper_funcs, x_hi = 100.)
    crit_points = jnp.vstack([y_crit_points, lp_crit_points]).T
    
    boundary_crit_points = jnp.array([[y_arr[0], y_crit_to_lens_param(y_arr[0])],
                                  [y_arr[-1], y_crit_to_lens_param(y_arr[-1])]])
    boundary_crit_points = boundary_crit_points[~jnp.isnan(boundary_crit_points).any(axis = 1)]

    if irregular_crit_points_dict is not None:
        crit_points_all = jnp.vstack([boundary_crit_points, crit_points, irregular_crit_points])
        y_crit_points_all = jnp.concatenate((boundary_crit_points[:,0], y_crit_points, irregular_y_crit_points))
        x_crit_points_all = jnp.concatenate((y_crit_to_x_crit(boundary_crit_points[:,0]), -x_crit_points, irregular_x_crit_points))
        lp_crit_points_all = jnp.concatenate((boundary_crit_points[:,1], lp_crit_points, irregular_lp_crit_points))
    else:
        crit_points_all = jnp.vstack([boundary_crit_points, crit_points])
        y_crit_points_all = jnp.append((boundary_crit_points[:,0]), y_crit_points)
        x_crit_points_all = jnp.append((y_crit_to_x_crit(boundary_crit_points[:,0])), -x_crit_points)
        lp_crit_points_all = jnp.append((boundary_crit_points[:,1]), lp_crit_points)

    in_bound = crit_points_all[:, 0] >= y_low

    crit_points_in_bound = crit_points_all[in_bound]
    y_crit_points_in_bound = y_crit_points_all[in_bound]
    x_crit_points_in_bound = x_crit_points_all[in_bound]
    lp_crit_points_in_bound = lp_crit_points_all[in_bound]

    get_crit_points_1D_vec = jnp.vectorize(
        get_crit_points_1D, 
        excluded = {0, 1, 2, 5},
        signature = '(),()->(3)')
    get_crit_points_2D_arr = lambda x: get_crit_points_1D_vec(
        im_x_init_arr, 
        newt_cond_fun, 
        newt_step_fun,  
        x[:, 0], x[:, 1], 
        im_screen_round_decimal)
    
    crit_image_x_newt = get_crit_points_2D_arr(crit_points_in_bound)
    crit_sad_x, crit_max_x, crit_min_x = jnp.hsplit(crit_image_x_newt, crit_image_x_newt.shape[1])

    crit_image_x = jnp.vstack([x_crit_points_in_bound, x_crit_points_in_bound, crit_min_x.ravel()]).T
    T_1D_vec = T_funcs['T_1D_vec']
    crit_T_vir = T_1D_vec(x_crit_points_in_bound, crit_points_in_bound[:,0], crit_points_in_bound[:,1], Psi)
    crit_sad_x, crit_max_x, crit_min_x = jnp.hsplit(crit_image_x, crit_image_x.shape[1])

    return weak_points, strong_points, crit_points_in_bound, crit_T_vir, lens_param_to_y_crit

def interpolate(settings: Dict[str, Union[str, int, float]], save_dir: Optional[str] = None, strong: bool = True, weak: bool = True, interp_crit: bool = True):
    """
    Interpolate the time domain amplification factor and save the interpolation tables.

    Parameters:
        settings: a dictionary containing the settings for the interpolation.
        save_dir: the directory to save the interpolation tables.
        strong: whether to compute and interpolate the strong-lensing points.
        weak: whether to compute and interpolate the weak-lensing points.
        interp_crit: whether to interpolate the caustic curve.
    """
    print('''
          
########################################################          
#                                                      #
#   Interpolate the time domain amplification factor   #
#                                                      #
########################################################    
          
          ''')
          
    if strong and weak:
        print('Both strong and weak-lensing points will be computed.')
    elif strong:
        print('Only strong-lensing points will be computed.')
    elif weak:
        print('Only weak-lensing points will be computed.')
    else:
        raise ValueError('At least one of strong and weak must be True.')
    
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'interpolation')

    lens_model_name = settings['lens_model_name']

    lp_low = settings['lp_low']
    lp_high = settings['lp_high']
    crit_lp_N = settings['crit_lp_N']
    y_low = settings['y_low']
    y_high = settings['y_high']
    N_grid = settings['N_grid']
    N_grid_strong = settings['N_grid_strong']
    N_crit = settings['N_crit']
    T0_max = settings['T0_max']
    N = settings['N']
    lp_name = settings['lens_param_name']

    im_x_init_low = settings['im_x_init_low']
    im_x_init_high = settings['im_x_init_high']
    im_x_init_num = settings['im_x_init_num']
    im_screen_round_decimal = settings['im_screen_round_decimal']

    im_x_init_arr = jnp.linspace(
    im_x_init_low, 
    im_x_init_high, 
    im_x_init_num)

    print('Initializing...')

    lm = get_lens_model(lens_model_name)
    Psi = lm.get_Psi()

    T_funcs, helper_funcs = amplification_computation_prep(Psi)
    crit_curve_helper_funcs = make_crit_curve_helper_func(T_funcs)

    newt_cond_fun = helper_funcs['newt_cond_fun']
    newt_step_fun = helper_funcs['newt_step_fun']

    param_arr = jnp.linspace(lp_low, lp_high, crit_lp_N)
    crit_funcs = crtical_curve_interpolants(param_arr, T_funcs, crit_curve_helper_funcs)
    if interp_crit:
        crit_funcs_np = crtical_curve_interpolants_np(param_arr, T_funcs, crit_curve_helper_funcs)
        os.makedirs(os.path.join(save_dir, 'crit_funcs'), exist_ok = True)
        with open(os.path.join(save_dir, f'crit_funcs/{lens_model_name}_crit_funcs_{lp_low:.3f}_{lp_high:.3f}.pkl'), 'wb') as f:
            pickle.dump(crit_funcs_np, f)

    lens_param_to_y_crit = crit_funcs['lens_param_to_y_crit']
    y_crit_to_lens_param = crit_funcs['y_crit_to_lens_param']
    y_crit_to_x_crit = crit_funcs['y_crit_to_x_crit']

    override_funcs_dict = lm.get_override_funcs_dict(T_funcs['dT_1D'])

    functions_dict = { 'T_funcs': T_funcs,
                       'helper_funcs': helper_funcs,
                       'crit_funcs': crit_funcs,
                       'crit_curve_helper_funcs': crit_curve_helper_funcs}

    grid_points_out = make_grid_points(settings,
                                       functions_dict = functions_dict)
    (weak_points, strong_points, crit_points_in_bound, 
     crit_T_vir, lens_param_to_y_crit) = grid_points_out 
    
    if strong:
        u_strong_min_out_list = []
        u_strong_sad_max_list = []

        for y0, kappa in tqdm(strong_points, desc = 'Computing strong-lensing integrals'):
            y =jnp.array([y0, 0.])
            lens_params = jnp.array([kappa])
            contour_int = amplification_computation_for_interpolation(
                        T_funcs, 
                        helper_funcs, 
                        crit_funcs, 
                        y, 
                        lens_params, 
                        crit_run = False,
                        N = N,
                        T0_max = T0_max,
                        **override_funcs_dict)
            u_strong_min_out_list.append(contour_int.u_min_out)
            u_strong_sad_max_list.append(contour_int.u_sad_max)

    u_crit_min_out_list = []
    u_crit_sad_max_list = []
    mu_min_crit_list = []    

    for y0, kappa in tqdm(crit_points_in_bound, desc = 'Computing critical-curve integrals'):
        y =jnp.array([y0, 0.])
        lens_params = jnp.array([kappa])
        contour_int = amplification_computation_for_interpolation(
                    T_funcs, 
                    helper_funcs, 
                    crit_funcs, 
                    y, 
                    lens_params, 
                    crit_run = True,
                    N = N,
                    T0_max = T0_max,
                    **override_funcs_dict)
        u_crit_min_out_list.append(contour_int.u_min_out)
        u_crit_sad_max_list.append(contour_int.u_sad_max)
        mu_min_crit_list.append(contour_int.mu_min)

    interpolate_dir_name = f'{lens_model_name}_amp_y_{y_low:.3f}_{y_high:.3f}_{lp_name}_{lp_low:.3f}_{lp_high:.3f}_N_grid_{N_grid}_N_grid_strong_{N_grid_strong}_N_crit_{N_crit}_N_{N}'
    interpolate_dir_path = os.path.join(save_dir, interpolate_dir_name)
    os.makedirs(interpolate_dir_path, exist_ok = True)
    
    if strong:
        print('Interpolating and saving strong-lensing points...')

        u_strong_full_min_out = jnp.array(u_strong_min_out_list + u_crit_min_out_list)
        u_strong_full_sad_max = jnp.array(u_strong_sad_max_list + u_crit_sad_max_list)
        strong_full_points = jnp.vstack([strong_points, crit_points_in_bound])

        u_points_low = u_strong_full_min_out[:,0,:]
        u_points_mid_1 = u_strong_full_min_out[:,1,:]
        u_points_mid_2 = u_strong_full_min_out[:,2,:]
        u_points_high = u_strong_full_min_out[:,3,:]

        interp_low = LinearNDInterpolator(strong_full_points, u_points_low)
        interp_mid_1 = LinearNDInterpolator(strong_full_points, u_points_mid_1)
        interp_mid_2 = LinearNDInterpolator(strong_full_points, u_points_mid_2)
        interp_high = LinearNDInterpolator(strong_full_points, u_points_high)
        interp_sad_max = LinearNDInterpolator(strong_full_points, u_strong_full_sad_max)

        with open(os.path.join(interpolate_dir_path,'interp_strong_low.pkl'), 'wb') as f:
            pickle.dump(interp_low, f)

        with open(os.path.join(interpolate_dir_path,'interp_strong_mid_1.pkl'), 'wb') as f:
            pickle.dump(interp_mid_1, f)

        with open(os.path.join(interpolate_dir_path,'interp_strong_mid_2.pkl'), 'wb') as f:
            pickle.dump(interp_mid_2, f)

        with open(os.path.join(interpolate_dir_path,'interp_strong_high.pkl'), 'wb') as f:
            pickle.dump(interp_high, f)

        with open(os.path.join(interpolate_dir_path,'interp_strong_sad_max.pkl'), 'wb') as f:
            pickle.dump(interp_sad_max, f)

    if weak:
        u_weak_min_out_list = []
        T_vir_list = []
        mu_min_weak_list = []

        override_funcs_dict_weak = override_funcs_dict.copy()
        override_funcs_dict_weak['add_to_strong'] = add_to_strong_default

        for y0, kappa in tqdm(weak_points, desc = 'Computing weak-lensing integrals'):
            y =jnp.array([y0, 0.])
            lens_params = jnp.array([kappa])
            contour_int = amplification_computation_for_interpolation(
                        T_funcs, 
                        helper_funcs, 
                        crit_funcs, 
                        y, 
                        lens_params, 
                        crit_run = False,
                        N = N,
                        T0_max = T0_max,
                        **override_funcs_dict_weak)
            u_weak_min_out_list.append(contour_int.u_min_out)
            T_vir_list.append(contour_int.T_vir)
            mu_min_weak_list.append(contour_int.mu_min)

        print('Interpolating and saving weak-lensing points...')

        u_weak_full_min_out = jnp.array(u_weak_min_out_list + u_crit_min_out_list)
        weak_full_points = jnp.vstack([weak_points, crit_points_in_bound])
        T_vir_full = jnp.array(T_vir_list + crit_T_vir.tolist())
        mu_min_weak_full = jnp.array(mu_min_weak_list + mu_min_crit_list)

        u_weak_points_low = u_weak_full_min_out[:,0,:]
        u_weak_points_mid_1 = u_weak_full_min_out[:,1,:]
        u_weak_points_mid_2 = u_weak_full_min_out[:,2,:]
        u_weak_points_high = u_weak_full_min_out[:,3,:]

        interp_weak_low = LinearNDInterpolator(weak_full_points, u_weak_points_low)
        interp_weak_mid_1 = LinearNDInterpolator(weak_full_points, u_weak_points_mid_1)
        interp_weak_mid_2 = LinearNDInterpolator(weak_full_points, u_weak_points_mid_2)
        interp_weak_high = LinearNDInterpolator(weak_full_points, u_weak_points_high)
        interp_T_vir = LinearNDInterpolator(weak_full_points, T_vir_full)
        interp_mu_min_weak = LinearNDInterpolator(weak_full_points, mu_min_weak_full)

        with open(os.path.join(interpolate_dir_path,'interp_weak_low.pkl'), 'wb') as f:
            pickle.dump(interp_weak_low, f)

        with open(os.path.join(interpolate_dir_path,'interp_weak_mid_1.pkl'), 'wb') as f:
            pickle.dump(interp_weak_mid_1, f)

        with open(os.path.join(interpolate_dir_path,'interp_weak_mid_2.pkl'), 'wb') as f:
            pickle.dump(interp_weak_mid_2, f)

        with open(os.path.join(interpolate_dir_path,'interp_weak_high.pkl'), 'wb') as f:
            pickle.dump(interp_weak_high, f)

        with open(os.path.join(interpolate_dir_path,'interp_T_vir.pkl'), 'wb') as f:
            pickle.dump(interp_T_vir, f)

        with open(os.path.join(interpolate_dir_path,'interp_mu_min_weak.pkl'), 'wb') as f:
            pickle.dump(interp_mu_min_weak, f)

        print(f'Interpolants saved to {interpolate_dir_path}')

    print('Done!')


def interpolate_im(settings: Dict[str, Union[str, float, int]], save_dir: Optional[str] = None):
    """
    Interpolate the time delay and magnification of images.

    Parameters:
        settings: a dictionary containing the settings for the interpolation.
        save_dir: the directory to save the interpolation tables.
    """

    print('''
          
##############################################################          
#                                                            #
#   Interpolate the time delay and magnification of images   #
#                                                            #
##############################################################    
          
          ''')

    lens_model_name = settings['lens_model_name']

    lp_low = settings['lp_low']
    lp_high = settings['lp_high']
    crit_lp_N = settings['crit_lp_N']
    y_low = settings['y_low']
    y_high = settings['y_high']
    N_grid = settings['N_grid']
    N_grid_strong = settings['N_grid_strong']
    N_crit = settings['N_crit']
    T0_max = settings['T0_max']
    N = settings['N']
    lp_name = settings['lens_param_name']

    im_x_init_low = settings['im_x_init_low']
    im_x_init_high = settings['im_x_init_high']
    im_x_init_num = settings['im_x_init_num']
    im_screen_round_decimal = settings['im_screen_round_decimal']

    lp_low_im = settings['lp_low_im']
    lp_high_im = settings['lp_high_im']
    crit_lp_N_im = settings['crit_lp_N_im']
    newt_max_iter_im = settings['newt_max_iter_im']
    N_grid_im = settings['N_grid_im']
    N_crit_im = settings['N_crit_im']

    im_x_init_arr = jnp.linspace(
    im_x_init_low, 
    im_x_init_high, 
    im_x_init_num)

    print('Initializing...')

    lm = get_lens_model(lens_model_name)
    Psi = lm.get_Psi()

    T_funcs, helper_funcs = amplification_computation_prep(
                                                Psi, 
                                                newt_max_iter = newt_max_iter_im
                                                )
    crit_curve_helper_funcs = make_crit_curve_helper_func(T_funcs)

    newt_cond_fun = helper_funcs['newt_cond_fun']
    newt_step_fun = helper_funcs['newt_step_fun']

    param_arr = jnp.linspace(lp_low_im, lp_high_im, crit_lp_N_im)
    
    Psi_1D = T_funcs['Psi_1D']
    mu_vec = T_funcs['mu_vec']

    irregular_crit_points_dict = lm.get_irregular_crit_points_dict()
    if irregular_crit_points_dict is None:
        add_y = jnp.array([])
        add_x = jnp.array([])
        add_lp = jnp.array([])
        add_indxs = jnp.array([-1])
    else:
        add_y = irregular_crit_points_dict['irregular_y_crit_points']
        add_x = irregular_crit_points_dict['irregular_x_crit_points']
        add_lp = irregular_crit_points_dict['irregular_lp_crit_points']
        if len(add_lp) > 1:
            raise ValueError('Only one irregular critical point is supported')
        add_indxs = jnp.where(jnp.abs(param_arr - add_lp[0]) < 1e-15)[0]
        if len(add_indxs) == 0:
            add_indxs = jnp.where(param_arr > add_lp[0])[0]

    print('Computing critical curves...')
    crit_funcs = crtical_curve_interpolants(param_arr, T_funcs, crit_curve_helper_funcs, add_y = add_y, 
                                        add_x = add_x, add_param = add_lp, add_indxs = add_indxs)
    
    lens_param_to_y_crit = crit_funcs['lens_param_to_y_crit']
    y_crit_to_lens_param = crit_funcs['y_crit_to_lens_param']
    y_crit_to_x_crit = crit_funcs['y_crit_to_x_crit']
    
    lp_arr = jnp.linspace(lp_low, lp_high, num = N_grid_im)
    y_arr = jnp.linspace(y_low, y_high, num = N_grid_im)

    grid_points = make_points_arr_mesh(y_arr, lp_arr)

    y_crit_override = lm.get_y_crit_override()
    x_im_nan_sub = lm.get_x_im_nan_sub(T_funcs['dT_1D'])
    add_to_strong = lm.get_add_to_strong()

    @partial(jnp.vectorize, signature = '(n,2)->(n)')
    def is_strong(point):
        return y_crit_override(lens_param_to_y_crit(point[:,1]), point[:,1]) > point[:,0]
    
    weak_points = grid_points[~is_strong(grid_points)]
    if add_to_strong is None:
        strong_points = grid_points[is_strong(grid_points)]
    else:
        strong_points = grid_points[is_strong(grid_points) | add_to_strong(grid_points)]

    lp_crit_points = jnp.linspace(lp_low, lp_high, num = N_crit_im)
    x_crit_points, y_crit_points = get_crit_curve_1D(lp_crit_points, T_funcs, crit_curve_helper_funcs, x_hi = 100.)
    if irregular_crit_points_dict is not None:
        lp_crit_points = jnp.concatenate((lp_crit_points, add_lp))
        y_crit_points = jnp.concatenate((y_crit_points, add_y))
        x_crit_points = jnp.concatenate((x_crit_points, add_x))
    crit_points = jnp.vstack([y_crit_points, lp_crit_points]).T
    y_crit_final_2 = jnp.sort(y_crit_points)[-2:]
    y_crit_adds = y_arr[(y_arr > y_crit_final_2[0]) & (y_arr < y_crit_final_2[1])]
    y_crit_points_overriden = y_crit_override(y_crit_points, lp_crit_points)
    boundary_crit_points = jnp.concatenate((jnp.array([[y_arr[0], y_crit_to_lens_param(y_arr[0])],
                                  [y_arr[-1], y_crit_to_lens_param(y_arr[-1])]]),
                                  jnp.array([y_crit_adds, y_crit_to_lens_param(y_crit_adds)]).T,
                                  ))
    boundary_crit_points = boundary_crit_points[~jnp.isnan(boundary_crit_points).any(axis = 1)]

    strong_points_aug = make_points_arr_mesh(jnp.linspace(y_low, 1., N_crit), jnp.array([1.]))
    weak_points_aug = make_points_arr_mesh(jnp.linspace(1., y_high, N_crit), jnp.array([1.]))

    crit_points_all = jnp.vstack([boundary_crit_points, crit_points, ])
    y_crit_points_all = jnp.concatenate((boundary_crit_points[:,0], y_crit_points, ))
    x_crit_points_all = jnp.concatenate((y_crit_to_x_crit(boundary_crit_points[:,0]), -x_crit_points, ))
    kappa_crit_points_all = jnp.concatenate((boundary_crit_points[:,1], lp_crit_points, ))

    crit_points_all = crit_points_all.at[2:, 0].set(crit_points_all[2:, 0] + 1e-15)
    # crit_points_all = crit_points_all.at[2:, 1].set(crit_points_all[2:, 1] - 1e-15)

    in_bound = (lp_high >= crit_points_all[:, 1]) & (crit_points_all[:, 1] >= lp_low) & (y_high >= crit_points_all[:, 0]) & (crit_points_all[:,0] >= y_low)

    crit_points_in_bound = crit_points_all[in_bound]
    y_crit_points_in_bound = y_crit_points_all[in_bound]
    x_crit_points_in_bound = x_crit_points_all[in_bound]
    kappa_crit_points_in_bound = kappa_crit_points_all[in_bound]

    im_x_init_arr = jnp.linspace(
    im_x_init_low, 
    im_x_init_high, 
    im_x_init_num)
    im_x_init_arr = jnp.concatenate((im_x_init_arr, jnp.linspace(-0.1, 0.1, 100)))

    get_crit_points_1D_vec = jnp.vectorize(
        get_crit_points_1D, 
        excluded = {0, 1, 2, 5},
        signature = '(),()->(3)')
    get_crit_points_2D_arr = lambda x: get_crit_points_1D_vec(
        im_x_init_arr, 
        newt_cond_fun, 
        newt_step_fun,  
        x[:, 0], x[:, 1], 
        im_screen_round_decimal)
    
    strong_image_x = get_crit_points_2D_arr(strong_points)
    strong_image_x = x_im_nan_sub(strong_image_x, strong_points[:,0], strong_points[:,1])
    strong_sad_x, strong_max_x, strong_min_x = jnp.hsplit(strong_image_x, strong_image_x.shape[1])

    weak_image_x = get_crit_points_2D_arr(weak_points)
    weak_image_x = x_im_nan_sub(weak_image_x, weak_points[:,0], weak_points[:,1])
    weak_sad_x, weak_max_x, weak_min_x = jnp.hsplit(weak_image_x, weak_image_x.shape[1])

    crit_image_x_newt = get_crit_points_2D_arr(crit_points_in_bound)
    crit_image_x_newt = x_im_nan_sub(crit_image_x_newt, crit_points_in_bound[:,0], crit_points_in_bound[:,1])
    crit_sad_x, crit_max_x, crit_min_x = jnp.hsplit(crit_image_x_newt, crit_image_x_newt.shape[1])

    crit_image_x = jnp.vstack([x_crit_points_in_bound, x_crit_points_in_bound, crit_min_x.ravel()]).T

    crit_sad_x, crit_max_x, crit_min_x = jnp.hsplit(crit_image_x, crit_image_x.shape[1])

    @partial(jnp.vectorize, excluded = {3}, signature = '(),(),()->()')
    @partial(jit, static_argnums = (3))
    def T_1D_vec(x, y, lens_params, Psi_1D):
        lens_params = jnp.atleast_1d(lens_params)
        return jnp.linalg.norm(x - y)**2/2 - Psi_1D(x, lens_params)
    
    strong_points_full = jnp.vstack([strong_points, crit_points_in_bound])

    strong_full_sad_x = jnp.concatenate([strong_sad_x, crit_sad_x]).T[0]
    strong_full_max_x = jnp.concatenate([strong_max_x, crit_max_x]).T[0]
    strong_full_min_x = jnp.concatenate([strong_min_x, crit_min_x]).T[0]

    print('Computing time delays...')

    strong_full_sad_T = T_1D_vec(strong_full_sad_x, strong_points_full[:,0], strong_points_full[:,1], Psi_1D)
    strong_full_max_T = T_1D_vec(nan_to_const(strong_full_max_x, const = 0), strong_points_full[:,0], strong_points_full[:,1], Psi_1D)
    strong_full_min_T = T_1D_vec(strong_full_min_x, strong_points_full[:,0], strong_points_full[:,1], Psi_1D)

    weak_points_for_T_adj_nan = weak_points[~add_to_strong(weak_points)]
    weak_points_nan = jnp.ones(len(weak_points_for_T_adj_nan))*jnp.nan

    interp_strong_full_sad_T_adj = LinearNDInterpolator(
        jnp.vstack([strong_points_full, 
                    weak_points_for_T_adj_nan
                    ]),
        jnp.concatenate((strong_full_sad_T - strong_full_min_T, 
                        weak_points_nan
                        )))
    interp_strong_full_max_T_adj = LinearNDInterpolator(strong_points_full, strong_full_max_T - strong_full_min_T)

    interp_name = f'{lens_model_name}_im_y_{y_low:.3f}_{y_high:.3f}_{lp_name}_{lp_low:.3f}_{lp_high:.3f}_N_{N_grid_im}_N_crit_{N_crit_im}'
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'interpolation')
    os.makedirs(os.path.join(save_dir, interp_name), exist_ok = True)

    with open(os.path.join(save_dir, interp_name, 'strong_full_sad_T_adj.pkl'), 'wb') as f:
        pickle.dump(interp_strong_full_sad_T_adj, f)

    with open(os.path.join(save_dir, interp_name, 'strong_full_max_T_adj.pkl'), 'wb') as f:
        pickle.dump(interp_strong_full_max_T_adj, f)

    print('Computing magnifications...')

    strong_full_sad_mu = jnp.abs(mu_vec(strong_full_sad_x, strong_points_full[:,1]))
    strong_full_max_mu = jnp.abs(mu_vec(strong_full_max_x, strong_points_full[:,1]))
    strong_full_min_mu = jnp.abs(mu_vec(strong_full_min_x, strong_points_full[:,1]))

    interp_strong_full_sad_mu = LinearNDInterpolator(strong_points_full, strong_full_sad_mu)
    interp_strong_full_max_mu = LinearNDInterpolator(strong_points_full, strong_full_max_mu)
    interp_strong_full_min_mu = LinearNDInterpolator(strong_points_full, strong_full_min_mu)

    with open(os.path.join(save_dir, interp_name, 'strong_full_sad_mu.pkl'), 'wb') as f:
        pickle.dump(interp_strong_full_sad_mu, f)

    with open(os.path.join(save_dir, interp_name, 'strong_full_max_mu.pkl'), 'wb') as f:
        pickle.dump(interp_strong_full_max_mu, f)

    with open(os.path.join(save_dir, interp_name, 'strong_full_min_mu.pkl'), 'wb') as f:
        pickle.dump(interp_strong_full_min_mu, f)

    print(f'Saved interpolants to {os.path.join(save_dir, interp_name)}.\n Done!')




        


    