import numpy as np
import pickle
import os

def get_interp_dir_name(settings):

    y_low = settings['y_low']
    y_high = settings['y_high']
    lens_param_name = settings['lens_param_name']
    lp_low = settings['lp_low']
    lp_high = settings['lp_high']
    N_grid = settings['N_grid']
    N_grid_strong = settings['N_grid_strong']
    N_crit = settings['N_crit']
    N = settings['N']
    lens_model_name = settings['lens_model_name']
    y_low_im = settings['y_low_im']
    y_high_im = settings['y_high_im']
    lp_low_im = settings['lp_low_im']
    lp_high_im = settings['lp_high_im']
    N_grid_im = settings['N_grid_im']
    N_crit_im = settings['N_crit_im']

    interpolate_dir_name = f'{lens_model_name}_y_{y_low:.3f}_{y_high:.3f}_{lens_param_name}_{lp_low:.3f}_{lp_high:.3f}_N_grid_{N_grid}_N_grid_strong_{N_grid_strong}_N_crit_{N_crit}_N_{N}'
    image_interp_dir_name = f'{lens_model_name}_y_{y_low_im:.3f}_{y_high_im:.3f}_{lens_param_name}_{lp_low_im:.3f}_{lp_high_im:.3f}_N_{N_grid_im}_N_crit_{N_crit_im}'

    return interpolate_dir_name, image_interp_dir_name

def load_interpolators(interpolation_root_dir, **kwargs):

    y_low = kwargs['y_low']
    y_high = kwargs['y_high']
    lens_param_name = kwargs['lens_param_name']
    lp_low = kwargs['lp_low']
    lp_high = kwargs['lp_high']
    N_grid = kwargs['N_grid']
    N_grid_strong = kwargs['N_grid_strong']
    N_crit = kwargs['N_crit']
    N = kwargs['N']
    lens_model_name = kwargs['lens_model_name']

    T0_max = kwargs['T0_max']

    y_low_im = kwargs['y_low_im']
    y_high_im = kwargs['y_high_im']
    lp_low_im = kwargs['lp_low_im']
    lp_high_im = kwargs['lp_high_im']
    N_grid_im = kwargs['N_grid_im']
    N_crit_im = kwargs['N_crit_im']

    interpolate_dir_name, image_interp_dir_name = get_interp_dir_name(kwargs)

    interpolate_dir = os.path.join(interpolation_root_dir, interpolate_dir_name)
    image_interp_dir = os.path.join(interpolation_root_dir, image_interp_dir_name)

    with open(os.path.join(interpolate_dir, 'interp_strong_low.pkl'), 'rb') as f:
        interp_strong_low = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_strong_mid_1.pkl'), 'rb') as f:
        interp_strong_mid_1 = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_strong_mid_2.pkl'), 'rb') as f:
        interp_strong_mid_2 = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_strong_high.pkl'), 'rb') as f:
        interp_strong_high = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_strong_sad_max.pkl'), 'rb') as f:
        interp_strong_sad_max = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_weak_low.pkl'), 'rb') as f:
        interp_weak_low = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_weak_mid_1.pkl'), 'rb') as f:
        interp_weak_mid_1 = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_weak_mid_2.pkl'), 'rb') as f:
        interp_weak_mid_2 = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_weak_high.pkl'), 'rb') as f:
        interp_weak_high = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_T_vir.pkl'), 'rb') as f:
        interp_T_vir = pickle.load(f)

    with open(os.path.join(interpolate_dir, 'interp_mu_min_weak.pkl'), 'rb') as f:
        interp_weak_full_min_mu = pickle.load(f)

    with open(os.path.join(image_interp_dir, 'strong_full_sad_T_adj.pkl'), 'rb') as f:
        interp_strong_full_sad_T_adj = pickle.load(f)

    with open(os.path.join(image_interp_dir, 'strong_full_max_T_adj.pkl'), 'rb') as f:
        interp_strong_full_max_T_adj = pickle.load(f)  

    with open(os.path.join(image_interp_dir, 'strong_full_sad_mu.pkl'), 'rb') as f:
        interp_strong_full_sad_mu = pickle.load(f)

    with open(os.path.join(image_interp_dir, 'strong_full_max_mu.pkl'), 'rb') as f:
        interp_strong_full_max_mu = pickle.load(f)

    with open(os.path.join(image_interp_dir, 'strong_full_min_mu.pkl'), 'rb') as f:
        interp_strong_full_min_mu = pickle.load(f)

    interpolators = {
        'interp_strong_low': interp_strong_low,
        'interp_strong_mid_1': interp_strong_mid_1,
        'interp_strong_mid_2': interp_strong_mid_2,
        'interp_strong_high': interp_strong_high,
        'interp_strong_sad_max': interp_strong_sad_max,
        'interp_weak_low': interp_weak_low,
        'interp_weak_mid_1': interp_weak_mid_1,
        'interp_weak_mid_2': interp_weak_mid_2,
        'interp_weak_high': interp_weak_high,
        'interp_T_vir': interp_T_vir,
        'interp_weak_full_min_mu': interp_weak_full_min_mu,
        'interp_strong_full_sad_T_adj': interp_strong_full_sad_T_adj,
        'interp_strong_full_max_T_adj': interp_strong_full_max_T_adj,
        'interp_strong_full_sad_mu': interp_strong_full_sad_mu,
        'interp_strong_full_max_mu': interp_strong_full_max_mu,
        'interp_strong_full_min_mu': interp_strong_full_min_mu
    }

    return interpolators

def chev_points_np(a, b, n):
    chev = -np.cos(np.pi*(np.arange(n)+0.5)/n)
    chev_inner_width = chev[-1]-chev[0]
    return (a+b)/2 + (b-a)/chev_inner_width * chev

def chev_first_half_np(a, b, n):
    chev = -np.cos(np.pi*(np.arange(n)+0.5)/n)
    chev_inner_width = chev[-1]-chev[0]
    reg = np.linspace(-1, 1, n)
    chev_half = np.where(reg < 0, 2*chev/chev_inner_width, reg)
    return (a+b)/2 + (b-a)/2 * chev_half

def make_T0_arr_multiple_chev_np(N, T_images, T0_max):
    T_im_sad = T_images[0]
    T_im_max = T_images[1]
    dt_around_image = T_im_sad / N**3
    T0_arr_low = chev_points_np(dt_around_image,
                              T_im_sad - dt_around_image,
                              N)
    T0_arr_mid_1 = chev_first_half_np(T_im_sad + dt_around_image,
                              2*T_im_sad,
                              N)
    T0_arr_mid_2 = np.linspace(2*T_im_sad + dt_around_image,
                              10*T_im_sad,
                              N)
    T0_arr_high = np.logspace(np.log10(10*T_im_sad + dt_around_image),
                               np.log10(T0_max),
                               N)
    T0_arr_sad_max = chev_points_np(T_im_sad + dt_around_image,
                                  T_im_max - dt_around_image,
                                  N)
    return np.array([T0_arr_low, T0_arr_mid_1,
                            T0_arr_mid_2, T0_arr_high]), T0_arr_sad_max

def make_T0_arr_multiple_np(N, T_images, T0_max):
    T_im_sad = T_images[0]
    T_im_max = T_images[1]
    dt_around_image = T_im_sad / N**3
    T0_arr_low = np.linspace(dt_around_image,
                              T_im_sad - dt_around_image,
                              N)
    T0_arr_mid_1 = np.linspace(T_im_sad + dt_around_image,
                              2*T_im_sad,
                              N)
    T0_arr_mid_2 = np.linspace(2*T_im_sad + dt_around_image,
                              10*T_im_sad,
                              N)
    T0_arr_high = np.logspace(np.log10(10*T_im_sad + dt_around_image),
                               np.log10(T0_max),
                               N)
    T0_arr_sad_max = np.linspace(T_im_sad + dt_around_image,
                                  T_im_max - dt_around_image,
                                  N)
    return np.array([T0_arr_low, T0_arr_mid_1,
                            T0_arr_mid_2, T0_arr_high]), T0_arr_sad_max

def interp_F_fft_strong(t_fft, T0_min_out_interp_full, u_min_out_interp_full, 
                 T0_sad_max_interp, u_interp_sad_max):
    F_fft = np.interp(t_fft, T0_min_out_interp_full, u_min_out_interp_full)
    F_fft += np.interp(t_fft, T0_sad_max_interp, u_interp_sad_max, left = 0., right = 0.)
    return F_fft

def interp_F_fft_weak(t_fft, T0_min_out_interp_full, u_min_out_interp_full):
    F_fft = np.interp(t_fft, T0_min_out_interp_full, u_min_out_interp_full)
    return F_fft

def amplification_fft_np(t_fft, Ft_fft):

    fft_len = len(t_fft)
    dt = t_fft[1] - t_fft[0]
    w_arr = np.linspace(0, 2*np.pi/dt, num = fft_len)
    Fw_raw = w_arr*np.fft.fft(Ft_fft)*dt
    Fw = -np.imag(Fw_raw) - 1.j*np.real(Fw_raw) + Ft_fft[-1]

    return w_arr, Fw

def F_geom(ws, T_im, mu_im):
    F = np.zeros(len(ws), dtype = np.complex128)
    morse_indx = [0.5, 1, 0]
    mu_im = np.nan_to_num(mu_im)
    for i in range(3):
        F += np.sqrt(np.abs(mu_im[i]))*np.exp(1.j*(ws*T_im[i] - np.pi*morse_indx[i]))
    return F

def smooth_increase(x, x0, a):
    return (1 + np.tanh((x - x0)/a))/2

def smooth_decrease(x, x0, a):
    return 1 - smooth_increase(x, x0, a)

def interp_partitions(w_interp, ws, Fs, partitions, sigs, T_im, mu_im, return_geom = False):
    F_interp = np.zeros_like(w_interp, dtype = np.complex128)
    F_interp_raw = [np.ones_like(w_interp, dtype=np.complex128)]
    for i, (w, F) in enumerate(zip(ws, Fs)):
        F_interp_raw.append(np.interp(w_interp, w, F, left = 1., right = 0.))
    F_geometric = F_geom(w_interp, T_im, mu_im)
    F_interp_raw.append(F_geometric)
    for i in range(len(partitions)):
        if i == 0:
            F_interp += F_interp_raw[i]*smooth_decrease(w_interp, partitions[i], sigs[i])
        if i < len(partitions) - 1:
            F_interp += F_interp_raw[i+1]*smooth_increase(w_interp, partitions[i], sigs[i])\
                        *smooth_decrease(w_interp, partitions[i+1], sigs[i+1])
        else:
            F_interp += F_interp_raw[i+1]*smooth_increase(w_interp, partitions[i], sigs[i])
    if return_geom:
        return F_interp, F_geometric
    return F_interp

def strong_lens_cond_override_default(strongly_lensed, y_interp, kappa_interp):
    return strongly_lensed

def F_interp(w_interp, y_interp, kappa_interp, interpolators, settings, return_geom = False,
             strong_lens_cond_override = strong_lens_cond_override_default):

    N = settings['N']
    T0_max = settings['T0_max']

    interp_strong_low = interpolators['interp_strong_low']
    interp_strong_mid_1 = interpolators['interp_strong_mid_1']
    interp_strong_mid_2 = interpolators['interp_strong_mid_2']
    interp_strong_high = interpolators['interp_strong_high']
    interp_strong_sad_max = interpolators['interp_strong_sad_max']
    interp_weak_low = interpolators['interp_weak_low']
    interp_weak_mid_1 = interpolators['interp_weak_mid_1']
    interp_weak_mid_2 = interpolators['interp_weak_mid_2']
    interp_weak_high = interpolators['interp_weak_high']
    interp_T_vir = interpolators['interp_T_vir']
    interp_weak_full_min_mu = interpolators['interp_weak_full_min_mu']
    interp_strong_full_sad_T_adj = interpolators['interp_strong_full_sad_T_adj']
    interp_strong_full_max_T_adj = interpolators['interp_strong_full_max_T_adj']
    interp_strong_full_sad_mu = interpolators['interp_strong_full_sad_mu']
    interp_strong_full_max_mu = interpolators['interp_strong_full_max_mu']
    interp_strong_full_min_mu = interpolators['interp_strong_full_min_mu']

    T_sad = interp_strong_full_sad_T_adj(y_interp, kappa_interp)
    strongly_lensed = ~np.isnan(T_sad)
    strongly_lensed = strong_lens_cond_override(strongly_lensed, y_interp, kappa_interp)

    if strongly_lensed:
        u_interp_low = interp_strong_low(y_interp, kappa_interp).ravel()
        u_interp_mid_1 = interp_strong_mid_1(y_interp, kappa_interp).ravel()
        u_interp_mid_2 = interp_strong_mid_2(y_interp, kappa_interp).ravel()
        u_interp_high = interp_strong_high(y_interp, kappa_interp).ravel()
        u_interp_sad_max = interp_strong_sad_max(y_interp, kappa_interp).ravel()
        T_sad_interp = interp_strong_full_sad_T_adj(y_interp, kappa_interp)
        T_max_interp = interp_strong_full_max_T_adj(y_interp, kappa_interp)
        mu_sad_interp = interp_strong_full_sad_mu(y_interp, kappa_interp)
        mu_max_interp = interp_strong_full_max_mu(y_interp, kappa_interp)
        mu_min_interp = interp_strong_full_min_mu(y_interp, kappa_interp)
    else:
        u_interp_low = interp_weak_low(y_interp, kappa_interp).ravel()
        u_interp_mid_1 = interp_weak_mid_1(y_interp, kappa_interp).ravel()
        u_interp_mid_2 = interp_weak_mid_2(y_interp, kappa_interp).ravel()
        u_interp_high = interp_weak_high(y_interp, kappa_interp).ravel()
        u_interp_sad_max = None
        T_sad_interp = interp_T_vir(y_interp, kappa_interp)
        T_max_interp = np.nan
        mu_sad_interp = np.nan
        mu_max_interp = np.nan
        mu_min_interp = interp_weak_full_min_mu(y_interp, kappa_interp)

    T0_min_out_interp_seg, T0_sad_max_interp = make_T0_arr_multiple_chev_np(
        N,
        np.array([T_sad_interp, T_max_interp, 0]),
        T0_max = T0_max)
    T0_min_out_interp_full = np.concatenate(T0_min_out_interp_seg)
    u_min_out_interp_full = np.concatenate([u_interp_low, u_interp_mid_1, u_interp_mid_2, u_interp_high])

    T_im_hi = np.nanmax((T_max_interp, T_sad_interp))
    t_fft_short_max = T_im_hi*20
    t_fft_long_max = np.min((T_im_hi*2000, 1000))
    t_fft_short = np.linspace(0, t_fft_short_max, 2**16)
    t_fft_long = np.linspace(0, t_fft_long_max, 2**16)
    if strongly_lensed:
        F_fft_short = interp_F_fft_strong(t_fft_short, 
                                          T0_min_out_interp_full,
                                          u_min_out_interp_full, 
                                          T0_sad_max_interp, 
                                          u_interp_sad_max)
        F_fft_long = interp_F_fft_strong(t_fft_long,
                                         T0_min_out_interp_full,
                                         u_min_out_interp_full,
                                         T0_sad_max_interp,
                                         u_interp_sad_max)
    else:
        F_fft_short = interp_F_fft_weak(t_fft_short,
                                        T0_min_out_interp_full,
                                        u_min_out_interp_full)
        F_fft_long = interp_F_fft_weak(t_fft_long,
                                       T0_min_out_interp_full,
                                       u_min_out_interp_full)

    w_arr_high, Fw_high = amplification_fft_np(t_fft_short, F_fft_short)
    w_arr_low, Fw_low = amplification_fft_np(t_fft_long, F_fft_long)

    if strongly_lensed:
        mu_im = [mu_sad_interp, mu_max_interp, mu_min_interp]
        T_im = [T_sad_interp, T_max_interp, 0]
    else:
        mu_im = [0, 0, mu_min_interp]
        T_im = [0, 0, 0]

    w_trans_1 = 2.5/T_im_hi
    w_trans_2 = 250/T_im_hi if strongly_lensed else 50/T_im_hi

    w_list = [w_arr_low, w_arr_high]
    F_list = [Fw_low, Fw_high]
    w_low_trans = w_arr_low[0]
    partitions = np.array([w_low_trans, w_trans_1, w_trans_2])
    sigs = partitions/10

    if return_geom:

        F_interp, F_geometric = interp_partitions(w_interp, w_list, F_list, partitions, sigs, T_im, mu_im, 
                                 return_geom = return_geom)
        return F_interp, F_geometric, partitions, T_im, mu_im, u_interp_low, u_interp_mid_1, u_interp_mid_2, u_interp_high, u_interp_sad_max, w_arr_high, Fw_high, w_arr_low, Fw_low
    
    else:
            
        F_interp = interp_partitions(w_interp, w_list, F_list, partitions, sigs, T_im, mu_im, 
                                        return_geom = return_geom)
        return F_interp
    
def crit_mask(y, lp, y_interp_func, fac = 0.1):
    y_crit = y_interp_func(lp)
    return np.abs(y - y_crit) < fac*y_crit