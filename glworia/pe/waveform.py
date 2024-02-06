import numpy as np

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from bilby.gw.utils import (lalsim_GetApproximantFromString,
                    lalsim_SimInspiralFD,
                    lalsim_SimInspiralChooseFDWaveform,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda1,
                    lalsim_SimInspiralWaveformParamsInsertTidalLambda2,
                    lalsim_SimInspiralChooseFDWaveformSequence)
from bilby.gw.source import _base_lal_cbc_fd_waveform, _base_waveform_frequency_sequence
from bisect import bisect_left, bisect_right
from scipy.constants import c, G
Msun = 1.9884099e+30
Mtow = 8*np.pi*G/c**3*Msun

from ..amp.load_interp import *

def lal_binary_black_hole_lensed(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, MLz, y, lp, **kwargs):

    ws = frequency_array*Mtow*MLz
    F_interp = kwargs['F_interp']
    Fs = F_interp(ws, y, lp)
    del kwargs['F_interp']
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXAS', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    waveform_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
    waveform_dict["plus"] *= Fs
    waveform_dict["cross"] *= Fs
    return waveform_dict

def lal_binary_black_hole_unlensed(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXAS', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    waveform_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
    return waveform_dict

def lal_binary_black_hole_lensed_relative_binning(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, MLz, y, lp, fiducial, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXAS', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    if fiducial == 1:
        waveform_dict = _base_lal_cbc_fd_waveform(
            frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
            luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
            a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
            phi_jl=phi_jl, **waveform_kwargs)
        ws = frequency_array*Mtow*MLz
    else:
        waveform_kwargs["frequencies"] = waveform_kwargs.pop("frequency_bin_edges")
        waveform_dict = _base_waveform_frequency_sequence(
            frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
            luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
            a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_jl=phi_jl,
            phi_12=phi_12, lambda_1=0.0, lambda_2=0.0, **waveform_kwargs)
        ws = waveform_kwargs["frequencies"]*Mtow*MLz
    F_interp = kwargs['F_interp']
    Fs = F_interp(ws, y, lp)
    del kwargs['F_interp']
    waveform_dict["plus"] *= Fs
    waveform_dict["cross"] *= Fs
    return waveform_dict
