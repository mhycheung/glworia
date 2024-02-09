#!/usr/bin/env python

import os
import sys
import bilby
import numpy as np
import argparse
import json

# get number of cpus available in the slurm job
nproc = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
print('nproc = ', nproc)

project_dir = os.environ['GLWORIA_PREFIX']
interpolator_dir = os.path.join(project_dir, 'interpolation')

sys.path.append(project_dir)

from glworia.amp.load_interp import *
from glworia.pe.waveform import *
from glworia.pe.prior import *

parser = argparse.ArgumentParser()

parser.add_argument('config_path', type=str, help='Path to config file')
parser.add_argument('-iul', '--inject_unlensed', dest = 'inject_lensed', action='store_false', help='Inject unlensed waveform')
parser.add_argument('-mul', '--model_unlensed', dest = 'model_lensed', action='store_false', help='Model unlensed waveform')

args = parser.parse_args()

config_path = args.config_path
inject_lensed = args.inject_lensed
model_lensed = args.model_lensed
runname = os.path.splitext(os.path.basename(config_path))[0]

# config = configparser.ConfigParser()
# config.optionxform = str
# config.read(config_path)
with open(config_path, 'r') as f:
    config = json.load(f)

injection_parameters_lensed = config['injection_parameters']
injection_parameters_lensed.update(fiducial = 1)
if inject_lensed or model_lensed:
    interpolator_settings = config['interpolator_settings']
prior_settings = config['prior_settings']
waveform_arguments_unlensed = config['waveform_arguments']
sampler_settings = config['sampler_settings']
misc = config['misc']

if 'log_level' not in misc:
    misc['log_level'] = 'info'

outdir = os.path.join(project_dir, 'outdir', misc['outdir_ext'])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

if inject_lensed or model_lensed:
    lp_name = misc['lp_name']
    lp_latex = misc['lp_latex']

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = misc['duration']
sampling_frequency = misc['sampling_frequency']
minimum_frequency = misc['minimum_frequency']

# Specify the output directory and the name of the simulation.
label = runname
outdir = os.path.join(outdir, misc['outdir_ext'], label)
os.makedirs(outdir, exist_ok=True)

bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level = misc['log_level'])

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(misc['seed'])

if inject_lensed or model_lensed:
    interpolators = load_interpolators(interpolator_dir, **interpolator_settings)
    F_interp_loaded = lambda w, y, kappa: F_interp(w, y, kappa, interpolators, interpolator_settings)

    waveform_arguments_lensed = waveform_arguments_unlensed.copy()
    # Fixed arguments passed into the source model
    waveform_arguments_lensed.update(F_interp = F_interp_loaded)

injection_parameters_unlensed = injection_parameters_lensed.copy()
for key in ['y', 'lp', 'MLz']:
    try:
        injection_parameters_unlensed.pop(key)
    except KeyError:
        pass

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
if model_lensed or inject_lensed:
    waveform_generator_lensed = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole_lensed_relative_binning,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments_lensed,
    )
elif not (model_lensed and inject_lensed):
    waveform_generator_unlensed = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments_unlensed,
    )

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
if misc['zero_noise']:
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters_lensed["geocent_time"] - 2,
    )
else:
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters_lensed["geocent_time"] - 2,
    )
if inject_lensed:
    ifos.inject_signal(
        waveform_generator=waveform_generator_lensed, parameters=injection_parameters_lensed
    )
    inject_wf_generator = waveform_generator_lensed
    injection_parameters = injection_parameters_lensed
else:
    ifos.inject_signal(
        waveform_generator=waveform_generator_unlensed, parameters=injection_parameters_unlensed
    )
    inject_wf_generator = waveform_generator_unlensed
    injection_parameters = injection_parameters_unlensed

# Set up a PriorDict, which inherits from dict.

priors = bilby.gw.prior.BBHPriorDict()

if model_lensed:
    if prior_settings['MLz_prior_type'] == 'uniform':
        MLz_prior = bilby.core.prior.Uniform
    elif prior_settings['MLz_prior_type'] == 'loguniform':
        MLz_prior = bilby.core.prior.LogUniform
    else:
        raise ValueError('MLz_prior_type must be either uniform or loguniform')

    priors["MLz"] = MLz_prior(
        minimum=prior_settings['MLz_min'],
        maximum=prior_settings['MLz_max'],
        name="MLz",
        latex_label="$M_{Lz}$",
        unit="$M_\odot$",
    )

    crit_mask_settings = prior_settings['crit_mask_settings']
    crit_mask_settings.update(lens_param_to_y_crit = interpolators['lens_param_to_y_crit'])

    ylp_masked = Uniform2DMaskDist(
        names = ['y', 'lp'],
        bounds = {
            'y': (prior_settings['y_min'], prior_settings['y_max']),
            'lp': (prior_settings['lp_min'], prior_settings['lp_max'])
        },
        crit_mask_settings = crit_mask_settings)

    priors['y'] = Uniform2DMask(ylp_masked, name = 'y', latex_label = '$y$', unit = None)
    priors['lp'] = Uniform2DMask(ylp_masked, name = 'lp', latex_label = lp_latex, unit = None)

if 'luminosity_distance_prior_type' in prior_settings:
    if prior_settings['luminosity_distance_prior_type'] == 'uniform':
        luminosity_distance_prior = bilby.core.prior.Uniform
    elif prior_settings['luminosity_distance_prior_type'] == 'loguniform':
        luminosity_distance_prior = bilby.core.prior.LogUniform
    elif prior_settings['luminosity_distance_prior_type'] == 'uniformsourceframe':
        luminosity_distance_prior = bilby.gw.prior.UniformSourceFrame
    else:
        raise ValueError('luminosity_distance_prior_type must be either uniform or loguniform')

    priors["luminosity_distance"] = luminosity_distance_prior(
        minimum=prior_settings['luminosity_distance_min'],
        maximum=prior_settings['luminosity_distance_max'],
        name="luminosity_distance",
        latex_label="$d_L$",
        unit="Mpc",
    )

time_delay = ifos[0].time_delay_from_geocenter(
    injection_parameters_lensed["ra"],
    injection_parameters_lensed["dec"],
    injection_parameters_lensed["geocent_time"],
)
priors["H1_time"] = bilby.core.prior.Uniform(
    minimum=injection_parameters_lensed["geocent_time"] + time_delay - 0.1,
    maximum=injection_parameters_lensed["geocent_time"] + time_delay + 0.1,
    name="H1_time",
    latex_label="$t_H$",
    unit="$s$",
)
del priors["ra"], priors["dec"]
priors["zenith"] = bilby.core.prior.Sine(latex_label="$\\kappa_z$")
priors["azimuth"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * np.pi, latex_label="$\\epsilon_a$", boundary="periodic"
)

if waveform_arguments_unlensed["waveform_approximant"] in ["IMRPhenomXAS", "IMRPhenomXPHM", "IMRPhenomXHM", "IMRPhenomD"]:
    for key in ["tilt_1", "tilt_2", "phi_12", "phi_jl"]:
        priors[key] = injection_parameters_lensed[key]

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

if model_lensed:
    fiducial_parameters = injection_parameters_lensed.copy()
else:
    fiducial_parameters = injection_parameters_unlensed.copy()
m1 = fiducial_parameters.pop("mass_1")
m2 = fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(
    m1, m2
)
fiducial_parameters["mass_ratio"] = m2 / m1


# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit distance marginalization is turned on to improve
# convergence, and the posterior is recovered by the conversion function.
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=inject_wf_generator,
    priors=priors,
    distance_marginalization=False,
    phase_marginalization=False,
    time_marginalization=True,
    reference_frame="H1L1",
    time_reference="H1",
    fiducial_parameters=fiducial_parameters,
)


# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    npool=nproc,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
    **sampler_settings,
)

# Make a corner plot.
result.plot_corner()
