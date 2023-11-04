#!/usr/bin/env python

import os
import sys
import bilby
import numpy as np
import argparse
import configparser

import subprocess

# get number of cpus available in the slurm job
nproc = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
print('nproc = ', nproc)

project_dir = os.environ['GLWORIA_PREFIX']
interpolator_dir = os.path.join(project_dir, 'interpolation')

sys.path.append(project_dir)

from glworia.load_interp import *
from bilby_scripts.waveform import *

parser = argparse.ArgumentParser()

parser.add_argument('config_path', type=str, help='Path to config file')

args = parser.parse_args()

config_path = args.config_path
runname = os.path.splitext(os.path.basename(config_path))[0]

config = configparser.ConfigParser()
config.optionxform = str
config.read(config_path)

injection_parameters = dict(config.items('injection_parameters'))
interpolator_settings = dict(config.items('interpolator_settings'))
prior_settings = dict(config.items('prior_settings'))
waveform_arguments = dict(config.items('waveform_arguments'))
sampler_settings = dict(config.items('sampler_settings'))
misc = dict(config.items('misc'))

outdir = os.path.join(project_dir, 'outdir', misc['outdir_ext'])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# convert string to string, float, int, or bool in above dicts
for d in [injection_parameters, interpolator_settings, prior_settings, waveform_arguments, sampler_settings, misc]:
    for key, value in d.items():
        if is_number(value):
            d[key] = eval(value)
        elif value == 'True':
            d[key] = True
        elif value == 'False':
            d[key] = False

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

bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(misc['seed'])

interpolators = load_interpolators(interpolator_dir, **interpolator_settings)
F_interp_loaded = lambda w, y, kappa: F_interp(w, y, kappa, interpolators, interpolator_settings)

# Fixed arguments passed into the source model
waveform_arguments.update(F_interp = F_interp_loaded)

# Create the waveform_generator using a LAL BinaryBlackHole source function
# the generator will convert all the parameters
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=lal_binary_black_hole_lensed,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.

priors = bilby.gw.prior.BBHPriorDict()

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

if prior_settings['y_prior_type'] == 'uniform':
    y_prior = bilby.core.prior.Uniform
elif prior_settings['y_prior_type'] == 'loguniform':
    y_prior = bilby.core.prior.LogUniform
else:
    raise ValueError('y_prior_type must be either uniform or loguniform')

priors["y"] = y_prior(
    minimum=prior_settings['y_min'],
    maximum=prior_settings['y_max'],
    name="y",
    latex_label="$y$",
)

if prior_settings[lp_name + '_prior_type'] == 'uniform':
    lp_prior = bilby.core.prior.Uniform
elif prior_settings[lp_name + '_prior_type'] == 'loguniform':
    lp_prior = bilby.core.prior.LogUniform
else:
    raise ValueError('lp_prior_type must be either uniform or loguniform')

priors[lp_name] = lp_prior(
    minimum=prior_settings[lp_name + '_min'],
    maximum=prior_settings[lp_name + '_max'],
    name=lp_name,
    latex_label=lp_latex,
)

for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    npool=nproc,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    **sampler_settings,
)

# Make a corner plot.
result.plot_corner()
