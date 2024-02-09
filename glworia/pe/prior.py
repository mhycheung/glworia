import bilby
from bilby.core.prior.joint import BaseJointPriorDist, JointPrior, JointPriorDistError
from bilby.core.utils import random
import numpy as np
from ..amp.load_interp import load_interpolators
import json
from bilby.core.utils.io import *

class Uniform2DMaskDist(BaseJointPriorDist):
    """
    A 2D joint prior with a mask around the caustic curve in the $y$-$l$ plane.
    """

    def __init__(
            self,
            names,
            bounds,
            crit_mask_settings = None,
            mask_boxes = [],
    ):
        super(Uniform2DMaskDist, self).__init__(names=names, bounds=bounds)
        self.names = names
        self.bounds = bounds
        self.bounds_arr = np.array([bounds[names[0]], bounds[names[1]]])
        if crit_mask_settings is not None:
            self.crit_mask_settings = {'fac': 0.5, 'cap_high': 0.2, 'cap_low':0.0}
            self.crit_mask_settings.update(crit_mask_settings)
            self.lens_param_to_y_crit = self.crit_mask_settings['lens_param_to_y_crit']
            self.fac = self.crit_mask_settings['fac']
            self.cap_high = self.crit_mask_settings['cap_high']
            self.cap_low = self.crit_mask_settings['cap_low']
        else:
            self.crit_mask_settings = None
        self.mask_boxes = mask_boxes

    def _rescale(self, samp, **kwargs):
        scaled_samps = self.bounds_arr[:, 0] + (self.bounds_arr[:, 1] - self.bounds_arr[:, 0]) * samp
        if self.crit_mask_settings is not None:
            for i in range(scaled_samps.shape[0]):
                samp_i = scaled_samps[i, :]
                masked = self.check_mask(samp_i)
                while masked:
                    samp_new = random.rng.uniform(0, 1, 2) * (self.bounds_arr[:, 1] - self.bounds_arr[:, 0]) + self.bounds_arr[:, 0]
                    masked = self.check_mask(samp_new)
                    scaled_samps[i, :] = samp_new
        return scaled_samps
    
    def _sample(self, size, **kwargs):
        
        samps = np.zeros((size, 2))
        for i in range(size):
            # masked = True
            # while masked:
            vals = random.rng.uniform(0, 1, 2)
            samp = np.atleast_1d(self.rescale(vals))

                # if self.crit_mask_settings is not None:
                #     masked = self.check_mask(samp)
                # else:
                #     masked = False
            samps[i, :] = samp
    
        return samps
    
    def _ln_prob(self, samp, lnprob, outbounds):
        lnprob = np.logaddexp(lnprob, 0)
        lnprob[outbounds] = -np.inf
        for i in range(samp.shape[0]):
            if self.check_mask(samp[i, :]):
                lnprob[i] = -np.inf
        return lnprob

    def check_mask(self, samp):
        y = samp[0]
        lp = samp[1]
        y_crit = self.lens_param_to_y_crit(lp)
        masked = np.abs(y - y_crit) < np.max((np.min((self.fac*y_crit, self.cap_high)), self.cap_low))
        for box in self.mask_boxes:
            masked = masked or (box[0][0] < y < box[0][1] and box[1][0] < lp < box[1][1])
        return masked

class Uniform2DMask(JointPrior):
    def __init__(self, dist, name = None, latex_label = None, unit = None):
        if not isinstance(dist, Uniform2DMaskDist):
            raise JointPriorDistError(
                "dist object must be instance of Uniform2DMaskDist"
            )
        super(Uniform2DMask, self).__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit
        )

    def to_json(self):
        return json.dumps(self, cls = BilbyJsonEncoderLens)
    
def make_conversion_y_pow_alpha_MLz(alpha):
    return lambda parameters: convert_to_lal_BBH_and_y_pow_alpha_MLz(parameters, alpha)

def y_pow_alpha_MLz_conversion(parameters, added_keys, alpha):
    converted_parameters = parameters.copy()
    converted_parameters['MLz'] = parameters['y_to_alpha_MLz']/(parameters['y']**alpha)
    added_keys.append('MLz')
    return converted_parameters, added_keys

def convert_to_lal_BBH_and_y_pow_alpha_MLz(parameters, alpha):
    parameters, added_keys = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(parameters)
    parameters, added_keys = y_pow_alpha_MLz_conversion(parameters, added_keys, alpha)
    return parameters, added_keys

# allow for dumping Uniform2DMaskDist to json
class BilbyJsonEncoderLens(json.JSONEncoder):
    def default(self, obj):
        from bilby.core.prior import MultivariateGaussianDist, Prior, PriorDict
        from bilby.gw.prior import HealPixMapPriorDist
        from bilby.bilby_mcmc.proposals import ProposalCycle
        from scipy.interpolate._interpolate import interp1d

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, PriorDict):
            return {"__prior_dict__": True, "content": obj._get_json_dict()}
        if isinstance(obj, (MultivariateGaussianDist, HealPixMapPriorDist, Prior, Uniform2DMaskDist)):
            return {
                "__prior__": True,
                "__module__": obj.__module__,
                "__name__": obj.__class__.__name__,
                "kwargs": dict(obj.get_instantiation_dict()),
            }
        if isinstance(obj, interp1d):
            return {"__interp1d__": True, "x": obj.x.tolist(), "y": obj.y.tolist()}
        if isinstance(obj, ProposalCycle):
            return str(obj)
        try:
            from astropy import cosmology as cosmo, units

            if isinstance(obj, cosmo.FLRW):
                return encode_astropy_cosmology(obj)
            if isinstance(obj, units.Quantity):
                return encode_astropy_quantity(obj)
            if isinstance(obj, units.PrefixUnit):
                return str(obj)
        except ImportError:
            logger.debug("Cannot import astropy, cannot write cosmological priors")
        if isinstance(obj, np.ndarray):
            return {"__array__": True, "content": obj.tolist()}
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        if isinstance(obj, pd.DataFrame):
            return {"__dataframe__": True, "content": obj.to_dict(orient="list")}
        if isinstance(obj, pd.Series):
            return {"__series__": True, "content": obj.to_dict()}
        if inspect.isfunction(obj):
            return {
                "__function__": True,
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        if inspect.isclass(obj):
            return {
                "__class__": True,
                "__module__": obj.__module__,
                "__name__": obj.__name__,
            }
        if isinstance(obj, (timedelta)):
            return {
                "__timedelta__": True,
                "__total_seconds__": obj.total_seconds()
            }
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
