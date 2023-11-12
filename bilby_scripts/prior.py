import bilby
from bilby.core.prior.joint import BaseJointPriorDist, JointPrior, JointPriorDistError
from bilby.core.utils import random
import numpy as np
from glworia.load_interp import load_interpolators

class Uniform2DMaskDist(BaseJointPriorDist):

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
        return self.bounds_arr[:, 0] + (self.bounds_arr[:, 1] - self.bounds_arr[:, 0]) * samp
    
    def _sample(self, size, **kwargs):
        
        samps = np.zeros((size, 2))
        for i in range(size):
            masked = True
            while masked:
                vals = random.rng.uniform(0, 1, 2)
                samp = np.atleast_1d(self.rescale(vals))

                if self.crit_mask_settings is not None:
                    masked = self.check_mask(samp)
                else:
                    masked = False
                samps[i, :] = samp
    
        return samps
    
    def _ln_prob(self, samp, lnprob, outbounds):
        return np.logaddexp(lnprob, 0)

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
                "dist object must be instance of MultivariateGaussianDist"
            )
        super(Uniform2DMask, self).__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit
        )
