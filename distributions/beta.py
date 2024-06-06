import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import Akima1DInterpolator
from scipy.stats import gaussian_kde

# have set rng seed for the sake of reproducibility in the beta invcdf
# and logpdf functions.
# NOTE - talk to Collin about whether this is something problematic.
# Potential FIXME - see if better way to create the logpdf and invcdf functions
rng = np.random.default_rng(10203)

uniform_sampler1 = rng.uniform(0, 1, 100000)
uniform_sampler2 = rng.uniform(0, 1, 100000)

_alpha = truncnorm.ppf(uniform_sampler1, -5.0, 5.0, loc=1.0, scale=0.1)
_distance = truncnorm.ppf(uniform_sampler2, -5.0, 5.0, loc=0.325, scale=0.009)

_beta_created = _alpha/np.power(_distance, 2)
_ordered_beta = np.unique(_beta_created)
_beta_cdf = np.array(range(len(_ordered_beta)))/float(len(_ordered_beta))

# Hardcoded bandwidth value for gaussian_kde according to Silverman's Rule of
# Thumb
_beta_pdf = gaussian_kde(_ordered_beta, bw_method=0.098325)

_beta_invcdf = Akima1DInterpolator(_beta_cdf, _ordered_beta)
_beta_invcdf.extrapolate = True

# NOTE - have to check whether importing param names from the distribution
# that imports this module is possible. If not, will have to hardcode
# the param names here. (as done in _cdfinv_beta)


def _logpdf_beta(beta=None):
    """ Logarithm of the probability density function for beta. """
    return np.log(_beta_pdf(beta))


# Finicky. Need to test this.
def _cdfinv_beta(**kwargs):
    """ Inverse of the cumulative distribution function for beta. """
    updated = {}
    params = ['beta']
    for param in params:
        updated[param] = _beta_invcdf(kwargs[param])
    return updated
