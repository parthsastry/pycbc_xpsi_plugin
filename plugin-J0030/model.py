#!/usr/bin/env python

import numpy
import shlex
import xpsi
# from xpsi.Parameter import Derive
from xpsi.global_imports import gravradius

from .utils import (CustomInstrument, CustomSignal, CustomInterstellar,
                    CustomPhotosphere)

# from pycbc.workflow import WorkflowConfigParser
from pycbc.inference.models import BaseModel


class XPSIModel(BaseModel):
    """Model wrapper around XPSI likelihood function."""
    name = 'xpsi'

    # we need to alias some of the parameter names to be compliant with
    # pycbc config file sections
    """
    _param_aliases = {
        'XTI__alpha': 'nicer_alpha',
        'PN__alpha': 'xmm_alpha'
        }
    """
    # Comment - Currently testing out a NICER only likelihood
    # evaluation, hence these parameter names aren't used. In fact,
    # there is no parameter names list in the main run script at all.
    # Curious - need to confirm if this is a caveat of the implementation of
    # the MCMC sampler or something else. (WHY NO NAME LIST?)

    def __init__(self, variable_params, star, signals, num_energies, **kwargs):
        super().__init__(variable_params, **kwargs)
        # set up the xpsi likelihood.
        self._xpsi_likelihood = xpsi.Likelihood(star=star, signals=signals,
                                                num_energies=num_energies,
                                                externally_updated=True)
        # store a dictionary of param aliases. Will be needed for NICERxXMM
        """
        self.param_aliases = {p: p for p in self._xpsi_likelihood.names}
        self.param_aliases.update(self._param_aliases)
        """

    @property
    def star(self):
        return self._xpsi_likelihood.star

    def _loglikelihood(self):
        # map the current parameters to the ordered list
        # params = self.current_params
        # update the underlying likelihood. Don't need for now, no param
        # aliases.
        """for p in self._xpsi_likelihood.names:
            self._xpsi_likelihood[p] = params[self.param_aliases[p]]
        """
        # check addtional constraints
        if not self.apply_additional_constraints():
            return -numpy.inf
        logl = self._xpsi_likelihood()
        # FIXME: for some reason, this occasionally returns arrays of len 1
        # if so, force to float
        if isinstance(logl, numpy.ndarray):
            logl = logl.item()
        return logl

    def apply_additional_constraints(self):
        # FIXME: these should really be applied in the prior, but hard to
        # do right now because of the special functions involved, so we'll
        # just apply them in the likelihood call
        # Following copied from ST_U/CustomPrior.py
        spacetime = self.star.spacetime

        # limit polar radius to be outside the Schwarzschild photon sphere
        R_p = 1.0 + spacetime.epsilon * (-0.788 + 1.030 * spacetime.zeta)
        if R_p < 1.5 / spacetime.R_r_s:
            return False

        mu = numpy.sqrt(-1.0 /
                        (3.0 * spacetime.epsilon *
                         (-0.788 + 1.030 * spacetime.zeta)
                         ))
        if mu < 1.0:
            return False
        # check effective gravity at pole (where it is maximum) and
        # at equator (where it is minimum) are in NSX limits
        grav = xpsi.surface_radiation_field.effective_gravity(
            numpy.array([1.0, 0.0]),
            numpy.array([spacetime.R] * 2),
            numpy.array([spacetime.zeta] * 2),
            numpy.array([spacetime.epsilon] * 2))
        for g in grav:
            if not 13.7 <= g <= 15.0:
                return False
        return True

    @classmethod
    def from_config(cls, cp, **kwargs):
        # get the standard init args
        args = cls._init_args_from_config(cp)
        # get what instruments to analyze
        section = 'model'
        instruments = shlex.split(cp.get(section, 'instruments'))
        num_energies = int(cp.get(section, 'num-energies'))
        # get alpha/beta (QUESTION - WHY RENAMED for nicer vs nicerxXMM?) value
        # FIXME: Put this in prior. For now, to test out likelihood, use
        # Collin's old config file, so keep as is.

        # NOTE - for NICER only, parameter name is actually beta, not alpha.
        section = 'model'

        # NOTE - Current CustomInstrument implementation sets bounds on beta
        # directly in the from_SWG function. Hence, no need for bounds to be
        # supplied. CHECK IF INTENDED! WILL PROBABLY HAVE TO CHANGE FOR
        # NICERxXMM

        # min_alpha = float(cp.get(section, 'min-alpha'))
        # max_alpha = float(cp.get(section, 'max-alpha'))
        # alpha_bounds = (min_alpha, max_alpha)
        # load the instruments
        # NOTE - Current implementation only supports loading NICER. Look at
        # Collin's old repo for implementation to load XMM instruments.
        signals = []
        nicer = None
        if 'nicer' in instruments:
            nicer = nicer_from_config(cp)
            signals.append(nicer.signal)
        # load the spacetime
        spacetime = spacetime_from_config(cp)
        # load the hotregions
        hotregions = hotregions_from_config(cp)
        photosphere = photosphere_from_config(cp, hotregions, spacetime.f)
        star = xpsi.Star(spacetime=spacetime,
                         photospheres=photosphere)
        args['star'] = star
        args['signals'] = [signals]
        args['num_energies'] = num_energies
        args.update(kwargs)
        return cls(**args)


# -----------------------------------------------------------------------------
#
#                         Helper functions/classes
#
# -----------------------------------------------------------------------------

class Instrument:
    """Generic class for storing instrument properties."""
    def __init__(self, data, instrument, signal):
        self.data = data
        self.instrument = instrument
        self.signal = signal


def interstellar_from_config(cp):
    section = 'interstellar'
    attenuation_path = cp.get(section, 'attenuation-path')
    # NOTE - should this be drawn from prior as well?
    # OVERARCHING QUESTION -
    # Why are these classes initialized with the bounds on these parameters
    # in the first place? Shouldn't the bounds be drawn from the prior?
    column_density = (float(cp.get(section, 'min-column-density')),
                      float(cp.get(section, 'max-column-density')))
    interstellar = CustomInterstellar.from_SWG(
        attenuation_path,
        bounds=dict(column_density=column_density))
    return interstellar


def signal_from_config(cp, data, instrument, interstellar, **kwargs):
    section = 'signal'
    workspace_intervals = int(cp.get(section, 'workspace-intervals'))
    epsrel = float(cp.get(section, 'epsrel'))
    epsilon = float(cp.get(section, 'epsilon'))
    sigmas = float(cp.get(section, 'sigmas'))
    signal = CustomSignal(
        data=data,
        instrument=instrument,
        interstellar=interstellar,
        cache=False,
        workspace_intervals=workspace_intervals,
        epsrel=epsrel, epsilon=epsilon, sigmas=sigmas, **kwargs)
    return signal


def nicer_from_config(cp):
    section = 'nicer'
    # NOTE - changes in how this is initialised, will require changes in
    # Collin's old config file as well.
    counts = numpy.loadtxt(cp.get(section, 'matrix-path'))
    # channels
    min_channel = int(cp.get(section, 'min-channels'))
    max_channel = int(cp.get(section, 'max-channels'))
    channels = numpy.arange(min_channel, max_channel)
    # phase bins
    phmin = float(cp.get(section, 'min-phases'))
    phmax = float(cp.get(section, 'max-phases'))
    phbins = int(cp.get(section, 'phases-nbins'))
    phases = numpy.linspace(phmin, phmax, phbins)
    # other stuff
    first = int(cp.get(section, 'first'))
    last = int(cp.get(section, 'last'))
    exposure_time = float(cp.get(section, 'exposure-time'))
    # load the data
    data = xpsi.Data(counts,
                     channels=channels,
                     phases=phases,
                     first=first,
                     last=last,
                     exposure_time=exposure_time)
    # load the instrument specs
    # files
    arf = cp.get(section, 'arf-path')
    rmf = cp.get(section, 'rmf-path')
    # channels
    min_input = int(cp.get(section, 'min-input'))
    max_input = int(cp.get(section, 'max-input'))
    channel_edges = cp.get(section, 'channels-path')
    # load the instrument
    instrument = CustomInstrument.from_SWG(bounds=dict(beta=(None, None)),
                                           values={},
                                           ARF=arf,
                                           RMF=rmf,
                                           max_input=max_input,
                                           min_input=min_input,
                                           channel_edges=channel_edges)
    # load the interstellar medium
    interstellar = interstellar_from_config(cp)
    # load the signal
    signal = signal_from_config(cp, data, instrument, interstellar)
    return Instrument(data, instrument, signal)


def spacetime_from_config(cp):
    # setup the spacetime
    section = 'spacetime'
    # FIXME: these should be pulled from the prior
    # NOTE - bounds different in main script. Check if we should bound
    # stuff here or not.
    spacetime_bounds = {
        'mass': (1.0, 3.0),
        'radius': (3.0*gravradius(1.0), 16.0),
        'cos_inclination': (0.0, 1.0),
    }
    spacetime_freq = float(cp.get(section, 'frequency'))
    return xpsi.Spacetime(spacetime_bounds,
                          dict(frequency=spacetime_freq,
                               distance=0.01))  # Fixed dummy distance param


def read_hotspot_args(cp, section, common=None):
    if common is None:
        out = {}
    else:
        out = common.copy()
    # boolean options
    boolopts = ['symmetry', 'omit', 'cede', 'concentric',
                'is_antiphased']
    for opt in boolopts:
        out[opt] = cp.has_option(section, opt.replace('_', '-'))
    # options to cast to integer
    intopts = ['sqrt_num_cells',
               'min_sqrt_num_cells',
               'max_sqrt_num_cells',
               'num_leaves',
               'num_rays',
               'image_order_limit']
    # all other options will be read as string
    for opt in cp.options(section):
        storeopt = opt.replace('-', '_')
        if storeopt in boolopts:
            continue
        val = cp.get(section, opt)
        if storeopt in intopts:
            val = int(val)
        out[storeopt] = val
    return out


def hotregions_from_config(cp):
    # setup the hotspots
    # get the number and common options
    section = 'hotspots'
    spottags = cp.get_subsections(section)
    # FIXME: these should come from the prior
    # NEED TO PULL HOTREGION PRIORS FROM PRIOR
    hotspots_bounds = {
        'p': dict(
            super_colatitude=(0.001, numpy.pi-0.001),
            super_radius=(0.001, numpy.pi/2.0 - 0.001),
            phase_shift=(-0.25, 0.75),  # NOTE - changed from -0.5, 0.5. why?
            super_temperature=(5.1, 6.8)),
        's': dict(
            super_colatitude=(0.001, numpy.pi - 0.001),
            super_radius=(0.001, numpy.pi/2.0 - 0.001),
            phase_shift=(-0.5, 0.5),
            super_temperature=(5.1, 6.8)
        )
    }
    # get the common options
    common_opts = read_hotspot_args(cp, section)
    # load the spots
    spots = []
    for tag in spottags:
        spotopts = read_hotspot_args(cp, '-'.join([section, tag]),
                                     common=common_opts)
        bnds = hotspots_bounds[tag]
        spots.append(xpsi.HotRegion(bounds=bnds,
                                    values={},
                                    prefix=tag,
                                    **spotopts))
    return xpsi.HotRegions(tuple(spots))


def photosphere_from_config(cp, hotregions, spacetime_freq):
    # load the photosphere
    section = 'photosphere'
    photosphere = CustomPhotosphere(
        hot=hotregions, elsewhere=None,
        values={'mode_frequency': spacetime_freq})
    photosphere.hot_atmosphere = cp.get(section, 'atmosphere-path')
    return photosphere
