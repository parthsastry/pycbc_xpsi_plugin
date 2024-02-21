#!/usr/bin/env python

"""Utilities for setting up NICER and XMM Newtwon analysis.

FIXME: these settings are specific to the STU model with J0030
"""

import numpy as np
import math
import six as _six

from scipy.interpolate import Akima1DInterpolator

import xpsi
from xpsi import Parameter
from xpsi.utils import make_verbose
from xpsi.likelihoods.default_background_marginalisation import (
                                    eval_marginal_likelihood)
from xpsi.likelihoods.default_background_marginalisation import precomputation

#
#   Following copied from the ST_U directory in:
#   updated_analyses_PSRJ0030_up_to_2018_NICER_data
#   which was downloaded from https://zenodo.org/records/8239000
#

class CustomInstrument(xpsi.Instrument):
    """ Methods and attributes specific to the NICER instrument.

    Currently tailored to the NICER light-curve SWG model specification.

    """
    def construct_matrix(self):
        """ Implement response matrix parameterisation. """
        # Multiplying beta to response matrix
        beta_d = self['beta'] * 0.01**2
        # beta_d = beta * (d kpc)^2; 0.01 dummy distance
        matrix = beta_d*self.matrix
        matrix[matrix < 0.0] = 0.0

        return matrix

    def __call__(self, signal, *args):
        """ Overwrite. """

        matrix = self.construct_matrix()

        self._cached_signal = np.dot(matrix, signal)

        return self._cached_signal

    @classmethod
    def from_SWG(cls,
                 bounds, values,
                 ARF, RMF,
                 max_input, min_input=0,
                 channel_edges=None):
        """ Constructor which converts files into :class:`numpy.ndarray`s.

        :param str ARF: Path to ARF which is compatible with
                                :func:`numpy.loadtxt`.

        :param str RMF: Path to RMF which is compatible with
                                :func:`numpy.loadtxt`.

        """
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=3)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=3, usecols=-1)

        if channel_edges:
            channel_edges = np.loadtxt(channel_edges,
                                       dtype=np.double,
                                       skiprows=3)

        matrix = np.zeros((1501, 3451))

        for i in range(3451):
            matrix[:, i] = RMF[i*1501:(i+1)*1501]

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        edges = np.zeros(ARF[min_input:max_input, 3].shape[0]+1,
                         dtype=np.double)

        edges[0] = ARF[min_input, 1]
        edges[1:] = ARF[min_input:max_input, 2]

        RSP = np.ascontiguousarray(
            np.zeros(
                     matrix[30:300, min_input:max_input].shape
            ), dtype=np.double)

        for i in range(RSP.shape[0]):
            RSP[i, :] = (matrix[i+30, min_input:max_input] *
                         ARF[min_input:max_input, 3]*49./52.)

        channels = np.arange(30, 300)

        beta = Parameter('beta',
                         strict_bounds=(0.1, 30.0),
                         bounds=bounds.get('beta', None),
                         doc='Units of kpc^-2',
                         symbol=r'$\beta$',
                         value=values.get('beta', None))

        return cls(RSP, edges, channels, channel_edges[30:301, 1], beta)


class CustomSignal(xpsi.Signal):
    """ A custom calculation of the logarithm of the NICER likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We overwrite the body of the __call__ method. The docstring for the
    abstract method is copied.

    """

    def __init__(self, workspace_intervals=1000, epsabs=0, epsrel=1.0e-8,
                 epsilon=1.0e-3, sigmas=10.0, support=None, *args, **kwargs):
        """ Perform precomputation. """

        super(CustomSignal, self).__init__(*args, **kwargs)

        try:
            self._precomp = precomputation(self._data.counts.astype(np.int32))
        except AttributeError:
            print('No data... can synthesise data but cannot evaluate a '
                  'likelihood function.')
        else:
            self._workspace_intervals = workspace_intervals
            self._epsabs = epsabs
            self._epsrel = epsrel
            self._epsilon = epsilon
            self._sigmas = sigmas

            if support is not None:
                self._support = support
            else:
                self._support = -1.0 * np.ones((self._data.counts.shape[0], 2))
                self._support[:, 0] = 0.0

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, obj):
        self._support = obj

    def __call__(self, *args, **kwargs):
        self.loglikelihood, self.expected_counts, self.background_signal, \
            self.background_signal_given_support = \
            eval_marginal_likelihood(self._data.exposure_time,
                                     self._data.phases,
                                     self._data.counts,
                                     self._signals,
                                     self._phases,
                                     self._shifts,
                                     self._precomp,
                                     self._support,
                                     self._workspace_intervals,
                                     self._epsabs,
                                     self._epsrel,
                                     self._epsilon,
                                     self._sigmas,
                                     kwargs.get('llzero'))


class CustomInterstellar(xpsi.Interstellar):
    """ Apply interstellar attenuation. """

    def __init__(self, energies, attenuation, bounds, values={}):

        assert len(energies) == len(attenuation), 'Array length mismatch.'

        self._lkp_energies = energies  # for lookup
        self._lkp_attenuation = attenuation  # for lookup

        N_H = Parameter('column_density',
                        strict_bounds=(0.0, 10.0),
                        bounds=bounds.get('column_density', None),
                        doc='Units of 10^20 cm^-2.',
                        symbol=r'$N_{\rm H}$',
                        value=values.get('column_density', None))

        self._interpolator = Akima1DInterpolator(self._lkp_energies,
                                                 self._lkp_attenuation)
        self._interpolator.extrapolate = True

        super(CustomInterstellar, self).__init__(N_H)

    def attenuation(self, energies):
        """ Interpolate the attenuation coefficients.

        Useful for post-processing.

        """
        return self._interpolate(energies)**(self['column_density']/0.4)

    def _interpolate(self, energies):
        """ Helper. """
        _att = self._interpolator(energies)
        _att[_att < 0.0] = 0.0
        if len(_att[_att > 1.0]) > 0:
            raise ValueError('Interpolation applied to the instrument ' +
                             'energies, generates attenuation numbers higher' +
                             ' than 1.')
        return _att

    @classmethod
    def from_SWG(cls, path, **kwargs):
        """ Load attenuation file from the NICER SWG. """
    
        temp = np.loadtxt(path, dtype=np.double)

        energies = temp[:, 0]

        attenuation = temp[:, 2]

        return cls(energies, attenuation, **kwargs)


class CustomPhotosphere(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX.

    Fully-ionized hydrogen, v200802 (W.C.G. Ho).

    """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        NSX = np.loadtxt(path, dtype=np.double)
        logT = np.zeros(35)
        logg = np.zeros(14)
        mu = np.zeros(67)
        logE = np.zeros(166)

        reorder_buf = np.zeros((35, 14, 67, 166))

        index = 0
        for i in range(reorder_buf.shape[0]):
            for j in range(reorder_buf.shape[1]):
                for k in range(reorder_buf.shape[3]):
                    for l in range(reorder_buf.shape[2]):
                        logT[i] = NSX[index, 3]
                        logg[j] = NSX[index, 4]
                        logE[k] = NSX[index, 0]
                        mu[reorder_buf.shape[2] - l - 1] = NSX[index, 1]
                        reorder_buf[i, j, reorder_buf.shape[2] - l - 1, k] = \
                            10.0**(NSX[index, 2])
                        index += 1

        buf = np.zeros(np.prod(reorder_buf.shape))

        bufdex = 0
        for i in range(reorder_buf.shape[0]):
            for j in range(reorder_buf.shape[1]):
                for k in range(reorder_buf.shape[2]):
                    for l in range(reorder_buf.shape[3]):
                        buf[bufdex] = reorder_buf[i, j, k , l]
                        bufdex += 1

        self._hot_atmosphere = (logT, logg, mu, logE, buf)

    @property
    def global_variables(self):
        """ This method is needed if we also want to invoke the image-plane
        signal simulator.

        The extension module compiled is
        surface_radiation_field/archive/local_variables/two_spots.pyx,
        which replaces the contents of
        surface_radiation_field/local_variables.pyx.

        """
        return np.array([self['p__super_colatitude'],
                         self['p__phase_shift'] * 2.0 * math.pi,
                         self['p__super_radius'],
                         self['p__super_temperature'],
                         self['s__super_colatitude'],
                         (self['s__phase_shift'] + 0.5) * 2.0 * math.pi,
                         self['s__super_radius'],
                         self['s__super_temperature']])
