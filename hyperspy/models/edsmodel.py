# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import copy
import numpy as np
import traits.api as t

from hyperspy.model import Model
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds import model as model_eds
import hyperspy.components as create_component


def _get_ratio(element, line):
    ratio_line = elements_db[
        element]['Atomic_properties']['Xray_lines'][line]['factor']
    return lambda x: x * ratio_line


def _get_iratio(element, line):
    ratio_line = elements_db[
        element]['Atomic_properties']['Xray_lines'][line]['factor']
    return lambda x: x / ratio_line


class EDSModel(Model):

    """Build a fit a model

    Parameters
    ----------
    spectrum : an Spectrum (or any Spectrum subclass) instance
    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically
        a powerlaw to the model and estimate the parameters by the
        two-area method.

    """

    def __init__(self, spectrum, auto_background=True,
                 auto_add_lines=True,
                 *args, **kwargs):

                    #,
                 #auto_add_edges=True, ll=None,
                 # GOS=None, *args, **kwargs):
        Model.__init__(self, spectrum, *args, **kwargs)
        #self._suspend_auto_fine_structure_width = False
        #self.convolved = False
        #self.low_loss = ll
        #self.GOS = GOS
        self.xray_lines = list()
        # if auto_background is True:
        #    interactive_ns = get_interactive_ns()
        #    background = PowerLaw()
        #    background.name = 'background'
        #    interactive_ns['background'] = background
        #    self.append(background)

        # if self.spectrum.xray_lines and auto_add_lines is True:
        if auto_add_lines is True:
            self._add_lines()
        # if auto_background is True:
        #    self._add_background()

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, EDSSpectrum):
            self._spectrum = value
            self.spectrum._are_microscope_parameters_missing()
        else:
            raise ValueError(
                "This attribute can only contain an EDSSpectrum "
                "but an object of type %s was provided" %
                str(type(value)))

    def _add_lines(self, xray_lines=None, only_one=False,
                   only_lines=("Ka", "La", "Ma")):
        """Create the Xray-lines instances and configure them appropiately

        Parameters
        -----------
        xray_lines: {None, "best", list of string}
            If None,
            if `mapped.parameters.Sample.elements.xray_lines` contains a
            list of lines use those.
            If `mapped.parameters.Sample.elements.xray_lines` is undefined
            or empty but `mapped.parameters.Sample.elements` is defined,
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
            Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, use only the given lines.
        """
        if xray_lines is None:
            if 'Sample.xray_lines' in self.spectrum.metadata:
                xray_lines = self.spectrum.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.spectrum.metadata:
                xray_lines = self.spectrum._get_lines_from_elements(
                    self.spectrum.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "Not X-ray line, set them with `add_elements`")
        self.xray_lines = xray_lines
        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy, line_FWHM = self.spectrum._get_line_energy(xray_line,
                                                                    FWHM_MnKa='auto')
            fp = create_component.Gaussian()
            fp.centre.value = line_energy
            fp.sigma.value = line_FWHM / 2.355
            fp.centre.free = False
            fp.sigma.free = False
            fp.name = xray_line
            self.append(fp)
            init = True
            if init:
                self[xray_line].A.map[
                    'values'] = self.spectrum[..., line_energy].data
                self[xray_line].A.map['is_set'] = (
                    np.ones(self.spectrum[..., line_energy].data.shape) == 1)
            # if bounded:
            #    fp.A.ext_bounded = True
            #    fp.A.ext_force_positive = True
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    line_energy, line_FWHM = self.spectrum._get_line_energy(
                        xray_sub, FWHM_MnKa='auto')
                    fp_sub = create_component.Gaussian()
                    fp_sub.centre.value = line_energy
                    fp_sub.name = xray_sub
                    fp_sub.sigma.value = line_FWHM / 2.355
                    fp_sub.A.twin = fp.A
                    fp_sub.centre.free = False
                    fp_sub.sigma.free = False
                    fp_sub.A.twin_function = _get_ratio(element, li)
                    fp_sub.A.twin_inverse_function = _get_iratio(
                        element, li)
                    self.append(fp_sub)

    def get_line_intensities(self,
                             plot_result=True,
                             store_in_mp=True,
                             **kwargs):
        """

        Parameters
        ----------
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
        store_in_mp : bool
            store the result in metadata.Sample
        kwargs
            The extra keyword arguments for plotting. See
            `utils.plot.plot_signals`
        """
        xray_lines = self.xray_lines
        intensities = []
        if self.spectrum.metadata.Sample.has_item(
                'xray_lines') is False and store_in_mp:
            self.spectrum.metadata.Sample.xray_lines = xray_lines
        for i, xray_line in enumerate(xray_lines):
            line_energy = self.spectrum._get_line_energy(xray_line)
            data_res = self[xray_line].A.map['values']
            if self.axes_manager.navigation_dimension == 0:
                data_res = data_res[0]
            img = self.spectrum._set_result(xray_line, 'intensities',
                                            data_res, plot_result=False,
                                            store_in_mp=store_in_mp)
            intensities.append(img)
            if plot_result and img.axes_manager.signal_dimension == 0:
                print("%s at %s %s : Intensity = %.2f"
                      % (xray_line,
                         line_energy,
                         self.spectrum.axes_manager.signal_axes[0].units,
                         img.data))
        if plot_result and img.axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(intensities, **kwargs)
        if store_in_mp is False:
            return intensities
