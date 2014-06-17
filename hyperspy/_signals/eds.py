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
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import copy

from hyperspy import utils
from hyperspy._signals.spectrum import Spectrum
from hyperspy.signal import Signal
from hyperspy._signals.image import Image
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds import image_eds
from hyperspy.misc.eds import database
from hyperspy.misc.utils import isiterable
import hyperspy.components as create_component
from hyperspy.drawing import marker
from hyperspy.drawing.utils import plot_histograms
from hyperspy.misc.eds import model as model_eds


def _get_weight(element, line):
    weight_line = elements_db[
        element]['Atomic_properties']['Xray_lines'][line]['weight']
    return lambda x: x * weight_line


def _get_iweight(element, line):
    weight_line = elements_db[
        element]['Atomic_properties']['Xray_lines'][line]['weight']
    return lambda x: x / weight_line


class EDSSpectrum(Spectrum):
    _signal_type = "EDS"

    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        if self.metadata.Signal.signal_type == 'EDS':
            print('The microscope type is not set. Use '
                  'set_signal_type(\'EDS_TEM\') or set_signal_type(\'EDS_SEM\')')
        self.metadata.Signal.binned = True

    def _get_line_energy(self, Xray_line, FWHM_MnKa=None):
        """
        Get the line energy and the energy resolution of a Xray line.

        The return values are in the same units than the signal axis

        Parameters
        ----------

        Xray_line : strings
            Valid element X-ray lines e.g. Fe_Kb.

        FWHM_MnKa: {None, float, 'auto'}
            The energy resolution of the detector in eV
            if 'auto', used the one in
            'self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa'

        Returns
        ------

        float: the line energy, if FWHM_MnKa is None
        (float,float): the line energy and the energy resolution, if FWHM_MnKa is not None
        """

        units_name = self.axes_manager.signal_axes[0].units

        if FWHM_MnKa == 'auto':
            if self.metadata.Signal.signal_type == 'EDS_SEM':
                FWHM_MnKa = self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa
            elif self.metadata.Signal.signal_type == 'EDS_TEM':
                FWHM_MnKa = self.metadata.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa
            else:
                raise NotImplementedError(
                    "This method only works for EDS_TEM or EDS_SEM signals. "
                    "You can use `set_signal_type(\"EDS_TEM\")` or"
                    "`set_signal_type(\"EDS_SEM\")` to convert to one of these"
                    "signal types.")
        line_energy = utils_eds._get_energy_xray_line(Xray_line)
        if units_name == 'eV':
            line_energy *= 1000
            if FWHM_MnKa is not None:
                line_FWHM = utils_eds.get_FWHM_at_Energy(FWHM_MnKa,
                                                         line_energy / 1000) * 1000
        elif units_name == 'keV':
            if FWHM_MnKa is not None:
                line_FWHM = utils_eds.get_FWHM_at_Energy(FWHM_MnKa,
                                                         line_energy)
        else:
            raise ValueError(
                "%s is not a valid units for the energy axis. "
                "Only `eV` and `keV` are supported. "
                "If `s` is the variable containing this EDS spectrum:\n "
                ">>> s.axes_manager.signal_axes[0].units = \'keV\' \n"
                % (units_name))
        if FWHM_MnKa is None:
            return line_energy
        else:
            return line_energy, line_FWHM

    def _get_beam_energy(self):
        """
        Get the beam energy.

        The return value is in the same units than the signal axis
        """

        if "Acquisition_instrument.SEM.beam_energy" in self.metadata:
            beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
        elif "Acquisition_instrument.TEM.beam_energy" in self.metadata:
            beam_energy = self.metadata.Acquisition_instrument.TEM.beam_energy
        else:
            raise AttributeError(
                "To use this method the beam energy `Acquisition_instrument.TEM.beam_energy` "
                "or `Acquisition_instrument.SEM.beam_energy` must be defined in "
                "`metadata`.")

        units_name = self.axes_manager.signal_axes[0].units

        if units_name == 'eV':
            beam_energy = beam_energy * 1000
        return beam_energy

    def sum(self, axis):
        """Sum the data over the given axis.

        Parameters
        ----------
        axis : {int, string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1).data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.sum(-1, True).plot()

        """
        # modify time spend per spectrum
        if "Acquisition_instrument.SEM" in self.metadata:
            mp = self.metadata.Acquisition_instrument.SEM
        else:
            mp = self.metadata.Acquisition_instrument.TEM
        if mp.has_item('Detector.EDS.live_time'):
            mp.Detector.EDS.live_time = mp.Detector.EDS.live_time * \
                self.axes_manager.shape[axis]
        return super(EDSSpectrum, self).sum(axis)

    def rebin(self, new_shape):
        """Rebins the data to the new shape

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape must be a divisor of the original shape

        """
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) /
                   np.array(new_shape_in_array))
        s = super(EDSSpectrum, self).rebin(new_shape)
        # modify time per spectrum
        if "Acquisition_instrument.SEM.Detector.EDS.live_time" in s.metadata:
            for factor in factors:
                s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time *= factor
        if "Acquisition_instrument.TEM.Detector.EDS.live_time" in s.metadata:
            for factor in factors:
                s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time *= factor
        return s

    def set_elements(self, elements):
        """Erase all elements and set them.

        Parameters
        ----------
        elements : list of strings
            A list of chemical element symbols.

        See also
        --------
        add_elements, set_line, add_lines.

        Examples
        --------

        >>> s = signals.EDSSEMSpectrum(np.arange(1024))
        >>> s.set_elements(['Ni', 'O'],['Ka','Ka'])
        Adding Ni_Ka Line
        Adding O_Ka Line
        >>> s.mapped_paramters.Acquisition_instrument.SEM.beam_energy = 10
        >>> s.set_elements(['Ni', 'O'])
        Adding Ni_La Line
        Adding O_Ka Line

        """
        # Erase previous elements and X-ray lines
        if "Sample.elements" in self.metadata:
            del self.metadata.Sample.elements
        self.add_elements(elements)

    def add_elements(self, elements):
        """Add elements and the corresponding X-ray lines.

        The list of elements is stored in `metadata.Sample.elements`

        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.


        See also
        --------
        set_elements, add_lines, set_lines.

        """
        if not isiterable(elements) or isinstance(elements, basestring):
            raise ValueError(
                "Input must be in the form of a list. For example, "
                "if `s` is the variable containing this EDS spectrum:\n "
                ">>> s.add_elements(('C',))\n"
                "See the docstring for more information.")
        if "Sample.elements" in self.metadata:
            elements_ = set(self.metadata.Sample.elements)
        else:
            elements_ = set()
        for element in elements:
            if element in elements_db:
                elements_.add(element)
            else:
                raise ValueError(
                    "%s is not a valid chemical element symbol." % element)

        if not hasattr(self.metadata, 'Sample'):
            self.metadata.add_node('Sample')

        self.metadata.Sample.elements = sorted(list(elements_))

    def set_lines(self,
                  lines,
                  only_one=True,
                  only_lines=("Ka", "La", "Ma")):
        """Erase all Xrays lines and set them.

        See add_lines for details.

        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb.
            Additionally, if `metadata.Sample.elements` is
            defined, add the lines of those elements that where not
            given in this list.
        only_one: bool
            If False, add all the lines of each element in
            `metadata.Sample.elements` that has not line
            defined in lines. If True (default),
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.

        See also
        --------
        add_lines, add_elements, set_elements..

        """
        if "Sample.xray_lines" in self.metadata:
            del self.metadata.Sample.xray_lines
        self.add_lines(lines=lines,
                       only_one=only_one,
                       only_lines=only_lines)

    def add_lines(self,
                  lines=(),
                  only_one=True,
                  only_lines=("Ka", "La", "Ma")):
        """Add X-rays lines to the internal list.

        Although most functions do not require an internal list of
        X-ray lines because they can be calculated from the internal
        list of elements, ocassionally it might be useful to customize the
        X-ray lines to be use by all functions by default using this method.
        The list of X-ray lines is stored in
        `metadata.Sample.xray_lines`

        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb.
            Additionally, if `metadata.Sample.elements` is
            defined, add the lines of those elements that where not
            given in this list. If the list is empty (default), and
            `metadata.Sample.elements` is
            defined, add the lines of all those elements.
        only_one: bool
            If False, add all the lines of each element in
            `metadata.Sample.elements` that has not line
            defined in lines. If True (default),
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.

        See also
        --------
        set_lines, add_elements, set_elements.

        """
        if "Sample.xray_lines" in self.metadata:
            xray_lines = set(self.metadata.Sample.xray_lines)
        else:
            xray_lines = set()
        # Define the elements which Xray lines has been customized
        # So that we don't attempt to add new lines automatically
        elements = set()
        for line in xray_lines:
            elements.add(line.split("_")[0])
        end_energy = self.axes_manager.signal_axes[0].high_value
        for line in lines:
            try:
                element, subshell = line.split("_")
            except ValueError:
                raise ValueError(
                    "Invalid line symbol. "
                    "Please provide a valid line symbol e.g. Fe_Ka")
            if element in elements_db:
                elements.add(element)
                if subshell in elements_db[element]['Atomic_properties']['Xray_lines']:
                    lines_len = len(xray_lines)
                    xray_lines.add(line)
                    # if lines_len != len(xray_lines):
                    #    print("%s line added," % line)
                    # else:
                    #    print("%s line already in." % line)
                    if (self._get_line_energy(element + '_' + subshell) > end_energy):
                        print("Warning: %s %s is above the data energy range."
                              % (element, subshell))
                else:
                    raise ValueError(
                        "%s is not a valid line of %s." % (line, element))
            else:
                raise ValueError(
                    "%s is not a valid symbol of an element." % element)
        if "Sample.elements" in self.metadata:
            extra_elements = (set(self.metadata.Sample.elements) -
                              elements)
            if extra_elements:
                new_lines = self._get_lines_from_elements(
                    extra_elements,
                    only_one=only_one,
                    only_lines=only_lines)
                if new_lines:
                    self.add_lines(list(new_lines) + list(lines))
        self.add_elements(elements)
        if not hasattr(self.metadata, 'Sample'):
            self.metadata.add_node('Sample')
        if "Sample.xray_lines" in self.metadata:
            xray_lines = xray_lines.union(
                self.metadata.Sample.xray_lines)
        self.metadata.Sample.xray_lines = sorted(list(xray_lines))

    def _get_lines_from_elements(self,
                                 elements,
                                 only_one=False,
                                 only_lines=("Ka", "La", "Ma")):
        """Returns the X-ray lines of the given elements in spectral range
        of the data.

        Parameters
        ----------
        elements : list of strings
            A list containing the symbol of the chemical elements.
        only_one : bool
            If False, add all the lines of each element in the data spectral
            range. If True only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be returned.


        Returns
        -------

        """

        beam_energy = self._get_beam_energy()

        #end_energy = self.axes_manager.signal_axes[0].high_value
        end_energy = self.axes_manager[-1].high_value
        if beam_energy < end_energy:
            end_energy = beam_energy
        lines = []
        for element in elements:
            # Possible line (existing and excited by electron)
            element_lines = []
            for subshell in elements_db[element]['Atomic_properties']['Xray_lines'].keys():
                if only_lines and subshell not in only_lines:
                    continue
                if (self._get_line_energy(element + '_' + subshell) < end_energy):
                    element_lines.append(element + "_" + subshell)
            if only_one and element_lines:
            # Choose the best line
                select_this = -1
                for i, line in enumerate(element_lines):
                    if (self._get_line_energy(line) < beam_energy / 2):
                        select_this = i
                        break
                element_lines = [element_lines[select_this], ]

            if not element_lines:
                print(("There is not X-ray line for element %s " % element) +
                      "in the data spectral range")
            else:
                lines.extend(element_lines)
        return lines

    # suppress lines_deconvolution="model"
    # suppress standard? add option in "edsmodel"
    def get_lines_intensity(self,
                            xray_lines=None,
                            plot_result=False,
                            integration_window_factor=2.,
                            only_one=True,
                            only_lines=("Ka", "La", "Ma"),
                            lines_deconvolution=None,
                            bck=0,
                            plot_fit=False,
                            store_in_mp=False,
                            bounded=False,
                            grad=False,
                            init=True,
                            return_model=False,
                            **kwargs):
        """Return the intensity map of selected Xray lines.

        The intensities, the number of X-ray counts, are computed by
        suming the spectrum over the
        different X-ray lines. The sum window width
        is calculated from the energy resolution of the detector
        defined as defined in
        `self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa` or
        `self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa`.


        Parameters
        ----------

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
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
        integration_window_factor: Float
            The integration window is centered at the center of the X-ray
            line and its width is defined by this factor (2 by default)
            times the calculated FWHM of the line.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, use only the given lines.
        lines_deconvolution : None or 'model' or 'standard' or 'top_hat'
            Deconvolution of the line with a gaussian model. Take time
        bck : float
            background to substract. Only for deconvolution
        store_in_mp : bool
            store the result in metadata.Sample
        bounded: bool
            force positive fit, fast with PCA
        grad: bool
            fit option, fast with PCA
        init: bool
            initialize value
        return_model: bool
            return the model instead of the intensities
        kwargs
            The extra keyword arguments for plotting. See
            `utils.plot.plot_signals`

        Returns
        -------
        intensities : list
            A list containing the intensities as Signal subclasses.

        Examples
        --------

        >>> pyplot.set_cmap('RdYlBu_r')

        #Mode standard

        >>> s = database.spec3D('SEM')
        >>> s[102:134,125:152].get_lines_intensity(["Hf_Ma", "Ta_Ma"],
                plot_result=True)

        #Mode 'model'

        >>> s = database.spec3D('SEM')
        >>> s[102:134,125:152].get_lines_intensity(["Hf_Ma", "Ta_Ma"],
                plot_result=True,lines_deconvolution='model',plot_fit=True)

        #Mode 'standard'

        >>> s = database.spec3D('SEM')
        >>> from hyperspy.misc.config_dir import config_path
        >>> s.add_elements(['Hf','Ta'])
        >>> s.link_standard(config_path+'/database/std_RR')
        >>> s[102:134,125:152].get_lines_intensity(
                plot_result=True,lines_deconvolution='standard')

        See also
        --------

        set_elements, add_elements.

        """

        from hyperspy.hspy import create_model

        if xray_lines is None:
            if 'Sample.xray_lines' in self.metadata:
                xray_lines = self.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.metadata:
                xray_lines = self._get_lines_from_elements(
                    self.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "Not X-ray line, set them with `add_elements`")

        intensities = [0] * len(xray_lines)
        if lines_deconvolution == 'standard':
            m = create_model(self, auto_background=False,
                             auto_add_lines=False)
        elif lines_deconvolution == 'model':
            s = self - bck
            m = create_model(s, auto_background=False,
                             auto_add_lines=False)

        for i, xray_line in enumerate(xray_lines):
            line_energy, line_FWHM = self._get_line_energy(xray_line,
                                                           FWHM_MnKa='auto')
            element, line = utils_eds._get_element_and_line(xray_line)
            det = integration_window_factor * line_FWHM / 2.
            if lines_deconvolution is None:
                intensities[i] = self[..., line_energy - det:line_energy +
                                      det].integrate1D(-1).data
            elif lines_deconvolution == 'top_hat':
                intensities[i] = self.top_hat(line_energy
                                              ).integrate1D(-1).data
            else:
                if lines_deconvolution == 'model':
                    fp = create_component.Gaussian()
                    fp.centre.value = line_energy
                    fp.sigma.value = line_FWHM / 2.355
                    fp.centre.free = False
                    fp.sigma.free = False
                    if bounded:
                        fp.A.ext_bounded = True
                        fp.A.ext_force_positive = True
                elif lines_deconvolution == 'standard':
                    std = self.get_result(element, 'standard_spec').deepcopy()
                    std[:line_energy - det] = 0
                    std[line_energy + det:] = 0
                    fp = create_component.ScalableFixedPattern(std)
                    fp.set_parameters_not_free(['offset', 'xscale', 'shift'])
                    if bounded:
                        fp.yscale.ext_bounded = True
                        fp.yscale.ext_force_positive = True
                fp.name = xray_line
                m.append(fp)
                if init:
                    if lines_deconvolution == 'standard':
                        m[xray_line].yscale.map[
                            'values'] = self[..., line_energy].data
                        m[xray_line].yscale.map['is_set'] = (
                            np.ones(self[..., line_energy].data.shape) == 1)
                    elif lines_deconvolution == 'model':
                        m[xray_line].A.map[
                            'values'] = self[..., line_energy].data
                        m[xray_line].A.map['is_set'] = (
                            np.ones(self[..., line_energy].data.shape) == 1)
                # Other line of the family as twin
                if lines_deconvolution == 'model':
                    for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                        if line[0] in li and line != li:
                            xray_sub = element + '_' + li
                            line_energy, line_FWHM = self._get_line_energy(
                                xray_sub, FWHM_MnKa='auto')
                            fp_sub = create_component.Gaussian()
                            fp_sub.centre.value = line_energy
                            fp_sub.name = xray_sub
                            fp_sub.sigma.value = line_FWHM / 2.355
                            fp_sub.A.twin = fp.A
                            fp_sub.centre.free = False
                            fp_sub.sigma.free = False
                            fp_sub.A.twin_function = _get_weight(element, li)
                            fp_sub.A.twin_inverse_function = _get_iweight(
                                element, li)
                            m.append(fp_sub)
        if lines_deconvolution == 'model' or lines_deconvolution == 'standard':
            m.multifit(fitter='leastsq', grad=grad)
            if plot_fit:
                m.plot()
                plt.title('Fit')
        # data as image, store and plot
        for i, xray_line in enumerate(xray_lines):
            line_energy = self._get_line_energy(xray_line)
            if lines_deconvolution == 'model':
                data_res = m[xray_line].A.map['values']
                if self.axes_manager.navigation_dimension == 0:
                    data_res = data_res[0]
            elif lines_deconvolution == 'standard':
                data_res = m[xray_line].yscale.map['values']
                if self.axes_manager.navigation_dimension == 0:
                    data_res = data_res[0]
            else:
                data_res = intensities[i]

            img = self._set_result(xray_line, 'intensities',
                                   data_res, plot_result=False,
                                   store_in_mp=store_in_mp)
            img.metadata.General.title = (
                'Intensity of %s at %.2f %s from %s' %
                (xray_line,
                 line_energy,
                 self.axes_manager.signal_axes[0].units,
                 self.metadata.General.title))
            if plot_result and img.axes_manager.signal_dimension == 0:
                print("%s at %s %s : Intensity = %.2f"
                      % (xray_line,
                         line_energy,
                         self.axes_manager.signal_axes[0].units,
                         img.data))
            intensities[i] = img
        if plot_result and img.axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(intensities, **kwargs)

        if return_model:
            return m
        else:
            return intensities

    def convolve_sum(self, kernel='square', size=3, **kwargs):
        """
        Apply a running sum on the x,y dimension of a spectrum image

        Parameters
        ----------

        kernel: 'square' or 2D array
            Define the kernel

        size : int,optional
            if kernel = square, defined the size of the square
        """
        import scipy.ndimage
        if kernel == 'square':
            #[[1/float(size*size)]*size]*size
            kernel = [[1] * size] * size

        result = self.deepcopy()
        result = result.as_image([0, 1])
        result.map(scipy.ndimage.filters.convolve,
                   weights=kernel, **kwargs)
        result = result.as_spectrum(0)

        if hasattr(result.metadata.Acquisition_instrument, 'SEM'):
            mp = result.metadata.Acquisition_instrument.SEM
        else:
            mp = result.metadata.Acquisition_instrument.TEM
        if 'Detector.EDS.live_time' in mp:
            mp.Detector.EDS.live_time = mp.Detector.EDS.live_time * \
                np.sum(kernel)

        return result

    def get_result(self, xray_line, result):
        """
        get the result of one X-ray line (result stored in
        'metadata.Sample'):

         Parameters
        ----------
        result : string {'kratios'|'quant'|'intensities'}
            The result to get

        xray_lines: string
            the X-ray line to get.

        """
        mp = self.metadata
        for res in mp.Sample[result]:
            if xray_line in res.metadata.General.title:
                return res
        raise ValueError("Didn't find it")

#_get_navigation_signal do a great job, should use it
    def _set_result(self, xray_line, result, data_res, plot_result,
                    store_in_mp=True):
        """
        Transform data_res (a result) into an image or a signal and
        stored it in 'metadata.Sample'
        """

        mp = self.metadata
        if mp.has_item('Sample'):
            if mp.Sample.has_item('xray_lines'):
                if len(xray_line) < 3:
                    xray_lines = mp.Sample.elements
                else:
                    xray_lines = mp.Sample.xray_lines

                for j in range(len(xray_lines)):
                    if xray_line == xray_lines[j]:
                        break

        axes_res = self.axes_manager.deepcopy()
        axes_res.remove(-1)

        if self.axes_manager.navigation_dimension == 0:
            res_img = Signal(np.array(data_res))
        else:
            res_img = Signal(data_res)
            res_img.axes_manager = axes_res
            if self.axes_manager.navigation_dimension == 1:
                res_img = res_img.as_spectrum(0)
            else:
                res_img = res_img.as_image([0, 1])
        res_img.metadata.General.title = result + ' ' + xray_line
        if plot_result:
            if self.axes_manager.navigation_dimension == 0:
                # to be changed with new version
                print("%s of %s : %s" % (result, xray_line, data_res))
            else:
                res_img.plot(None)
        # else:
        #    print("%s of %s calculated" % (result, xray_line))

        res_img.get_dimensions_from_data()
        if store_in_mp:
            if result not in mp.Sample:
                mp.set_item('Sample.' + result, [0] * len(xray_lines))
            mp.Sample[result][j] = res_img

        return res_img

    def normalize_result(self, result, return_element='all'):
        """
        Normalize the result

        The sum over all elements for any pixel is equal to one.

        Paramters
        ---------

        result: str
            the result to normalize

        return_element: str
            If 'all', all elements are return.
        """
        # look at dim...
        mp = self.metadata
        res = copy.deepcopy(mp.Sample[result])

        re = utils.stack(res)
        if re.axes_manager.signal_dimension == 0:
            tot = re.sum(1)
            for r in range(re.axes_manager.shape[1]):
                res[r].data = (re[::, r] / tot).data
        elif re.axes_manager.signal_dimension == 1:
            tot = re.sum(0)
            for r in range(re.axes_manager.shape[0]):
                res[r].data = (re[r] / tot).data
        else:
            tot = re.sum(1)
            for r in range(re.axes_manager.shape[1]):
                res[r].data = (re[::, r] / tot).data

        if return_element == 'all':
            return res
        else:
            for el in res:
                if return_element in el.metadata.General.title:
                    return el

    def plot_histogram_result(self,
                              result,
                              bins='freedman',
                              color=None,
                              legend='auto',
                              line_style=None,
                              fig=None,
                              **kwargs):
        """
        Plot an histrogram of the result

        Paramters
        ---------

        result: str
            the result to plot

        bins : int or list or str (optional)
            If bins is a string, then it must be one of:
            'knuth' : use Knuth's rule to determine bins
            'scotts' : use Scott's rule to determine bins
            'freedman' : use the Freedman-diaconis rule to determine bins
            'blocks' : use bayesian blocks for dynamic bin widths

        color : valid matplotlib color or a list of them or `None`
            Sets the color of the lines of the plots when `style` is "cascade"
            or "mosaic". If a list, if its length is
            less than the number of spectra to plot, the colors will be cycled. If
            If `None`, use default matplotlib color cycle.

        line_style: valid matplotlib line style or a list of them or `None`
            Sets the line style of the plots for "cascade"
            or "mosaic". The main line style are '-','--','steps','-.',':'.
            If a list, if its length is less than the number of
            spectra to plot, line_style will be cycled. If
            If `None`, use continuous lines, eg: ('-','--','steps','-.',':')

        legend: None | list of str | 'auto'
           If list of string, legend for "cascade" or title for "mosaic" is
           displayed. If 'auto', the title of each spectra (metadata.General.title)
           is used.

        fig : {matplotlib figure, None}
            If None, a default figure will be created.
        """
        mp = self.metadata
        res = copy.deepcopy(mp.Sample[result])

        return plot_histograms(res, bins=bins, legend=legend,
                               color=color,
                               line_style=line_style, fig=fig,
                               **kwargs)

# should use plot ortho_animate
    def plot_orthoview_result(self,
                              element,
                              result,
                              isotropic_voxel=True,
                              normalize=False):
        """
        Plot an orthogonal view of a 3D images

        Parameters
        ----------

        element: str
            The element to get.

        result: str
            The result to get

        isotropic_voxel:
            If True, generate a new image, scaling z in order to obtain isotropic
            voxel.
        """
        if element == 'all':
            res_element = copy.deepcopy(self.metadata.Sample[result])
            res_element = utils.stack(res_element).sum(1)
        elif normalize:
            self.deepcopy()
            res_element = self.normalize_result(result, return_element=element)
        else:
            res_element = self.get_result(element, result)
        fig = image_eds.plot_orthoview_animated(res_element, isotropic_voxel)

        return fig

    def add_poissonian_noise(self, **kwargs):
        """Add Poissonian noise to the data"""
        original_type = self.data.dtype
        self.data = np.random.poisson(self.data, **kwargs).astype(
            original_type)

    def get_take_off_angle(self, tilt_stage='auto',
                           azimuth_angle='auto',
                           elevation_angle='auto'):
        """Calculate the take-off-angle (TOA).

        TOA is the angle with which the X-rays leave the surface towards
        the detector. Parameters are read in 'SEM.tilt_stage',
        'Acquisition_instrument.SEM.Detector.EDS.azimuth_angle' and 'SEM.Detector.EDS.elevation_angle'
         in 'metadata'.

        Returns
        -------
        take_off_angle: float (Degree)

        See also
        --------
        utils.eds.take_off_angle

        Notes
        -----
        Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)
        """
        if self.metadata.Signal.signal_type == 'EDS_SEM':
            mp = self.metadata.Acquisition_instrument.SEM
        elif self.metadata.Signal.signal_type == 'EDS_TEM':
            mp = self.metadata.Acquisition_instrument.TEM

        if tilt_stage == 'auto'and 'tilt_stage' in mp:
            tilt_stage = mp.tilt_stage
        if azimuth_angle == 'auto'and 'azimuth_angle' in mp.Detector.EDS:
            azimuth_angle = mp.Detector.EDS.azimuth_angle
        if elevation_angle == 'auto'and 'elevation_angle' in mp.Detector.EDS:
            elevation_angle = mp.Detector.EDS.elevation_angle

        TOA = utils.eds.take_off_angle(tilt_stage, azimuth_angle,
                                       elevation_angle)
        return TOA

# Other color for lines set

    def plot_xray_lines(self,
                        xray_lines=None,
                        only_lines=("a", "b"),
                        only_one=False,
                        **kwargs):
        """
        Annotate a spec.plot() with the name of the selected X-ray
        lines

        Parameters
        ----------
        xray_lines: {None, 'from_elements', list of string}
            If None,
            if `mapped.parameters.Sample.elements.xray_lines` contains a
            list of lines use those.
            If `mapped.parameters.Sample.elements.xray_lines` is undefined
            or empty or if xray_lines equals 'from_elements' and
            `mapped.parameters.Sample.elements` is defined,
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
            Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        only_lines : None or list of strings
            If not None, use only the given lines (eg. ('a','Kb')).
            If None, use all lines.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        kwargs
            The extra keyword arguments for plot()


        Examples
        --------

        >>> s = database.spec3D('Ti_SEM')
        >>> s.plot_Xray_lines()

        >>> s.plot_Xray_lines('from_elements')

        See also
        --------
        set_elements, add_elements

        """

        if only_lines is not None:
            only_lines = list(only_lines)
            for only_line in only_lines:
                if only_line == 'a':
                    only_lines.extend(['Ka', 'La', 'Ma'])
                elif only_line == 'b':
                    only_lines.extend(['Kb', 'Lb1', 'Mb'])

        if xray_lines is None or xray_lines == 'from_elements':
            if 'Sample.xray_lines' in self.metadata \
                    and xray_lines != 'from_elements':
                xray_lines = self.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.metadata:
                xray_lines = self._get_lines_from_elements(
                    self.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "No elements defined, set them with `add_elements`")

        line_energy = []
        intensity = []
        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy.append(self._get_line_energy(xray_line))
            relative_factor = elements_db[element][
                'Atomic_properties']['Xray_lines'][line]['weight']
            a_eng = self._get_line_energy(element + '_' + line[0] + 'a')
            # if fixed_height:
                # intensity.append(self[..., a_eng].data.flatten().mean()
                             #* relative_factor)
            # else:
            intensity.append(self[..., a_eng].data * relative_factor)

        self.plot(**kwargs)
        for i in range(len(line_energy)):
            line = marker.Marker()
            line.type = 'line'
            line.orientation = 'v'
            line.set_data(x1=line_energy[i], y2=intensity[i] * 0.8)
            self._plot.signal_plot.add_marker(line)
            line.plot()
            text = marker.Marker()
            text.type = 'text'
            text.set_marker_properties(rotation=90)
            text.set_data(x1=line_energy[i],
                          y1=intensity[i] * 1.1, text=xray_lines[i])
            self._plot.signal_plot.add_marker(text)
            text.plot()

    def get_decomposition_model_from(self,
                                     binned_signal,
                                     components,
                                     loadings_as_guess=True,
                                     **kwargs):
        """
        Return the spectrum generated with the selected number of principal
        components from another spectrum (binned_signal).

        The selected components are fitted on the spectrum

        Parameters
        ------------

        binned_signal: signal
            The components of the binned_signal are fitted to self.
            The dimension must have a common multiplicity.

        components :  int or list of ints
             if int, rebuilds SI from components in range 0-given int
             if list of ints, rebuilds SI from only components in given list
        kwargs
            keyword argument for multifit

        Returns
        -------
        Signal instance

        Example
        -------

        Quick example

        >>> s = database.spec3D()
        >>> s.change_dtype('float')
        >>> s = s[:6,:8]
        >>> s2 = s.deepcopy()
        >>> dim = s.axes_manager.shape
        >>> s2 = s2.rebin((dim[0]/2, dim[1]/2, dim[2]))
        >>> s2.decomposition(True)
        >>> a = s.get_decomposition_model_from(s2, components=5)

        Slower that makes sense

        >>> s = database.spec4D('TEM')[::,::,1]
        >>> s.change_dtype('float')
        >>> dim = s.axes_manager.shape
        >>> s = s.rebin((dim[0]/4, dim[1]/4, dim[2]))
        >>> s2 = s.deepcopy()
        >>> s2 = s2.rebin((dim[0]/8, dim[1]/8, dim[2]))
        >>> s2.decomposition(True)
        >>> a = s.get_decomposition_model_from(s2, components=5)

        """
        from hyperspy.hspy import create_model
        if hasattr(binned_signal, 'learning_results') is False:
            raise ValueError(
                "binned_signal must be decomposed")

        if isinstance(components, int):
            components = range(components)

        m = create_model(self, auto_background=False,
                         auto_add_lines=False)
        factors = binned_signal.get_decomposition_factors()
        if loadings_as_guess:
            loadings = binned_signal.get_decomposition_loadings()
            dim_bin = np.array(binned_signal.axes_manager.navigation_shape)
            dim = np.array(self.axes_manager.navigation_shape)
            bin_fact = dim / dim_bin
            if np.all([isinstance(bin_f, int)
                       for bin_f in bin_fact]) is False:
                raise ValueError(
                    "The dimension of binned_signal doesn't not result"
                    "from a binning")
        for i_comp in components:
            fp = create_component.ScalableFixedPattern(factors[i_comp])
            fp.set_parameters_not_free(['offset', 'xscale', 'shift'])
            fp.name = str(i_comp)
            if loadings_as_guess:
                load_data = loadings[i_comp].data
                for i, bin_f in enumerate(bin_fact[::-1]):
                    load_data = np.repeat(load_data, bin_f, axis=i)
            m.append(fp)
            if loadings_as_guess:
                m[str(i_comp)].yscale.map['values'] = load_data
                m[str(i_comp)].yscale.map['is_set'] = [[True] * dim[0]] * \
                    dim[1]
                #(np.ones(self[...,line_energy].data.shape)==1)
        m.multifit(fitter='leastsq', **kwargs)

        return m.as_signal()
    # to be improved, get line energy, FHWM, output...

    def top_hat(self, line_energy, width_windows=1.):
        """
        Substact the background with a top hat filter. The width of the
        lobs are defined with the width of the peak at the line_energy.

        Parameters
        ----------------
        line_energy: float
            The energy in keV used to set the lob width calculate with
            FHWM_eds.

        width_windows: float or list(min,max)
            The width of the windows on which is applied the top_hat.
            By default set to 1, which is equivalent to the size of the
            filtering object.

        Notes
        -----
        See the textbook of Goldstein et al., Plenum publisher,
        third edition p 399

        """
        offset = np.copy(self.axes_manager.signal_axes[0].offset)
        scale_s = np.copy(self.axes_manager.signal_axes[0].scale)
        #FWHM_MnKa = self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa
        if self.metadata.Signal.signal_type == 'EDS_SEM':
            FWHM_MnKa = self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa
        elif self.metadata.Signal.signal_type == 'EDS_TEM':
            FWHM_MnKa = self.metadata.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa
        line_FWHM = utils_eds.get_FWHM_at_Energy(FWHM_MnKa, line_energy)
        if np.ndim(width_windows) == 0:
            det = [width_windows * line_FWHM, width_windows * line_FWHM]
        else:
            det = width_windows

        olob = int(round(line_FWHM / scale_s / 2) * 2)
        g = []
        for lob in range(-olob, olob):
            if abs(lob) > olob / 2:
                g.append(-1. / olob)
            else:
                g.append(1. / (olob + 1))
        g = np.array(g)

        bornA = [int(round((line_energy - det[0] - offset) / scale_s)),
                 int(round((line_energy + det[1] - offset) / scale_s))]

        data_s = []
        for i in range(bornA[0], bornA[1]):
            data_s.append(self.data[..., i - olob:i + olob].dot(g))
            # data_s.append(self.data[...,i-olob:i+olob])
        data_s = np.array(data_s)

        dim = len(self.data.shape)
        #spec_th = EDSSEMSpectrum(np.rollaxis(data_s.dot(g),0,dim))

        spec_th = Spectrum(np.rollaxis(data_s, 0, dim))

        return spec_th

    def get_sample_density(self, weight_fraction='auto'):
        """Return the density of the sample

        Parameters
        ----------
        weight_fraction: {list of float| 'auto'}
            the composition of the sample
            if 'auto'. looks for the weight fraction in metadata
            if not there take the iso concentration

        Return
        ------
        density in g/cm^3
        """
        from hyperspy import signals
        elements = self.metadata.Sample.elements
        if weight_fraction == 'auto':
            if 'weight_fraction' in self.metadata.Sample:
                weight_fraction = self.metadata.Sample.weight_fraction
            else:
                weight_fraction = []
                for elm in elements:
                    weight_fraction.append(1. / len(elements))
                print 'Weight fraction is automatically set to ' + str(
                    weight_fraction)
        if isinstance(weight_fraction[0], signals.Signal):
            weight_frac = []
            for weight in weight_fraction:
                weight_frac.append(weight.data)
            density = utils.material.density_of_mixture_of_pure_elements(
                elements, weight_frac)
        else:
            density = utils.material.density_of_mixture_of_pure_elements(
                elements, weight_fraction)
        self.metadata.Sample.density = density
        return density

    def get_mass_absorption_coefficient_sample(self,
                       xray_lines='auto',
                       weight_fraction='auto',
                       elements='auto'):
        """Return the mass absorption coefficients of for the different
        xray in a sample

        Parameters
        ----------
        xray_lines: list of str
            The list of X-ray lines, e.g. ['Al_Ka','Zn_Ka','Zn_La']
        weight_fraction: list of float
            the composition of the sample
        elements: {list of str | 'auto'}
            The list of element symbol of the absorber, e.g. ['Al','Zn'].
            if 'auto', use the elements of the X-ray lines

        Return
        ------
        mass absorption coefficient in cm^2/g
        """

        if xray_lines == 'auto':
            if 'Sample.xray_lines' in self.metadata:
                xray_lines = copy.copy(self.metadata.Sample.xray_lines)
            else : 
                raise ValueError("Add lines first, see 'add_lines'")

        if elements == 'auto'and 'Sample.elements' in self.metadata:
            elements = self.metadata.Sample.elements
        if weight_fraction == 'auto':
            if 'Sample.weight_fraction' in self.metadata:
                weight_fraction = self.metadata.Sample.weight_fraction
            else:
                weight_fraction = []
                for elm in elements:
                    weight_fraction.append(1. / len(elements))
                print 'Weight fraction is automatically set to ' + str(weight_fraction)
        return utils_eds.get_mass_absorption_coefficient_sample(energies=xray_lines,
                                        weight_fraction=weight_fraction, elements=elements)

    def save(self, filename=None, overwrite=None, extension=None,
             **kwds):
        """Saves the signal in the specified format.

        The function gets the format from the extension.:
            - hdf5 for HDF5
            - rpl for Ripple (useful to export to Digital Micrograph)
            - msa for EMSA/MSA single spectrum saving.
            - Many image formats such as png, tiff, jpeg...

        If no extension is provided the default file format as defined
        in the `preferences` is used.
        Please note that not all the formats supports saving datasets of
        arbitrary dimensions, e.g. msa only suports 1D data.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None (default) and tmp_parameters.filename and
            `tmp_paramters.folder` are defined, the
            filename and path will be taken from there. A valid
            extension can be provided e.g. "my_file.rpl", see `extension`.
        overwrite : None, bool
            If None, if the file exists it will query the user. If
            True(False) it (does not) overwrites the file if it exists.
        extension : {None, 'hdf5', 'rpl', 'msa',common image extensions e.g. 'tiff', 'png'}
            The extension of the file that defines the file format.
            If None, the extesion is taken from the first not None in the follwoing list:
            i) the filename
            ii)  `tmp_parameters.extension`
            iii) `preferences.General.default_file_format` in this order.
        """
        mp = self.metadata

        if hasattr(mp, 'Sample'):
            if hasattr(mp.Sample, 'standard_spec'):
                l_time = []
                for el in range(len(mp.Sample.elements)):
                # for el in range(len(mp.Sample.xray_lines)):
                    std = mp.Sample.standard_spec[el]
                    if "Acquisition_instrument.SEM" in std.metadata:
                        microscope = std.metadata.Acquisition_instrument.SEM
                    elif "Acquisition_instrument.TEM" in std.metadata:
                        microscope = std.metadata.Acquisition_instrument.TEM
                    l_time.append(microscope.Detector.EDS.live_time)
                std_store = copy.deepcopy(mp.Sample.standard_spec)
                std = utils.stack(std_store)
                std.metadata.General.title = std_store[
                    0].metadata.General.title
                if "Acquisition_instrument.SEM" in std.metadata:
                    std.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = l_time
                elif "Acquisition_instrument.TEM" in std.metadata:
                    std.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time = l_time
                mp.Sample.standard_spec = std
            result_store = []
            for result in ['kratios', 'quant', 'quant_enh', 'intensities']:
                if hasattr(mp.Sample, result):
                    result_store.append(copy.deepcopy(mp.Sample[result]))
                    mp.Sample[result] = utils.stack(mp.Sample[result])
                    del mp.Sample[result].original_parameters.stack_elements

        super(EDSSpectrum, self).save(filename, overwrite, extension)

        if hasattr(mp, 'Sample'):
            if hasattr(mp.Sample, 'standard_spec'):
                mp.Sample.standard_spec = std_store
            i = 0
            for result in ['kratios', 'quant', 'quant_enh', 'intensities']:
                if hasattr(mp.Sample, result):
                    mp.Sample[result] = result_store[i]
                    i = i + 1

    def compute_continuous_xray_generation(self, generation_factor=1):
        """Continous X-ray generation.

        Kramer or Lisfshin equation

        Parameters
        ----------
        generation_factor: int
            The power law to use.
            1 si equivalent to Kramer equation.
            2 is equivalent to Lisfhisn modification of Kramer equation.
        beam_energy:  float
            The energy of the electron beam

        See also
        --------
        utils.misc.eds.model.continuous_xray_generation
        edsmodel.add_background
        """

        beam_energy = self._get_beam_energy()
        spec = self.deepcopy()
        for ax in self.axes_manager.navigation_axes:
            spec = spec[0]
        eng = mlab.frange(spec.axes_manager.signal_axes[0].low_value,
                          spec.axes_manager.signal_axes[0].high_value,
                          spec.axes_manager.signal_axes[0].scale)
        spec.data = model_eds.continuous_xray_generation(energy=eng,
                                                         generation_factor=generation_factor,
                                                         beam_energy=beam_energy)
        return spec

    def compute_continuous_xray_absorption(self,
                                           weight_fraction='auto'):
        """Contninous X-ray Absorption within sample

        PDH equation (Philibert-Duncumb-Heinrich)

        Parameters
        ----------
        weight_fraction: list of float
            The sample composition. If 'auto', takes value in metadata.
            If not there, use and equ-composition

        See also
        --------
        utils.misc.eds.model.continuous_xray_absorption
        edsmodel.add_background
        """

        beam_energy = self._get_beam_energy()
        elements = self.metadata.Sample.elements
        TOA = self.get_take_off_angle()
        units_name = self.axes_manager.signal_axes[0].units
        if weight_fraction == 'auto':
            if 'weight_fraction' in self.metadata.Sample:
                weight_fraction = self.metadata.Sample.weight_fraction
            else:
                weight_fraction = []
                for elm in elements:
                    weight_fraction.append(1. / len(elements))

        spec = self.deepcopy()
        for ax in self.axes_manager.navigation_axes:
            spec = spec[0]
        eng = mlab.frange(spec.axes_manager.signal_axes[0].low_value,
                          spec.axes_manager.signal_axes[0].high_value,
                          spec.axes_manager.signal_axes[0].scale)
        eng = eng[np.searchsorted(eng, 1e-10):]
        spec.data = np.append(np.array([0] * (len(spec.data) - len(eng))),
                              model_eds.continuous_xray_absorption(energy=eng,
                                                                   weight_fraction=weight_fraction,
                                                                   elements=elements,
                                                                   beam_energy=beam_energy,
                                                                   TOA=TOA,
                                                                   units_name=units_name))
        # for i, en in enumerate(eng):
            # if en <= 0:
                #spec.data[i] = 0.
            # else:
                # break

        return spec
    # doesn't work if scale too small

    def get_detector_efficiency(self,
                                detector_name,
                                gateway='auto'):
        """
        Return the detector efficiency.

        From DTSA II or from a database

        Parameters
        ----------
        det_name: int, str
            If {0,1,2,3,4}, INCA efficiency database
            If str, model from DTSAII
        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.

        See also
        --------
        database.detector_efficiency_INCA
        utils_eds.get_detector_properties
        """
        ax_s = self.axes_manager[-1]
        if isinstance(detector_name, str):
            if gateway == 'auto':
                gateway = utils_eds.get_link_to_jython()
            det_efficiency = utils_eds.get_detector_properties(
                detector_name, gateway=gateway)
        else:
            det_efficiency = database.detector_efficiency_INCA(detector_name)
        det_efficiency = det_efficiency[
            ax_s.low_value:ax_s.high_value +
            ax_s.scale]
        ax_det = det_efficiency.axes_manager[-1]

        if ax_s.low_value < ax_det.low_value:
            det_efficiency.data = np.append(np.array([0] *
                                                     (ax_s.size - ax_det.size)), det_efficiency.data)
            det_efficiency.get_dimensions_from_data()
            ax_det.offset = ax_s.offset

        return det_efficiency

# to be suppress, can be done with model->fit_energy_resolution

    # def calibrate_energy_resolution(self, xray_line, bck='auto',
                                    # set_Mn_Ka=True, model_plot=True):
        #"""
        # Calibrate the energy resolution from a peak

        # Estimate the FHWM of the peak, estimate the energy resolution and
        # extrapolate to FWHM of Mn Ka

        # Parameters:
        # xray_line : str
            # the selected X-ray line. It shouldn't have peak around

        # bck: float | 'auto'
            # the linear background to substract.

        # set_Mn_Ka : bool
            # If true, set the obtain resolution. If false, return the
            # FHWM at Mn Ka.

        # model_plot : bool
            # If True, plot the fit

        #"""

        #from hyperspy.hspy import create_model
        #mp = self.metadata
        #element, line = utils_eds._get_element_and_line(xray_line)
        # Xray_energy, FWHM = self._get_line_energy(xray_line,
                                                  # FWHM_MnKa='auto')

        # if bck == 'auto':
            #spec_bck = self[Xray_energy + 2.5 * FWHM:Xray_energy + 2.7 * FWHM]
            #bck = spec_bck.sum(0).data / spec_bck.axes_manager.shape[0]
        #sb = self - bck
        # m = create_model(sb,auto_background=False,
                 # auto_add_lines=False)

        #fp = create_component.Gaussian()
        #fp.centre.value = Xray_energy
        #fp.sigma.value = FWHM / 2.355
        # m.append(fp)
        #m.set_signal_range(Xray_energy - 1.2 * FWHM, Xray_energy + 1.6 * FWHM)
        # m.multifit()
        # if model_plot:
            # m.plot()

        # res_MnKa = utils_eds.get_FWHM_at_Energy(fp.sigma.value * 2.355 * 1000,
                                                # elements_db['Mn'][
                                                    #'Atomic_properties']['Xray_lines'][
                                                    #'Ka']['energy (keV)'], xray_line)
        # if set_Mn_Ka:
            #mp.SEM.Detector.EDS.energy_resolution_MnKa = res_MnKa * 1000
            # print 'Resolution at Mn Ka ', res_MnKa * 1000
            # print 'Shift eng eV ', (Xray_energy - fp.centre.value) * 1000
        # else:
            # return res_MnKa * 1000
############################
    # def running_sum(self, shape_convo='square', corner=-1):
        # cross not tested
        #"""
        # Apply a running sum on the data.
        # Parameters
        #----------
        # shape_convo: 'square'|'cross'
            # Define the shape to convolve with
        # corner : -1 || 1
            # For square, running sum induce a shift of the images towards
            # one of the corner: if -1, towards top left, if 1 towards
            # bottom right.
            # For 'cross', if -1 vertical/horizontal cross, if 1 from corner
            # to corner.
        #"""
        #dim = self.data.shape
        #data_s = np.zeros_like(self.data)
        #data_s = np.insert(data_s, 0, 0, axis=-3)
        #data_s = np.insert(data_s, 0, 0, axis=-2)
        # if shape_convo == 'square':
            #end_mirrors = [[0, 0], [-1, 0], [0, -1], [-1, -1]]
            # for end_mirror in end_mirrors:
                # tmp_s = np.insert(
                    # self.data,
                    # end_mirror[0],
                    # self.data[...,
                              # end_mirror[0],
                              #:,
                              #:],
                    # axis=-3)
                # data_s += np.insert(tmp_s, end_mirror[1],
                                    # tmp_s[..., end_mirror[1], :], axis=-2)
            # if corner == -1:
                #data_s = data_s[..., 1:, :, :][..., 1:, :]
            # else:
                #data_s = data_s[..., :-1, :, :][..., :-1, :]
        # elif shape_convo == 'cross':
            #data_s = np.insert(data_s, 0, 0, axis=-3)
            #data_s = np.insert(data_s, 0, 0, axis=-2)
            # if corner == -1:
                # end_mirrors = [[0, -1, 0, -1], [-1, -1, 0, -1],
                               #[0, 0, 0, -1], [0, -1, 0, 0], [0, -1, -1, -1]]
            # elif corner == 1:
                # end_mirrors = [[0, -1, 0, -1], [0, 0, 0, 0],
                               #[-1, -1, 0, 0], [0, 0, -1, -1], [-1, -1, -1, -1]]
            # else:
                # end_mirrors = [
                    #[0, -1, 0, -1], [-1, -1, 0, -1], [0,
                                                      # 0, 0, -1], [0, -1, 0, 0],
                    #[0, -1, -1, -1], [0, 0, 0, 0], [-1, -1, 0, 0], [0, 0, -1, -1], [-1, -1, -1, -1]]
            # for end_mirror in end_mirrors:
                # tmp_s = np.insert(
                    # self.data,
                    # end_mirror[0],
                    # self.data[...,
                              # end_mirror[0],
                              #:,
                              #:],
                    # axis=-3)
                # tmp_s = np.insert(
                    # tmp_s,
                    # end_mirror[1],
                    # tmp_s[...,
                          # end_mirror[0],
                          #:,
                          #:],
                    # axis=-3)
                # tmp_s = np.insert(
                    # tmp_s,
                    # end_mirror[2],
                    # tmp_s[...,
                          # end_mirror[1],
                          #:],
                    # axis=-2)
                # data_s += np.insert(tmp_s, end_mirror[3],
                                    # tmp_s[..., end_mirror[1], :], axis=-2)
            #data_s = data_s[..., 1:-2, :, :][..., 1:-2, :]
        # if hasattr(self.metadata, 'SEM'):
            #mp = self.metadata.Acquisition_instrument.SEM
        # else:
            #mp = self.metadata.TEM
        # if hasattr(mp, 'EDS') and hasattr(mp.Detector.EDS, 'live_time'):
            #mp.Detector.EDS.live_time = mp.Detector.EDS.live_time * len(end_mirrors)
        #self.data = data_s
############################
    # def plot_xray_line(self, line_to_plot='selected'):
        #"""
        # Annotate a spec.plot() with the name of the selected X-ray
        # lines
        # Parameters
        #----------
        # line_to_plot: string 'selected'|'a'|'ab|'all'
            # Defined which lines to annotate. 'selected': the selected one,
            #'a': all alpha lines of the selected elements, 'ab': all alpha and
            # beta lines, 'all': all lines of the selected elements
        # See also
        #--------
        #set_elements, add_elements
        #"""
        # if self.axes_manager.navigation_dimension > 0:
            #raise ValueError("Works only for single spectrum")
        #mp = self.metadata
        # if hasattr(self.metadata, 'SEM') and\
                # hasattr(self.metadata.Acquisition_instrument.SEM, 'beam_energy'):
            #beam_energy = mp.Acquisition_instrument.SEM.beam_energy
        # elif hasattr(self.metadata, 'TEM') and\
                # hasattr(self.metadata.TEM, 'beam_energy'):
            #beam_energy = mp.TEM.beam_energy
        # else:
            #beam_energy = 300
        #elements = []
        #lines = []
        # if line_to_plot == 'selected':
            #xray_lines = mp.Sample.xray_lines
            # for xray_line in xray_lines:
                #element, line = utils_eds._get_element_and_line(xray_line)
                # elements.append(element)
                # lines.append(line)
        # else:
            # for element in mp.Sample.elements:
                # for line, en in elements_db[element]['Atomic_properties']['Xray_lines'].items():
                    # if en < beam_energy:
                        # if line_to_plot == 'a' and line[1] == 'a':
                            # elements.append(element)
                            # lines.append(line)
                        # elif line_to_plot == 'ab':
                            # if line[1] == 'a' or line[1] == 'b':
                                # elements.append(element)
                                # lines.append(line)
                        # elif line_to_plot == 'all':
                            # elements.append(element)
                            # lines.append(line)
        #xray_lines = []
        #line_energy = []
        #intensity = []
        # for i, element in enumerate(elements):
            # line_energy.append(elements_db[element]['Atomic_properties']['Xray_lines'][lines[i]])
            # if lines[i] == 'a':
                # intensity.append(self[line_energy[-1]].data[0])
            # else:
                #relative_factor = elements_db['lines']['ratio_line'][lines[i]]
                #a_eng = elements_db[element]['Atomic_properties']['Xray_lines'][lines[i][0] + 'a']
                #intensity.append(self[a_eng].data[0] * relative_factor)
            #xray_lines.append(element + '_' + lines[i])
        # self.plot()
        # for i in range(len(line_energy)):
            # plt.text(line_energy[i], intensity[i] * 1.1, xray_lines[i],
                     # rotation=90)
            #plt.vlines(line_energy[i], 0, intensity[i] * 0.8, color='black')
