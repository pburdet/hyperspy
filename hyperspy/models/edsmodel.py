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
import math

from hyperspy.model import Model
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
import hyperspy.components as create_component


def _get_weight(element, line, weight_line=None):
    if weight_line is None:
        weight_line = elements_db[
            element]['Atomic_properties']['Xray_lines'][line]['weight']
    return lambda x: x * weight_line


def _get_iweight(element, line, weight_line=None):
    if weight_line is None:
        weight_line = elements_db[
            element]['Atomic_properties']['Xray_lines'][line]['weight']
    return lambda x: x / weight_line


def _get_sigma(E, E_ref, units_factor):
    # 2.5 from Goldstein, / 1000 eV->keV, / 2.355^2 for FWHM -> sigma
    return lambda sig_ref: math.sqrt(abs(
        4.5077 * 1e-4 * (E - E_ref) * units_factor + np.power(sig_ref, 2)))


def _get_offset(E, diff):
    return lambda E: E + diff


def _get_scale(E1, E_ref1, fact):
    return lambda E: E1 + fact * (E - E_ref1)


class EDSModel(Model):
    """Build a fit a model for EDS instance

    Parameters
    ----------
    spectrum : an EDSSpectrum (or any EDSSpectrum subclass) instance
    auto_add_lines : boolean
        If True, automatically add Gaussians for all X-rays generated
        in the energy range by an element, using the edsmodel.add_family_lines
        method
    auto_background : boolean
        If True, adds automatically a polynomial order 6 to the model,
        using the edsmodel.add_polynomial_background method.

    Example
    -------
    >>> m = create_model(s)
    >>> m.fit()
    >>> m.fit_background()
    >>> m.calibrate_energy_axis('resolution')
    >>> m.calibrate_xray_lines('energy',['Au_Ma'])
    >>> m.calibrate_xray_lines('sub_weight',['Mn_La'],bound=10)
    """

    def __init__(self, spectrum,
                 auto_add_lines=True,
                 *args, **kwargs):
        Model.__init__(self, spectrum, *args, **kwargs)
        self.xray_lines = list()
        self.background_components = list()
        end_energy = self.axes_manager.signal_axes[0].high_value
        if self.spectrum._get_beam_energy() < end_energy:
            self.end_energy = self.spectrum._get_beam_energy()
        else:
            self.end_energy = end_energy
        self.start_energy = self.axes_manager.signal_axes[0].low_value
        units_name = self.axes_manager.signal_axes[0].units
        if units_name == 'eV':
            self.units_factor = 1000.
        elif units_name == 'keV':
            self.units_factor = 1.
        else:
            raise ValueError("Energy units, %s, not supported" %
                             str(units_name))
        if auto_add_lines is True:
            self.add_family_lines()

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, EDSSpectrum):
            self._spectrum = value
        else:
            raise ValueError(
                "This attribute can only contain an EDSSpectrum "
                "but an object of type %s was provided" %
                str(type(value)))

    @property
    def _active_xray_lines(self):
        return [xray_line for xray_line
                in self.xray_lines if xray_line.active]

    def add_family_lines(self, xray_lines='from_elements'):
        """Create the Xray-lines instances and configure them appropiately

        If a X-ray line is given, all the the lines of the familiy is added.
        For instance if Zn Ka is given, Zn Kb is added too. The main lines
        (alpha) is added to self.xray_lines

        Parameters
        -----------
        xray_lines: {None, 'from_elements', list of string}
            If None, if `metadata` contains `xray_lines` list of lines use
            those. If 'from_elements', add all lines from the elements contains
            in `metadata`. Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols. (eg. ('Al_Ka','Zn_Ka')).
        """

        only_one = False
        only_lines = ("Ka", "La", "Ma")

        if xray_lines is None or xray_lines == 'from_elements':
            if 'Sample.xray_lines' in self.spectrum.metadata \
                    and xray_lines != 'from_elements':
                xray_lines = self.spectrum.metadata.Sample.xray_lines
            elif 'Sample.elements' in self.spectrum.metadata:
                xray_lines = self.spectrum._get_lines_from_elements(
                    self.spectrum.metadata.Sample.elements,
                    only_one=only_one,
                    only_lines=only_lines)
            else:
                raise ValueError(
                    "No elements defined, set them with `add_elements`")

        components_names = [xr.name for xr in self.xray_lines]
        xray_lines = filter(lambda x: x not in components_names, xray_lines)
        xray_lines, xray_not_here = self.spectrum.\
            _get_xray_lines_in_spectral_range(xray_lines)
        for xray in xray_not_here:
            print("Warning: %s is not in the data energy range." % (xray))

        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy, line_FWHM = self.spectrum._get_line_energy(
                xray_line,
                FWHM_MnKa='auto')
            component = create_component.Gaussian()
            component.centre.value = line_energy
            component.sigma.value = line_FWHM / 2.355
            component.centre.free = False
            component.sigma.free = False
            component.name = xray_line
            self.append(component)
            self.xray_lines.append(component)
            init = True
            if init:
                self[xray_line].A.map[
                    'values'] = self.spectrum.isig[line_energy].data * \
                    line_FWHM / self.spectrum.axes_manager[-1].scale
                self[xray_line].A.map['is_set'] = (
                    np.ones(self.spectrum.isig[line_energy].data.shape) == 1)

            component.A.ext_force_positive = True
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    if self.spectrum.\
                            _get_xray_lines_in_spectral_range(
                                [xray_sub])[0] != []:
                        line_energy, line_FWHM = self.spectrum.\
                            _get_line_energy(
                                xray_sub, FWHM_MnKa='auto')
                        component_sub = create_component.Gaussian()
                        component_sub.centre.value = line_energy
                        component_sub.name = xray_sub
                        component_sub.sigma.value = line_FWHM / 2.355
                        component_sub.centre.free = False
                        component_sub.sigma.free = False
                        component_sub.A.twin_function = _get_weight(
                            element, li)
                        component_sub.A.twin_inverse_function = _get_iweight(
                            element, li)
                        component_sub.A.twin = component.A
                        self.append(component_sub)
            self.fetch_stored_values()

    @property
    def _active_background_components(self):
        return [bc for bc in self.background_components
                if bc.coefficients.free]

    def add_polynomial_background(self, order=6):
        """
        Add a polynomial background.

        the background is added to self.background_components

        Parameters
        ----------
        order: int
            The order of the polynomial
        """
        background = create_component.Polynomial(order=order)
        background.name = 'background_order_' + str(order)
        background.isbackground = True
        self.append(background)
        self.background_components.append(background)

    def free_background(self):
        """
        Free the yscale of the background components.
        """
        for component in self.background_components:
            component.coefficients.free = True

    def fix_background(self):
        """
        Fix the background components.
        """
        for component in self._active_background_components:
            component.coefficients.free = False

    def enable_xray_lines(self):
        """Enable the X-ray lines components.

        """
        for component in self.xray_lines:
            component.active = True

    def disable_xray_lines(self):
        """Disable the X-ray lines components.

        """
        for component in self._active_xray_lines:
            component.active = False

    def fit_background(self,
                       start_energy=None,
                       end_energy=None,
                       windows_sigma=[4., 3.],
                       kind='single',
                       **kwargs):
        """
        Fit the background in the energy range containing no X-ray line.

        After the fit, the background is fixed.

        Parameters
        ----------
        start_energy : {float, None}
            If float, limit the range of energies from the left to the
            given value.
        end_energy : {float, None}
            If float, limit the range of energies from the right to the
            given value.
        windows_sigma: list of two float
            The uppet and lower bounds around each X-ray lines to define
            the energy range free of X-ray lines.
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or

        See also
        --------
        free_background
        """

        if end_energy is None:
            end_energy = self.end_energy
        if start_energy is None:
            start_energy = self.start_energy

        # desactivate line
        self.free_background()
        self.disable_xray_lines()
        self.set_signal_range(start_energy, end_energy)
        for component in self:
            if component.isbackground is False:
                self.remove_signal_range(
                    component.centre.value -
                    windows_sigma[0] * component.sigma.value,
                    component.centre.value +
                    windows_sigma[1] * component.sigma.value)
        if kind == 'single':
            self.fit(**kwargs)
        if kind == 'multi':
            self.multifit(**kwargs)
        self.reset_signal_range()
        self.enable_xray_lines()
        self.fix_background()

    def _twin_xray_lines_width(self, xray_lines):
        """
        Twin the width of the peaks

        The twinning models the energy resolution of the detector

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]

        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            if i == 0:
                component_ref = component
                component_ref.sigma.free = True
                E_ref = component_ref.centre.value
            else:
                component.sigma.free = True
                E = component.centre.value
                component.sigma.twin_function = _get_sigma(
                    E, E_ref, self.units_factor)
                component.sigma.twin_inverse_function = _get_sigma(
                    E_ref, E, self.units_factor)
                component.sigma.twin = component_ref.sigma

    def _set_energy_resolution(self, xray_lines, ref=None):
        """
        Adjust the width of all lines and set the fitted energy resolution
        to the spectrum

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        ref: None
            dummy args, to work like other set_..._energy
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        energy_Mn_Ka, FWHM_MnKa_old = self.spectrum._get_line_energy('Mn_Ka',
                                                                     'auto')
        FWHM_MnKa_old *= 1000. / self.units_factor
        get_sigma_Mn_Ka = _get_sigma(
            energy_Mn_Ka, self[xray_lines[0]].centre.value, self.units_factor)
        FWHM_MnKa = get_sigma_Mn_Ka(self[xray_lines[0]].sigma.value
                                    ) * 1000. / self.units_factor * 2.355
        if FWHM_MnKa < 110:
            print "FWHM_MnKa of " + str(FWHM_MnKa) + " smaller than " + \
                "physically possible"
        else:
            self.spectrum.set_microscope_parameters(
                energy_resolution_MnKa=FWHM_MnKa)
            print("Energy resolution (FWHM at Mn Ka) changed from " +
                  "%lf to %lf eV" % (FWHM_MnKa_old, FWHM_MnKa))
            for component in self:
                if component.isbackground is False:
                    line_energy, line_FWHM = self.spectrum._get_line_energy(
                        component.name, FWHM_MnKa='auto')
                    component.sigma.value = line_FWHM / 2.355

    def _twin_xray_lines_scale(self, xray_lines):
        """
        Twin the scale of the peaks

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        ax = self.spectrum.axes_manager[-1]
        ref = []
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            if i == 0:
                component_ref = component
                component_ref.centre.free = True
                E_ref = component_ref.centre.value
                ref.append(E_ref)
            else:
                component.centre.free = True
                E = component.centre.value
                fact = float(ax.value2index(E)) / ax.value2index(E_ref)
                component.centre.twin_function = _get_scale(E, E_ref, fact)
                component.centre.twin_inverse_function = _get_scale(
                    E_ref, E, 1./fact)
                component.centre.twin = component_ref.centre
                ref.append(E)
        return ref

    def _set_energy_scale(self, xray_lines, ref):
        """
        Adjust the width of all lines and set the fitted energy resolution
        to the spectrum

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        ref: float
            the centre of the first line before the fit
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        ax = self.spectrum.axes_manager[-1]
        scale_old = self.spectrum.axes_manager[-1].scale
        ind = np.argsort(np.array(
            [compo.centre.value for compo in self.xray_lines]))[-1]
        E = self[xray_lines[ind]].centre.value
        scale = (ref[ind] - ax.offset) / ax.value2index(E)
        ax.scale = scale
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            component.centre.value = ref[i]
        print "Scale changed from  %lf to %lf" % (scale_old, scale)

    def _twin_xray_lines_offset(self, xray_lines):
        """
        Twin the offset of the peaks

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        ref = []
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            if i == 0:
                component_ref = component
                component_ref.centre.free = True
                E_ref = component_ref.centre.value
                ref.append(E_ref)
            else:
                component.centre.free = True
                E = component.centre.value
                diff = E_ref - E
                component.centre.twin_function = _get_offset(E, -diff)
                component.centre.twin_inverse_function = _get_offset(E, diff)
                component.centre.twin = component_ref.centre
                ref.append(E)
        return ref

    def _set_energy_offset(self, xray_lines, ref):
        """
        Adjust the width of all lines and set the fitted energy resolution
        to the spectrum

        Parameters
        ----------
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        ref: float
            the centre of the first line before the fit
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        diff = self[xray_lines[0]].centre.value - ref[0]
        offset_old = self.spectrum.axes_manager[-1].offset
        self.spectrum.axes_manager[-1].offset -= diff
        offset = self.spectrum.axes_manager[-1].offset
        print "Offset changed from  %lf to %lf" % (offset_old, offset)
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            component.centre.value = ref[i]

    def calibrate_energy_axis(self,
                              calibrate='resolution',
                              xray_lines='all_alpha',
                              spread_to_all_lines=True,
                              **kwargs):
        """
        Calibrate the resolution, the scale or the offset of the energy axis
        by fitting.

        Parameters
        ----------
        calibrate: 'resolution' or 'scale' or 'offset'
            If 'resolution', calibrate the width of all Gaussian. The width is
            given by a model of the detector resolution, obtained by
            extrapolation the `energy_resolution_MnKa` in `metadata`
            If 'scale', calibrate the scale of the energy axis
            If 'offset', calibrate the offset of the energy axis
        xray_lines: list of str or 'all_alpha'
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
            multifit, depending on the value of kind.

        """

        if calibrate == 'resolution':
            free = self._twin_xray_lines_width
            fix = self.fix_xray_lines_width
            scale = self._set_energy_resolution
        elif calibrate == 'scale':
            free = self._twin_xray_lines_scale
            fix = self.fix_xray_lines_energy
            scale = self._set_energy_scale
        elif calibrate == 'offset':
            free = self._twin_xray_lines_offset
            fix = self.fix_xray_lines_energy
            scale = self._set_energy_offset
        ref = free(xray_lines=xray_lines)
        self.fit(**kwargs)
        fix(xray_lines=xray_lines)
        scale(xray_lines=xray_lines, ref=ref)
        self.update_plot()

    def free_sub_xray_lines_weight(self, xray_lines='all', bound=0.01):
        """
        Free the weight of a sub X-ray lines

        Remove the twin on the height of sub-Xray lines (non alpha)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bounds: float
            Bound the height of the peak to a fraction of
            its height
        """
        def free_twin():
            component.A.twin = None
            component.A.free = True
            if component.A.value - bound * component.A.value <= 0:
                component.A.bmin = 1e-10
                # print 'negative twin!'
            else:
                component.A.bmin = component.A.value - \
                    bound * component.A.value
            component.A.bmax = component.A.value + \
                bound * component.A.value
            component.A.ext_force_positive = True
        xray_families = [
            utils_eds._get_xray_lines_family(line) for line in xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    free_twin()
                elif utils_eds._get_xray_lines_family(
                        component.name) in xray_families:
                    free_twin()

    def fix_sub_xray_lines_weight(self, xray_lines='all'):
        """
        Fix the weight of a sub X-ray lines to the main X-ray lines

        Establish the twin on the height of sub-Xray lines (non alpha)
        """
        def fix_twin():
            component.A.bmin = 0.0
            component.A.bmax = None
            element, line = utils_eds._get_element_and_line(component.name)
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    if self.spectrum.\
                            _get_xray_lines_in_spectral_range(
                                [xray_sub])[0] != []:
                        component_sub = self[xray_sub]
                        component_sub.A.bmin = 1e-10
                        component_sub.A.bmax = None
                        weight_line = component_sub.A.value / component.A.value
                        component_sub.A.twin_function = _get_weight(
                            element, li, weight_line)
                        component_sub.A.twin_inverse_function = _get_iweight(
                            element, li, weight_line)
                        component_sub.A.twin = component.A
        for component in self.xray_lines:
            if xray_lines == 'all':
                fix_twin()
            elif component.name in xray_lines:
                fix_twin()
        self.fetch_stored_values()

    def free_xray_lines_energy(self, xray_lines='all', bound=0.001):
        """
        Free the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in keV or eV
        """

        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.centre.free = True
                    component.centre.bmin = component.centre.value - bound
                    component.centre.bmax = component.centre.value + bound
                elif component.name in xray_lines:
                    component.centre.free = True
                    component.centre.bmin = component.centre.value - bound
                    component.centre.bmax = component.centre.value + bound

    def set_xray_lines_energy(self, xray_lines='all'):
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.centre.assign_current_value_to_all()
                elif component.name in xray_lines:
                    component.centre.assign_current_value_to_all()

    def fix_xray_lines_energy(self, xray_lines='all'):
        """
        Fix the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in keV or eV
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.centre.twin = None
                    component.centre.free = False
                    component.centre.bmin = None
                    component.centre.bmax = None
                elif component.name in xray_lines:
                    component.centre.twin = None
                    component.centre.free = False
                    component.centre.bmin = None
                    component.centre.bmax = None

    def set_xray_lines_width(self, xray_lines='all'):
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.sigma.assign_current_value_to_all()
                elif component.name in xray_lines:
                    component.sigma.assign_current_value_to_all()

    def free_xray_lines_width(self, xray_lines='all', bound=0.01):
        """
        Free the X-ray line width (sigma of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in keV or eV
        """

        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.sigma.free = True
                    component.sigma.bmin = component.sigma.value - bound
                    component.sigma.bmax = component.sigma.value + bound
                elif component.name in xray_lines:
                    component.sigma.free = True
                    component.sigma.bmin = component.sigma.value - bound
                    component.sigma.bmax = component.sigma.value + bound

    def fix_xray_lines_width(self, xray_lines='all'):
        """
        Fix the X-ray line width (sigma of the Gaussian)

        Parameters
        ----------
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in keV or eV
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.sigma.twin = None
                    component.sigma.free = False
                    component.sigma.bmin = None
                    component.sigma.bmax = None
                elif component.name in xray_lines:
                    component.sigma.twin = None
                    component.sigma.free = False
                    component.sigma.bmin = None
                    component.sigma.bmax = None

    def calibrate_xray_lines(self,
                             calibrate='energy',
                             xray_lines='all',
                             bound=1,
                             kind='single',
                             **kwargs):
        """
        Calibrate individually the X-ray line parameters.

        The X-ray line energy, the weight of the sub-lines and the X-ray line
        width can be calibrated.

        Parameters
        ----------
        calibrate: 'energy' or 'sub_weight' or 'width'
            If 'energy', calibrate the X-ray line energy.
            If 'sub_weight', calibrate the ratio between the main line
            alpha and the other sub-lines of the family
            If 'width', calibrate the X-ray line width.
        xray_lines: list of str or 'all'
            The Xray lines. If 'all', fit all lines
        bounds: float
            for 'energy' and 'width' the bound in energy, in eV
            for 'sub_weight' Bound the height of the peak to fraction of
            its height
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
            multifit, depending on the value of kind.
        """

        if calibrate == 'energy':
            bound = bound / 1000. * self.units_factor
            free = self.free_xray_lines_energy
            fix = self.fix_xray_lines_energy
            scale = self.set_xray_lines_energy
        elif calibrate == 'sub_weight':
            free = self.free_sub_xray_lines_weight
            fix = self.fix_sub_xray_lines_weight
            scale = None
        elif calibrate == 'width':
            bound = bound / 1000. * self.units_factor
            free = self.free_xray_lines_width
            fix = self.fix_xray_lines_width
            scale = self.set_xray_lines_width
        free(xray_lines=xray_lines, bound=bound)
        if kind == 'single':
            self.fit(bounded=True, fitter='mpfit', **kwargs)
            # self.fit(**kwargs)
        elif kind == 'multi':
            self.multifit(bounded=True, fitter='mpfit', **kwargs)
        fix(xray_lines=xray_lines)
        if scale is not None:
            scale(xray_lines=xray_lines)

    def get_lines_intensity(self,
                            xray_lines=None,
                            plot_result=False,
                            **kwargs):
        """
        Return the fitted intensity of the X-ray lines.

        Return the area under the gaussian corresping to the X-ray lines

        Parameters
        ----------
        xray_lines: list of str or None or 'from_metadata'
            If None, all main X-ray lines (alpha)
            If 'from_metadata', take the Xray_lines stored in the `metadata`
            of the spectrum. Alternatively, provide an iterable containing
            a list of valid X-ray lines symbols.
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
        kwargs
            The extra keyword arguments for plotting. See
            `utils.plot.plot_signals`

        Returns
        -------
        intensities : list
            A list containing the intensities as Signal subclasses.

        Examples
        --------
        >>> m.multifit()
        >>> m.get_lines_intensity(["C_Ka", "Ta_Ma"])
        """
        from hyperspy import utils
        intensities = []
        if xray_lines is None:
            xray_lines = []
            for component in self.xray_lines:
                xray_lines.append(component.name)
        else:
            if xray_lines == 'from_metadata':
                xray_lines = self.spectrum.metadata.Sample.xray_lines
            xray_lines = filter(lambda x: x in [a.name for a in
                                self], xray_lines)
        if xray_lines == []:
            raise ValueError("These X-ray lines are not part of the model.")
        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy = self.spectrum._get_line_energy(xray_line)
            data_res = self[xray_line].A.map['values']
            if self.axes_manager.navigation_dimension == 0:
                data_res = data_res[0]
            img = self.spectrum.isig[0:1].integrate1D(-1)
            img.data = data_res
            img.metadata.General.title = (
                'Intensity of %s at %.2f %s from %s' %
                (xray_line,
                 line_energy,
                 self.spectrum.axes_manager.signal_axes[0].units,
                 self.spectrum.metadata.General.title))
            if img.axes_manager.navigation_dimension >= 2:
                img = img.as_image([0, 1])
            elif img.axes_manager.navigation_dimension == 1:
                img.axes_manager.set_signal_dimension(1)
            if plot_result and img.axes_manager.signal_dimension == 0:
                print("%s at %s %s : Intensity = %.2f"
                      % (xray_line,
                         line_energy,
                         self.spectrum.axes_manager.signal_axes[0].units,
                         img.data))
            img.metadata.set_item("Sample.elements", ([element]))
            img.metadata.set_item("Sample.xray_lines", ([xray_line]))
            intensities.append(img)
        if plot_result and img.axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(intensities, **kwargs)
        return intensities
