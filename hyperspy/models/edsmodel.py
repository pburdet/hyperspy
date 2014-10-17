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

# To do: weight_fraction different for different pixe. (so basckground)
# Calibrate on standard and transfer dictionnary
# k-ratios
from __future__ import division

import numpy as np
import math

from hyperspy.model import Model
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
import hyperspy.components as create_component
from hyperspy import utils


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
            # self.spectrum._are_microscope_parameters_missing()
        else:
            raise ValueError(
                "This attribute can only contain an EDSSpectrum "
                "but an object of type %s was provided" %
                str(type(value)))

    def add_family_lines(self, xray_lines='from_elements'):
        """Create the Xray-lines instances and configure them appropiately

        If a X-ray line is given, all the the lines of the familiy is added

        Parameters
        -----------
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
            a list of valid X-ray lines symbols. (eg. ('Al_Ka','Zn_Ka')).
        """

        only_one = False
        only_lines = ("Ka", "La", "Ma")

        # if only_lines is not None:
        # only_lines = list(only_lines)
        # for only_line in only_lines:
        # if only_line == 'a':
        # only_lines.extend(['Ka', 'La', 'Ma'])
        # elif only_line == 'b':
        # only_lines.extend(['Kb', 'Lb1', 'Mb'])

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
        xray_lines = [xray_line for xray_line in xray_lines if
                      self.start_energy <
                      self.spectrum._get_line_energy(xray_line)]
        xray_lines = [xray_line for xray_line in xray_lines if
                      self.end_energy >
                      self.spectrum._get_line_energy(xray_line)]

        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy, line_FWHM = self.spectrum._get_line_energy(
                xray_line,
                FWHM_MnKa='auto')
            component = create_component.Gaussian()
            component.centre.value = line_energy
            # component.fwhm = line_FWHM
            component.sigma.value = line_FWHM / 2.355
            # component.A.value = self.spectrum[..., line_energy].
            # data.flatten().mean()

            component.centre.free = False
            component.sigma.free = False
            component.name = xray_line
            self.append(component)
            self.xray_lines.append(component)
            init = True
            if init:
                self[xray_line].A.map[
                    'values'] = self.spectrum[..., line_energy].data * \
                    line_FWHM / self.spectrum.axes_manager[-1].scale
                self[xray_line].A.map['is_set'] = (
                    np.ones(self.spectrum[..., line_energy].data.shape) == 1)

            # if bounded:
            #    component.A.ext_bounded = True
            component.A.ext_force_positive = True
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    line_energy, line_FWHM = self.spectrum._get_line_energy(
                        xray_sub, FWHM_MnKa='auto')
                    component_sub = create_component.Gaussian()
                    component_sub.centre.value = line_energy
                    component_sub.name = xray_sub
                    # component_sub.fwhm = line_FWHM
                    component_sub.sigma.value = line_FWHM / 2.355
                    # component.A.ext_force_positive = True
                    component_sub.centre.free = False
                    component_sub.sigma.free = False
                    component_sub.A.twin_function = _get_weight(element, li)
                    component_sub.A.twin_inverse_function = _get_iweight(
                        element, li)
                    component_sub.A.twin = component.A
                    self.append(component_sub)
            self.fetch_stored_values()

    def get_lines_intensity(self,
                            xray_lines='auto',
                            plot_result=True,
                            store_in_mp=True,
                            **kwargs):
        """

        Parameters
        ----------
        xray_lines: {list of str | 'auto' | 'from_metadata'}
            The Xray lines. If 'auto' all the fitted alpha lines.
            If 'from_metadata', take the Xray_lines stored in the metadata
            of the spectrum.
        plot_result : bool
            If True, plot the calculated line intensities. If the current
            object is a single spectrum it prints the result instead.
        store_in_mp : bool
            store the result in metadata.Sample
        kwargs
            The extra keyword arguments for plotting. See
            `utils.plot.plot_signals`
        """
        intensities = []

        if xray_lines == 'auto':
            xray_lines = []
            components = self.xray_lines
            for component in components:
                xray_lines.append(component.name)
        else:
            if xray_lines == 'from_metadata':
                xray_lines = self.spectrum.metadata.Sample.xray_lines
            components = filter(lambda x: x.name in xray_lines,
                                self.xray_lines)

        if self.spectrum.metadata.Sample.has_item(
                'xray_lines') is False and store_in_mp:
            self.spectrum.metadata.Sample.xray_lines = xray_lines
        for xray_line, component in zip(xray_lines, components):
            line_energy = self.spectrum._get_line_energy(xray_line)
            data_res = component.A.map['values']
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
                         self.axes_manager.signal_axes[0].units,
                         img.data))
        if plot_result and img.axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(intensities, **kwargs)
        if store_in_mp is False:
            return intensities

    @property
    def _active_xray_lines(self):
        return [xray_line for xray_line
                in self.xray_lines if xray_line.active]

    @property
    def _active_background_components(self):
        return [bc for bc in self.background_components if bc.yscale.free]

    def free_background(self):
        """Free the yscale of the background components.

        """
        for component in self.background_components:
            component.set_parameters_free(['yscale'])

    def fix_background(self):
        """Fix the background components.

        """
        for component in self._active_background_components:
            component.set_parameters_not_free(['yscale'])

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
                       windows_sigma=[4, 3],
                       kind='single',
                       **kwargs):
        """
        Fit the background to energy range containing no X-ray line.

        Parameters
        ----------
        start_energy : {float, None}
            If float, limit the range of energies from the left to the
            given value.
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        """
        # If there is no active background component do nothing
        # if not self._active_background_components:
        #    return
        if end_energy is None:
            end_energy = self.end_energy
        if start_energy is None:
            start_energy = self.start_energy

        # desactivate line
        self.free_background()
        self.disable_xray_lines()
        self.set_signal_range(
            start_energy, end_energy)
        for component in self:
            if component.isbackground is False:
                try:
                    self.remove_signal_range(
                        component.centre.value -
                        windows_sigma[0] * component.sigma.value,
                        component.centre.value +
                        windows_sigma[1] * component.sigma.value)
                except:
                    pass

        if kind == 'single':
            self.fit(**kwargs)
        if kind == 'multi':
            self.multifit(**kwargs)
        self.reset_signal_range()
        self.enable_xray_lines()
        self.fix_background()

    def free_xray_lines_energy(self, xray_lines='all', bound=0.001):
        """
        Free the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        xray_lines: {list of str | 'all'}
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

    def fix_xray_lines_energy(self, xray_lines='all'):
        """
        Fix the X-ray line energy (shift or centre of the Gaussian)
        """
        for component in self:
            if component.isbackground is False:
                if xray_lines == 'all':
                    component.centre.free = False
                    component.centre.bmin = None
                    component.centre.bmax = None
                elif component.name in xray_lines:
                    component.centre.free = False
                    component.centre.bmin = None
                    component.centre.bmax = None

    def fit_xray_lines_energy(self, xray_lines='all',
                              bound=5.,
                              kind='single', fitter="mpfit",
                              **kwargs):
        """
        Fit the X-ray line energy (shift of centre of the Gaussian)

        Parameters
        ----------
        xray_lines: {list of str | 'all'}
            The Xray lines. If 'all', fit all lines
        bound: float
            the bound around the actual energy, in eV
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        multifit, depending on the value of kind.
        """
        bound = bound / 1000. * self.units_factor
        self.free_xray_lines_energy(xray_lines=xray_lines, bound=bound)

        if kind == 'single':
            if xray_lines != 'all':
                energy_before = []
                for xray_line in xray_lines:
                    energy_before.append(self[xray_line].centre.value)
            self.fit(fitter=fitter, bounded=True, **kwargs)
            if xray_lines != 'all':
                for i, xray_line in enumerate(xray_lines):
                    print xray_line + ' shift of ' + str(
                        self[xray_line].centre.value - energy_before[i])

        if kind == 'multi':
            self.multifit(fitter=fitter, bounded=True, **kwargs)
        self.fix_xray_lines_energy(xray_lines=xray_lines)

    def free_sub_xray_lines_weight(self, xray_lines='all', bound=0.01):
        """
        Free the weight of a sub X-ray lines

        Free the height of the gaussians

        Parameters
        ----------
        xray_lines: {list of str | 'all'}
            The Xray lines. If 'all', fit all lines
        bounds: float
            Bound the height of the peak to fraction (bound) of
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

        Fix the height of the gaussians with a twin function
        """
        def fix_twin():
            component.A.bmin = 0.0
            component.A.bmax = None
            element, line = utils_eds._get_element_and_line(component.name)
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
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

    def fit_sub_xray_lines_weight(self, xray_lines='all', bound=0.01,
                                  kind='single', fitter="mpfit",
                                  **kwargs):
        """
        Fit the weight of the sub X-ray lines

        Fit the height of the gaussians and fix them to the main line
        with a twin function

        Parameters
        ----------
        xray_lines: {list of str | 'all'}
            The Xray lines. If 'all', fit all lines
        bounds: float
            Bound the height of the peak to fraction (bound) of
            its height
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        multifit, depending on the value of kind.

        """
        self.free_sub_xray_lines_weight(xray_lines=xray_lines, bound=bound)
        if kind == 'single':
            self.fit(fitter=fitter, bounded=True, **kwargs)
        elif kind == 'multi':
            self.multifit(fitter=fitter, bounded=True, **kwargs)
        self.fix_sub_xray_lines_weight(xray_lines=xray_lines)

    def free_energy_resolution(self, xray_lines):
        """
        Free the energy resolution of the main X-ray lines

        Resolutions of the different peak are twinned

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

    def fix_energy_resolution(self, xray_lines):
        """
        Fix and remove twin of X-ray lines sigma
        """

        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        for i, xray_line in enumerate(xray_lines):
            component = self[xray_line]
            component.sigma.twin = None
            component.sigma.free = False

    def set_energy_resolution(self, xray_lines):
        """
        Set the fitted energy resolution to the spectrum and
        adjust the FHWM for all lines
        """
        if xray_lines == 'all_alpha':
            xray_lines = [compo.name for compo in self.xray_lines]
        energy_Mn_Ka = self.spectrum._get_line_energy('Mn_Ka')
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
            print 'FWHM_MnKa ' + str(FWHM_MnKa)

            for component in self:
                if component.isbackground is False:
                    line_energy, line_FWHM = self.spectrum._get_line_energy(
                        component.name, FWHM_MnKa='auto')
                    component.sigma.value = line_FWHM / 2.355

    def fit_energy_resolution(self, xray_lines='all_alpha',
                              kind='single',
                              spread_to_all_lines=True,
                              **kwargs):
        """
        Fit the energy resolution

        energy scaling of the spectrum

        Parameters
        ----------
        xray_lines: {list of str | 'all_alpha'}
            The Xray lines. If 'all_alpha', fit all using all alpha lines
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        spread_to_all_lines: bool
            if True, change the width of all lines and change the
            energy_resolution_MnKa of the spectrum
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        multifit, depending on the value of kind.

        """
        self.free_energy_resolution(xray_lines=xray_lines)
        if kind == 'single':
            self.fit(**kwargs)
        if kind == 'multi':
            self.multifit(**kwargs)
        self.fix_energy_resolution(xray_lines=xray_lines)
        if spread_to_all_lines:
            self.set_energy_resolution(xray_lines=xray_lines)

    def fit(self, fitter=None, method='ls', grad=False,
            bounded=False, ext_bounding=True, update_plot=False,
            kind='std', **kwargs):
        """Fits the model to the experimental data.

        The chi-squared, reduced chi-squared and the degrees of freedom are
        computed automatically when fitting. They are stored as signals, in the
        `chisq`, `red_chisq`  and `dof`. Note that,
        unless ``metadata.Signal.Noise_properties.variance`` contains an
        accurate estimation of the variance of the data, the chi-squared and
        reduced chi-squared cannot be computed correctly. This is also true for
        homocedastic noise.

        Parameters
        ----------
        fitter : {None, "leastsq", "odr", "mpfit", "fmin"}
            The optimizer to perform the fitting. If None the fitter
            defined in `preferences.Model.default_fitter` is used.
            "leastsq" performs least squares using the Levenberg–Marquardt
            algorithm.
            "mpfit"  performs least squares using the Levenberg–Marquardt
            algorithm and, unlike "leastsq", support bounded optimization.
            "fmin" performs curve fitting using a downhill simplex algorithm.
            It is less robust than the Levenberg-Marquardt based optimizers,
            but, at present, it is the only one that support maximum likelihood
            optimization for poissonian noise.
            "odr" performs the optimization using the orthogonal distance
            regression algorithm. It does not support bounds.
            "leastsq", "odr" and "mpfit" can estimate the standard deviation of
            the estimated value of the parameters if the
            "metada.Signal.Noise_properties.variance" attribute is defined.
            Note that if it is not defined the standard deviation is estimated
            using variance equal 1, what, if the noise is heterocedatic, will
            result in a biased estimation of the parameter values and errors.i
            If `variance` is a `Signal` instance of the
            same `navigation_dimension` as the spectrum, and `method` is "ls"
            weighted least squares is performed.
        method : {'ls', 'ml'}
            Choose 'ls' (default) for least squares and 'ml' for poissonian
            maximum-likelihood estimation.  The latter is only available when
            `fitter` is "fmin".
        grad : bool
            If True, the analytical gradient is used if defined to
            speed up the optimization.
        bounded : bool
            If True performs bounded optimization if the fitter
            supports it. Currently only "mpfit" support it.
        update_plot : bool
            If True, the plot is updated during the optimization
            process. It slows down the optimization but it permits
            to visualize the optimization progress.
        ext_bounding : bool
            If True, enforce bounding by keeping the value of the
            parameters constant out of the defined bounding area.
        kind : {'std', 'smart'}
            If 'std' (default) performs standard fit. If 'smart'
            performs smart_fit
        **kwargs : key word arguments
            Any extra key word argument will be passed to the chosen
            fitter. For more information read the docstring of the optimizer
            of your choice in `scipy.optimize`.

        See Also
        --------
        multifit, smart_fit

        """

        if kind == 'smart':
            self.smart_fit(fitter=fitter,
                           method=method,
                           grad=grad,
                           bounded=bounded,
                           ext_bounding=ext_bounding,
                           update_plot=update_plot,
                           **kwargs)
        elif kind == 'std':
            Model.fit(self,
                      fitter=fitter,
                      method=method,
                      grad=grad,
                      bounded=bounded,
                      ext_bounding=ext_bounding,
                      update_plot=update_plot,
                      **kwargs)
        else:
            raise ValueError('kind must be either \'std\' or \'smart\'.'
                             '\'%s\' provided.' % kind)
