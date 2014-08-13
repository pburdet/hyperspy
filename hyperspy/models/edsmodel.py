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

import copy
import numpy as np
#import traits.api as t
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


def _get_sigma(E, E_ref, is_eV):
    # 2.5 from Goldstein, / 1000 eV->keV, / 2.355^2 for FWHM -> sigma
    if is_eV:
        return lambda sig_ref: math.sqrt(abs(
            4.5077 * 1e-1 * (E - E_ref) + np.power(sig_ref, 2)))
    else:
        return lambda sig_ref: math.sqrt(abs(
            4.5077 * 1e-4 * (E - E_ref) + np.power(sig_ref, 2)))


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
        Model.__init__(self, spectrum, *args, **kwargs)
        self.xray_lines = list()
        self.background_components = list()
        unit_name = self.axes_manager.signal_axes[0].units
        if unit_name == 'eV':
            self.is_eV = True
        elif unit_name == 'keV':
            self.is_eV = False
        else:
            raise ValueError("Energy units, %s, not supported" %
                             str(unit_name))
        if auto_add_lines is True:
            self.add_lines()
        if auto_background is True:
            self.add_background()

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

    def add_lines(self, xray_lines=None, only_one=False,
                  only_lines=("Ka", "La", "Ma")):
        """Create the Xray-lines instances and configure them appropiately

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
            a list of valid X-ray lines symbols.
        only_lines : None or list of strings
            If not None, use only the given lines (eg. ('a','Kb')).
            If None, use all lines.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        """

        if only_lines is not None:
            only_lines = list(only_lines)
            for only_line in only_lines:
                if only_line == 'a':
                    only_lines.extend(['Ka', 'La', 'Ma'])
                elif only_line == 'b':
                    only_lines.extend(['Kb', 'Lb1', 'Mb'])

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

        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy, line_FWHM = self.spectrum._get_line_energy(xray_line,
                                                                    FWHM_MnKa='auto')
            component = create_component.Gaussian()
            component.centre.value = line_energy
            component.sigma.value = line_FWHM / 2.355
            #component.A.value = self.spectrum[..., line_energy].data.flatten().mean()

            component.centre.free = False
            component.sigma.free = False
            component.name = xray_line
            self.append(component)
            self.xray_lines.append(component)
            init = True
            if init:
                self[xray_line].A.map[
                    'values'] = self.spectrum[..., line_energy].data / line_FWHM
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
                    component_sub.sigma.value = line_FWHM / 2.355
                    component_sub.A.twin = component.A
                    component.A.ext_force_positive = True
                    component_sub.centre.free = False
                    component_sub.sigma.free = False
                    component_sub.A.twin_function = _get_weight(element, li)
                    component_sub.A.twin_inverse_function = _get_iweight(
                        element, li)
                    self.append(component_sub)

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
            if xray_lines == 'from_metadata' : 
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

    def add_background(self,
                       generation_factors=[1, 2],
                       detector_name=4,
                       weight_fraction='auto',
                       thickness=100,
                       density='auto',
                       gateway='auto'):
        """
        Add a backround to the model in the form of several
        scalable fixed patterns.

        Each pattern is the muliplication of the detector efficiency,
        the absorption in the sample (PDH equation for SEM, constant
        X-ray pdouction for TEM) and a continuous X-ray
        generation.

        Parameters
        ----------
        generation_factors: list of int
            For each number n, add (E0-E)^n/E
            [1] is equivalent to Kramer equation.
            [1,2] is equivalent to Lisfhisn modification of Kramer equation.
        det_name: int, str, None
            If None, no det_efficiency
            If {0,1,2,3,4}, INCA efficiency database
            If str, model from DTSAII
        weight_fraction: list of float
             The sample composition used for the sample absorption.
             If 'auto', takes value in metadata. If not there,
             use and equ-composition
        thickness : float
            Thickness of thin film. 
            Option only relevant for EDSTEMSpectrum. 
        density: float or 'auto'
            Set the density. in g/cm^3
            if 'auto', calculated from weight_fraction
            Option only relevant for EDSTEMSpectrum. 
        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.

        See also
        --------
        database.detector_efficiency_INCA,
        utils_eds.get_detector_properties
        """
        generation = []
        for exp_factor in generation_factors:
            generation.append(self.spectrum.compute_continuous_xray_generation(
                exp_factor))
            generation[-1].metadata.General.title = 'generation'\
                + str(exp_factor)
                
        if 'SEM' in self.spectrum.metadata.Signal.signal_type:
            absorption = self.spectrum.compute_continuous_xray_absorption(
                weight_fraction=weight_fraction)
        elif thickness == 0.:
            absorption = generation[0].deepcopy()
            absorption.data = np.ones_like(generation[0].data)
        else :
            absorption = self.spectrum.compute_continuous_xray_absorption(
                thickness=thickness, density=density,
                weight_fraction=weight_fraction)
        
        if detector_name is None:
            det_efficiency = generation[0].deepcopy()
            det_efficiency.data = np.ones_like(generation[0].data)
        else : 
            det_efficiency = self.spectrum.get_detector_efficiency(
                detector_name, gateway=gateway)

        for gen, gen_fact in zip(generation, generation_factors):
            bck = det_efficiency * gen * absorption
            # bck.plot()
            bck = bck[self.axes_manager[-1].scale:]
            bck.metadata.General.title = 'bck_' + str(gen_fact)
            component = create_component.ScalableFixedPattern(bck)
            component.set_parameters_not_free(['xscale', 'shift'])
            component.name = bck.metadata.General.title
            #component.yscale.ext_bounded = True
            #component.yscale.bmin = 0
            component.yscale.ext_force_positive = True
            component.isbackground = True
            self.append(component)
            self.background_components.append(component)

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
        if end_energy is None and \
                self.spectrum._get_beam_energy() < \
                self.axes_manager.signal_axes[0].high_value:
            end_energy = self.spectrum._get_beam_energy()
        else:
            end_energy = self.axes_manager.signal_axes[0].high_value

        # desactivate line
        self.free_background()
        self.disable_xray_lines()
        self.set_signal_range(
            start_energy, end_energy)
        for component in self:
            if component.isbackground is False:
                try:
                    self.remove_signal_range(component.centre.value -
                                             4 * component.sigma.value, component.centre.value +
                                             3 * component.sigma.value)
                except:
                    pass

        if kind == 'single':
            self.fit(**kwargs)
        if kind == 'multi':
            self.multifit(**kwargs)
        self.reset_signal_range()
        self.enable_xray_lines()
        self.fix_background()

    def free_xray_lines_energy(self, bound=0.001):
        """
        Free the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        bound: float
            the bound around the actual energy, in keV or eV
        """
        for component in self:
            if component.isbackground is False:
                component.centre.free = True
                component.centre.bmin = component.centre.value - bound
                component.centre.bmax = component.centre.value + bound

    def fix_xray_lines_energy(self):
        """
        Fix the X-ray line energy (shift or centre of the Gaussian)
        """
        for component in self:
            if component.isbackground is False:
                component.centre.free = False

    def fit_xray_lines_energy(self, bound=0.001, kind='single',
                              **kwargs):
        """
        Fit the X-ray line energy (shift or centre of the Gaussian)

        Parameters
        ----------
        bound: float
            the bound around the actual energy, in keV or eV
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        multifit, depending on the value of kind.
        """
        self.free_xray_lines_energy(bound)
        if kind == 'single':
            self.fit(fitter="mpfit", bounded=True, **kwargs)
        if kind == 'multi':
            self.multifit(fitter="mpfit", bounded=True, **kwargs)
        self.fix_xray_lines_energy()

    def free_sub_xray_lines_weight(self, bound=0.01):
        """
        Free the weight of a sub X-ray lines

        Free the height of the gaussians

        Parameters
        ----------
        bounds: float
            Bound the height of the peak to fraction (bound) of
            its height
        """
        for component in self:
            if component.isbackground is False:
                component.A.twin = None
                component.A.free = True
                if component.A.value - bound * component.A.value < 0:
                    component.A.bmin = 0.
                    print 'a'
                else:
                    component.A.bmin = component.A.value - \
                        bound * component.A.value
                component.A.bmax = component.A.value + \
                    bound * component.A.value
                #component.A.ext_force_positive = True

    def fix_sub_xray_lines_weight(self):
        """
        Fix the weight of a sub X-ray lines to the main X-ray lines

        Fix the height of the gaussians with a twin function
        """
        for component in self.xray_lines:
            element, line = utils_eds._get_element_and_line(component.name)
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    component_sub = self[xray_sub]
                    weight_line = component_sub.A.value / component.A.value
                    component_sub.A.twin_function = _get_weight(element,
                                                                li, weight_line)
                    component_sub.A.twin_inverse_function = _get_iweight(
                        element, li, weight_line)

    def fit_sub_xray_lines_weight(self, bound=0.01, kind='single',
                                  **kwargs):
        """
        Fit the weight of the sub X-ray lines

        Fit the height of the gaussians and fix them to the main line
        with a twin function

        Parameters
        ----------
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
        self.free_sub_xray_lines_weight(bound)
        if kind == 'single':
            self.fit(fitter="mpfit", bounded=True, **kwargs)
        if kind == 'multi':
            self.multifit(fitter="mpfit", bounded=True, **kwargs)
        self.fix_sub_xray_lines_weight()

    def free_energy_resolution(self):
        """
        Free the energy resolution of the main X-ray lines

        Resolutions of the different peak are twinned

        See also
        --------

        """
        xray_lines = self.xray_lines

        for i, component in enumerate(self.xray_lines):
            if i == 0:
                component_ref = component
                component_ref.sigma.free = True
                E_ref = component_ref.centre.value
            else:
                component.sigma.twin = component_ref.sigma
                component.sigma.free = True

                E = component.centre.value
                component.sigma.twin_function = _get_sigma(
                    E, E_ref, self.is_eV)
                component.sigma.twin_inverse_function = _get_sigma(
                    E_ref, E, self.is_eV)

    def fix_energy_resolution(self):
        """
        Fix the weight of a sub X-ray lines to the main X-ray lines

        Fix the height of the gaussians with a twin function
        """
        if self.is_eV:
            get_sigma = _get_sigma(5898.7, self[0].centre.value, self.is_eV)
            FWHM_MnKa = get_sigma(self[0].sigma.value) * 2.355
        else:
            get_sigma = _get_sigma(5.8987, self[0].centre.value, self.is_eV)
            FWHM_MnKa = get_sigma(self[0].sigma.value) * 1000 * 2.355
        self.spectrum.set_microscope_parameters(
            energy_resolution_MnKa=FWHM_MnKa)
        print 'FWHM_MnKa ' + str(FWHM_MnKa)
        for component in self.xray_lines:
            component.sigma.free = False
            #component.sigma.twin = None
            element, line = utils_eds._get_element_and_line(component.name)
            for li in elements_db[element]['Atomic_properties']['Xray_lines']:
                if line[0] in li and line != li:
                    xray_sub = element + '_' + li
                    component_sub = self[xray_sub]
                    component_sub.sigma.free = False
                    #component_sub.sigma.twin = None
                    line_energy, line_FWHM = self.spectrum._get_line_energy(
                        xray_sub, FWHM_MnKa='auto')
                    component_sub.sigma.value = line_FWHM / 2.355

    def fit_energy_resolution(self, kind='single',
                              **kwargs):
        """
        Fit the weight of the sub X-ray lines

        Fit the height of the gaussians and fix them to the main line
        with a twin function

        Parameters
        ----------
        kind : {'single', 'multi'}
            If 'single' fit only the current location. If 'multi'
            use multifit.
        **kwargs : extra key word arguments
            All extra key word arguments are passed to fit or
        multifit, depending on the value of kind.

        """
        self.free_energy_resolution()
        if kind == 'single':
            self.fit(**kwargs)
        if kind == 'multi':
            self.multifit(**kwargs)
        self.fix_energy_resolution()

    def fit(self, fitter=None, method='ls', grad=False,
            bounded=False, ext_bounding=True, update_plot=False,
            kind='std', **kwargs):
        """Fits the model to the experimental data.

        The chi-squared, reduced chi-squared and the degrees of freedom are
        computed automatically when fitting. They are stored as signals, in the
        `chisq`, `red_chisq`  and `dof`. Note that,
        unless ``metadata.Signal.Noise_properties.variance`` contains an accurate
        estimation of the variance of the data, the chi-squared and reduced
        chi-squared cannot be computed correctly. This is also true for
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
