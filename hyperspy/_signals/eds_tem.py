# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import traits.api as t
import numpy as np
from scipy import ndimage
from scipy import constants

from hyperspy import utils
from hyperspy._signals.eds import EDSSpectrum
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.eds import utils as utils_eds
from hyperspy import utils
from hyperspy.misc.eds import physical_model
from hyperspy.misc.eds import database


class EDSTEMSpectrum(EDSSpectrum):
    _signal_type = "EDS_TEM"

    def __init__(self, *args, **kwards):
        EDSSpectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        if 'Acquisition_instrument.TEM.Detector.EDS' not in self.metadata:
            if 'Acquisition_instrument.SEM.Detector.EDS' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.TEM",
                    self.metadata.Acquisition_instrument.SEM)
                del self.metadata.Acquisition_instrument.SEM
        self._set_default_param()

    def _set_default_param(self):
        """Set to value to default (defined in preferences)
        """

        mp = self.metadata
        mp.Signal.signal_type = 'EDS_TEM'

        mp = self.metadata
        if "mp.Acquisition_instrument.TEM.tilt_stage" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.tilt_stage",
                preferences.EDS.eds_tilt_stage)
        if "Acquisition_instrument.TEM.Detector.EDS.elevation_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa"\
                not in mp:
            mp.set_item("Acquisition_instrument.TEM.Detector.EDS." +
                        "energy_resolution_MnKa",
                        preferences.EDS.eds_mn_ka)
        if "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                preferences.EDS.eds_detector_azimuth)

    def set_microscope_parameters(self,
                                  beam_energy=None,
                                  live_time=None,
                                  tilt_stage=None,
                                  azimuth_angle=None,
                                  elevation_angle=None,
                                  energy_resolution_MnKa=None):
        """Set the microscope parameters.

        If no arguments are given, raises an interactive mode to fill
        the values.

        Parameters
        ----------
        beam_energy: float
            The energy of the electron beam in keV
        live_time : float
            In second
        tilt_stage : float
            In degree
        azimuth_angle : float
            In degree
        elevation_angle : float
            In degree
        energy_resolution_MnKa : float
            In eV

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> print(s.metadata.Acquisition_instrument.
        >>>       TEM.Detector.EDS.energy_resolution_MnKa)
        >>> s.set_microscope_parameters(energy_resolution_MnKa=135.)
        >>> print(s.metadata.Acquisition_instrument.
        >>>       TEM.Detector.EDS.energy_resolution_MnKa)
        133.312296
        135.0

        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.TEM.beam_energy ", beam_energy)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item("Acquisition_instrument.TEM.tilt_stage", tilt_stage)
        if azimuth_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                azimuth_angle)
        if elevation_angle is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                elevation_angle)
        if energy_resolution_MnKa is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS." +
                "energy_resolution_MnKa",
                energy_resolution_MnKa)

        if set([beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa]) == {None}:
            self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        tem_par = TEMParametersUI()
        mapping = {
            'Acquisition_instrument.TEM.beam_energy':
            'tem_par.beam_energy',
            'Acquisition_instrument.TEM.tilt_stage':
            'tem_par.tilt_stage',
            'Acquisition_instrument.TEM.Detector.EDS.live_time':
            'tem_par.live_time',
            'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle':
            'tem_par.azimuth_angle',
            'Acquisition_instrument.TEM.Detector.EDS.elevation_angle':
            'tem_par.elevation_angle',
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa':
            'tem_par.energy_resolution_MnKa', }
        for key, value in mapping.iteritems():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        tem_par.edit_traits()

        mapping = {
            'Acquisition_instrument.TEM.beam_energy':
            tem_par.beam_energy,
            'Acquisition_instrument.TEM.tilt_stage':
            tem_par.tilt_stage,
            'Acquisition_instrument.TEM.Detector.EDS.live_time':
            tem_par.live_time,
            'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle':
            tem_par.azimuth_angle,
            'Acquisition_instrument.TEM.Detector.EDS.elevation_angle':
            tem_par.elevation_angle,
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa':
            tem_par.energy_resolution_MnKa, }

        for key, value in mapping.iteritems():
            if value != t.Undefined:
                self.metadata.set_item(key, value)
        self._are_microscope_parameters_missing()

    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in metadata. Raise in interactive mode
         an UI item to fill or cahnge the values"""
        must_exist = (
            'Acquisition_instrument.TEM.beam_energy',
            'Acquisition_instrument.TEM.Detector.EDS.live_time',)

        missing_parameters = []
        for item in must_exist:
            exists = self.metadata.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            if preferences.General.interactive is True:
                par_str = "The following parameters are missing:\n"
                for par in missing_parameters:
                    par_str += '%s\n' % par
                par_str += 'Please set them in the following wizard'
                is_ok = messagesui.information(par_str)
                if is_ok:
                    self._set_microscope_parameters()
                else:
                    return True
            else:
                return True
        else:
            return False

    def get_calibration_from(self, ref, nb_pix=1):
        """Copy the calibration and all metadata of a reference.

        Primary use: To add a calibration to ripple file from INCA
        software

        Parameters
        ----------
        ref : signal
            The reference contains the calibration in its
            metadata
        nb_pix : int
            The live time (real time corrected from the "dead time")
            is divided by the number of pixel (spectrums), giving an
            average live time.

        Examples
        --------
        >>> ref = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s = hs.signals.EDSTEMSpectrum(
        >>>     hs.datasets.example_signals.EDS_TEM_Spectrum().data)
        >>> print s.axes_manager[0].scale
        >>> s.get_calibration_from(ref)
        >>> print s.axes_manager[0].scale
        1.0
        0.020028

        """

        self.original_metadata = ref.original_metadata.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units
        ax_m.offset = ax_ref.offset

        # Setup metadata
        if 'Acquisition_instrument.TEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.TEM
        elif 'Acquisition_instrument.SEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.SEM
        else:
            raise ValueError("The reference has no metadata." +
                             "Acquisition_instrument.TEM" +
                             "\n nor metadata.Acquisition_instrument.SEM ")

        mp = self.metadata
        mp.Acquisition_instrument.TEM = mp_ref.deepcopy()
        if mp_ref.has_item("Detector.EDS.live_time"):
            mp.Acquisition_instrument.TEM.Detector.EDS.live_time = \
                mp_ref.Detector.EDS.live_time / nb_pix

    def simulate_two_elements_standard(self,
                                       common_xray='Si_Ka',
                                       nTraj=10000,
                                       dose=100,
                                       density='auto',
                                       thickness=20,
                                       detector='SDD',
                                       gateway='auto'):
        """
        Simulate the mixed standard using DTSA-II (NIST-Monte)

        For each element of the spectrum (self.metadata.Sample.elements),
        simulate a spectrum with 50wt% of the element and 50wt% of the common
        element. Store the list of spectra in metadata.

        Parameters
        ----------

        common_xray: str
            The Xray line use as the common element for each standard.
        nTraj: int
            number of electron trajectories
        dose: float
            Electron current time the live time in nA*sec
        density: list of float
            Set the density. If 'auto', obtain from the compo_at.
        thickness: float
            Set the thickness.
        detector: str
            Give the detector name defined in DTSA-II
        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.

        Examples
        ---------
        >>> s = database.spec3D('TEM')
        >>> s.set_elements(["Ni", "Cr",'Al'])
        >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
        >>> s.set_microscope_parameters(live_time=30)
        >>> s.simulate_two_elements_standard(nTraj=100)
        """
        std_met = self.deepcopy()
        std_met.metadata.Sample.thickness = thickness
        common_element, line = utils_eds._get_element_and_line(common_xray)
        std = []
        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()
        for el, xray in zip(self.metadata.Sample.elements,
                            self.metadata.Sample.xray_lines):
            el_binary = [el] + [common_element]
            xray_binary = [xray] + [common_xray]
            # atomic_percent = np.array(utils.material.weight_to_atomic(
            #    el_binary, [0.5, 0.5]))
            std_met.set_elements(el_binary)
            std_met.set_lines(xray_binary)
            std.append(utils_eds.simulate_one_spectrum_TEM(
                nTraj, dose=dose, mp=std_met.metadata, detector=detector,
                weight_fraction=[0.5, 0.5], gateway=gateway))
            std[-1].metadata.General.title = el + '_'\
                + common_element + '_comon'
            std[-1].metadata.Sample.weight_percent = [0.5, 0.5]
        self.metadata.Sample.standard_spec = std

    def get_kfactors_from_standard(self,
                                   common_line='Ka',
                                   **kwargs):
        """
        Exctract the kfactor from two elements standard

        Store the kfactor in metadata.sample.kfactors

        Parameters
        ----------
        common_line: str
            The line for the common element to use.
        kwargs
        The extra keyword arguments for get_lines_intensity

        Examples
        ---------
        >>> s = database.spec3D('TEM')
        >>> s.set_elements(["Ni", "Cr",'Al'])
        >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
        >>> s.set_microscope_parameters(live_time=30)
        >>> s.simulate_two_elements_standard(nTraj=100)
        >>> s.get_kfactors_from_standard()

        See also
        -------
        simulate_two_elements_standard, get_lines_intensity


        """
        std_title = self.metadata.Sample.standard_spec[0
                                                       ].metadata.General.title
        if 'comon' not in std_title:
            raise ValueError(
                "Two elements standard are needed. See " +
                "simulate_two_elements_standard")
        else:
            common_element = std_title[std_title.find('_') + 1:]
            common_element = common_element[:common_element.find('_')]
            common_xray = str(common_element + '_' + common_line)

        kfactors = []
        for i, (std, el, xray) in enumerate(
                zip(self.metadata.Sample.standard_spec,
                    self.metadata.Sample.elements,
                    self.metadata.Sample.xray_lines)):
            intens = std.get_lines_intensity([xray] + [common_xray], **kwargs)
            kfactor = intens[1].data / intens[0].data
            kfactors.append(kfactor[0])
        self.metadata.Sample.kfactors = kfactors

    def get_kfactors_from_brucker(self,
                                  microscope_name='osiris_200'):
        """
        Get the kfactor from the brucker database

        Parameters
        ----------
        microscope_name: str
            name of the microscope

        Examples
        ---------
        >>> s = database.spec1D('TEM')
        >>> s.set_elements(["Ni", "Cr",'Al'])
        >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
        >>> s.get_kfactors_from_brucker()
        """

        xray_lines = self.metadata.Sample.xray_lines
        kfactors_db = database.kfactors_brucker(
            xray_lines=xray_lines, microscope_name=microscope_name)[0]
        self.metadata.Sample.kfactors = kfactors_db

    def get_kfactors_from_first_principles(self,
                                           detector_efficiency=None,
                                           common_lines="Si_Ka",
                                           gateway='auto'):
        """
        Get the kfactors from first principles

        Save them in metadata.Sample.kfactors

        Parameters
        ----------
        detector_efficiency: signals.Spectrum
            The efficiency of the detector
        common_lines: str
            The X-ray lines that is the common denominator. The kfactors of
            these value is one.
        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.


        Examples
        --------
        >>> s = database.spec3D('TEM')
        >>> s.get_kfactors_from_first_principles()
        >>> s.metadata.Sample

        See also
        --------
        utils_eds.get_detector_properties, simulate_two_elements_standard,
        get_link_to_jython

        """
        xrays = ["Si_Ka"] + list(self.metadata.Sample.xray_lines)
        beam_energy = self.metadata.Acquisition_instrument.TEM.beam_energy
        kfactors = []
        kfactors_name = []
        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()
        for i, xray in enumerate(xrays):
            if i != 0:
                kfactors.append(utils_eds.get_kfactors(
                    [xray, xrays[0]], beam_energy=beam_energy,
                    detector_efficiency=detector_efficiency, gateway=gateway))
                kfactors_name.append(xray + '/' + xrays[0])
        self.metadata.Sample.kfactors = kfactors
        self.metadata.Sample.kfactors_name = kfactors_name

# replace by estimate_background_windows
#    def get_two_windows_intensities(self, bck_position):
#        """
#        Quantified for giorgio, 21.05.2014
#
#        Parameters
#        ----------
#        bck_position: list
#            The position of the bck to substract eg [[1.2,1.4],[2.5,2.6]]
#
#        Examples
#        --------
#        >>> s = database.spec3D('TEM')
#        >>> s.set_elements(["Ni", "Cr",'Al'])
#        >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
#        >>> intensities = s.get_two_windows_intensities(
#        >>>      bck_position=[[1.2,3.0],[5.0,5.7],[5.0,9.5]])
#        """
#        if 'Sample.xray_lines' in self.metadata:
#            xray_lines = self.metadata.Sample.xray_lines
#        else:
#            print('Set the Xray lines with set_lines')
#        intensities = []
#        t = self.deepcopy()
#        for i, Xray_line in enumerate(xray_lines):
#            line_energy, line_FWHM = self._get_line_energy(Xray_line,
#                                                           FWHM_MnKa='auto')
#            det = line_FWHM
#            img = self[..., line_energy - det:line_energy + det
#                       ].integrate1D(-1)
#            img1 = self[..., bck_position[i][0] - det:bck_position[i][0] + det
#                        ].integrate1D(-1)
#            img2 = self[..., bck_position[i][1] - det:bck_position[i][1] + det
#                        ].integrate1D(-1)
#            img = img - (img1 + img2) / 2
#            img.metadata.General.title = (
#                'Intensity of %s at %.2f %s from %s' %
#                (Xray_line,
#                 line_energy,
#                 self.axes_manager.signal_axes[0].units,
#                 self.metadata.General.title))
#            intensities.append(img.as_image([0, 1]))
#
#            t[..., line_energy - det:line_energy + det] = 10
#            t[..., bck_position[i][0] - det:bck_position[i][0] + det] = 10
#            t[..., bck_position[i][1] - det:bck_position[i][1] + det] = 10
#        t.plot()
#        return intensities

        # Examples
        #---------
        #>>> s = database.spec3D('TEM')
        #>>> s.set_elements(["Ni", "Cr",'Al'])
        #>>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
        #>>> kfactors = [s.metadata.Sample.kfactors[2],
        #>>>         s.metadata.Sample.kfactors[6]]
        #>>> intensities = s.get_two_windows_intensities(
        #>>>      bck_position=[[1.2,3.0],[5.0,5.7],[5.0,9.5]])
        #>>> res = s.quant_cliff_lorimer_simple(intensities,kfactors)
        #>>> utils.plot.plot_signals(res)

#    def quant_cliff_lorimer(self,
#                            intensities='integrate',
#                            kfactors='auto',
#                            reference_line='auto',
#                            plot_result=True,
#                            store_in_mp=True,
#                            **kwargs):
#        """
#        Quantification using Cliff-Lorimer
#
#        Store the result in metadata.Sample.quant
#
#        Parameters
#        ----------
#        intensities: {'integrate','model',list of signal}
#            If 'integrate', integrate unde the peak using get_lines_intensity
#            if 'model', generate a model and fit it
#            Else a list of intensities (signal or image or spectrum)
#        kfactors: {list of float | 'auto'}
#            the list of kfactor, compared to the first
#            elements. eg. kfactors = [1.47,1.72]
#            for kfactors_name = ['Cr_Ka/Al_Ka', 'Ni_Ka/Al_Ka']
#            with kfactors_name in alphabetical order
#            if 'auto', take the kfactors stored in metadata
#        reference_line: 'auto' or str
#            The reference line. If 'auto', the first line in the alphabetic
#            order is chosen ('Al_Ka' in the previous example.)
#            If reference_line = 'Cr_Ka', then
#            kfactors should be ['Al_Ka/Cr_Ka', 'Ni_Ka/Cr_Ka']
#        plot_result: bool
#          If true (default option), plot the result.
#        kwargs
#            The extra keyword arguments for get_lines_intensity
#
#        Examples
#        ---------
#        >>> s = database.spec3D('TEM')
#        >>> s.set_elements(["Al", "Cr", "Ni"])
#        >>> s.set_lines(["Al_Ka","Cr_Ka", "Ni_Ka"])
#        >>> kfactors = [s.metadata.Sample.kfactors[2],
#        >>>         s.metadata.Sample.kfactors[6]]
#        >>> s.quant_cliff_lorimer(kfactors=kfactors)
#
#        See also
#        --------
#        get_kfactors_from_standard, simulate_two_elements_standard,
#            get_lines_intensity
#
#        """
#        # from hyperspy import signals
#
#        xray_lines = list(self.metadata.Sample.xray_lines)
#
#        # beam_energy = self._get_beam_energy()
#        if intensities == 'integrate':
#            intensities = self.get_lines_intensity(**kwargs)
#        elif intensities == 'model':
#            from hyperspy.hspy import create_model
#            m = create_model(self)
#            m.multifit()
#            intensities = m.get_line_intensities(plot_result=False,
#                                                 store_in_mp=False)
#        if kfactors == 'auto':
#            kfactors = self.metadata.Sample.kfactors
#        indexes = range(len(xray_lines))
#        indexes_f = range(len(xray_lines)-1)
#        if reference_line != 'auto':
#            index = [indexes.pop(xray_lines.index(reference_line))]
#            indexes = index + indexes
#            index = [indexes_f.pop(xray_lines.index(reference_line)-1)]
#            indexes_f = index + indexes_f
#            kfactors_s = [kfactors[i] for i in indexes_f]
#            kfactors_s[0] = 1/kfactors_s[0]
#            for i, index in enumerate(indexes_f):
#                if i != 0:
#                    kfactors_s[i] *= kfactors_s[0]
#        else:
#            kfactors_s = kfactors
#        data_res = utils_eds.quantification_cliff_lorimer(
#            kfactors=kfactors_s,
#            intensities=[intensities[i].data for i in indexes])
#        res = []
#        for data, index in zip(data_res, indexes):
#            res.append(self._set_result(xray_line=xray_lines[index],
#                                        result='quant',
#                                        data_res=data,
#                                        plot_result=plot_result,
#                                        store_in_mp=store_in_mp))
#        if store_in_mp is False:
#            return res

    def quantification(self,
                       intensities,
                       kfactors,
                       composition_units='weight',
                       navigation_mask=1.0,
                       closing=True,
                       plot_result=False,
                       **kwargs):
        """
        Quantification of intensities to return elemental composition

        Method: Cliff-Lorimer

        Parameters
        ----------
        intensities: list of signal
            the intensitiy for each X-ray lines.
        kfactors: list of float
            The list of kfactor in same order as intensities. Note that
            intensities provided by hyperspy are sorted by the aplhabetical
            order of the X-ray lines. eg. kfactors =[0.982, 1.32, 1.60] for
            ['Al_Ka','Cr_Ka', 'Ni_Ka'].
        composition_units: 'weight' or 'atomic'
            Quantification returns weight percent. By choosing 'atomic', the
            return composition is in atomic percent.
        navigation_mask : None or float or signal
            The navigation locations marked as True are not used in the
            quantification. If int is given the vacuum_mask method is used to
            generate a mask with the int value as threhsold.
            Else provides a signal with the navigation shape.
        closing: bool
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        plot_result : bool
            If True, plot the calculated composition. If the current
            object is a single spectrum it prints the result instead.
        kwargs
            The extra keyword arguments are passed to plot.

        Return
        ------
        A list of quantified elemental maps (signal) giving the composition of
        the sample in weight or atomic percent.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> kfactors = [1.450226, 5.075602] #For Fe Ka and Pt La
        >>> bw = s.estimate_background_windows(line_width=[5.0, 2.0])
        >>> s.plot(background_windows=bw)
        >>> intensities = s.get_lines_intensity(background_windows=bw)
        >>> res = s.quantification(intensities, kfactors, plot_result=True,
        >>>                        composition_units='atomic')
        Fe (Fe_Ka): Composition = 15.41 atomic percent
        Pt (Pt_La): Composition = 84.59 atomic percent

        See also
        --------
        vacuum_mask
        """
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        elif navigation_mask is not None:
            navigation_mask = navigation_mask.data
        xray_lines = self.metadata.Sample.xray_lines
        composition = utils.stack(intensities)
        composition.data = utils_eds.quantification_cliff_lorimer(
            composition.data, kfactors=kfactors,
            mask=navigation_mask) * 100.
        composition = composition.split()
        if composition_units == 'atomic':
            composition = utils.material.weight_to_atomic(composition)
        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            composition[i].metadata.General.title = composition_units + \
                ' percent of ' + element
            composition[i].metadata.set_item("Sample.elements", ([element]))
            composition[i].metadata.set_item(
                "Sample.xray_lines", ([xray_line]))
            if plot_result and \
                    composition[i].axes_manager.signal_dimension == 0:
                print("%s (%s): Composition = %.2f %s percent"
                      % (element, xray_line, composition[i].data,
                         composition_units))
        if plot_result and composition[i].axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(composition, **kwargs)
        return composition

    def quantification_old(self,
                           intensities='auto',
                           method='CL',
                           kfactors='auto',
                           composition_units='weight',
                           navigation_mask=1.0,
                           closing=True,
                           plot_result=False,
                           store_in_mp=True,
                           min_intensity = 0.1,
                           **kwargs):
        """
        Quantification using Cliff-Lorimer or zeta-factor method
        
        Parameters
        ----------
        intensities: list of signal
            the intensitiy for each X-ray lines.
        kfactors: list of float
            The list of kfactor (or zfactor) in same order as intensities.
            Note that intensities provided by hyperspy are sorted by the
            aplhabetical order of the X-ray lines.
            eg. kfactors =[0.982, 1.32, 1.60] for ['Al_Ka','Cr_Ka', 'Ni_Ka'].
        method: 'CL' or 'zeta'
            Set the quantification method: Cliff-Lorimer or zeta-factor method
        composition_units: 'weight' or 'atomic'
            Quantification return weight percent. By choosing 'atomic', the
            return composition is in atomic percent.
        navigation_mask : None or float or signal
            The navigation locations marked as True are not used in the
            quantification. If int is given the vacuum_mask method is used to
            generate a mask with the int value as threhsold.
            Else provides a signal with the navigation shape.
        closing: bool
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        plot_result : bool
            If True, plot the calculated composition. If the current
            object is a single spectrum it prints the result instead.
        kwargs
            The extra keyword arguments are passed to plot.

        Return
        ------
        A list of quantified elemental maps (signal) giving the composition of
        the sample in weight or atomic percent.

        Examples
        --------
        >>> s = utils.example_signals.EDS_TEM_Spectrum()
        >>> s.add_lines()
        >>> kfactors = [1.450226, 5.075602] #For Fe Ka and Pt La
        >>> bw = s.estimate_background_windows(line_width=[5.0, 2.0])
        >>> s.plot(background_windows=bw)
        >>> intensities = s.get_lines_intensity(background_windows=bw)
        >>> res = s.quantification(intensities, kfactors, plot_result=True,
        >>>                        composition_units='atomic')
        Fe (Fe_Ka): Composition = 15.41 atomic percent
        Pt (Pt_La): Composition = 84.59 atomic percent

        See also
        --------
        vacuum_mask
        """
        if kfactors == 'auto':
            if method == 'CL':
                kfactors = self.metadata.Sample.kfactors
            elif method == 'zeta':
                kfactors = self.metadata.Sample.zfactors
        if intensities is 'auto':
            intensities = self.metadata.Sample.intensities
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        elif navigation_mask is not None:
            navigation_mask = navigation_mask.data
        xray_lines = self.metadata.Sample.xray_lines
        composition = utils.stack(intensities)
        if method == 'CL':
            composition.data = utils_eds.quantification_cliff_lorimer(
                composition.data, kfactors=kfactors,
                mask=navigation_mask, min_intensity=min_intensity) * 100.
        elif method == 'zeta':
            results = utils_eds.quantification_zeta_factor(
                composition.data, zfactors=kfactors, dose=self.get_dose())
            composition.data = results[0] * 100.
            mass_thickness = intensities[0]
            mass_thickness.data = results[1]
        composition = composition.split()
        if composition_units == 'atomic':
            composition = utils.material.weight_to_atomic(composition)
        for i, xray_line in enumerate(xray_lines):
            element, line = utils_eds._get_element_and_line(xray_line)
            composition[i].metadata.General.title = composition_units + \
                ' percent of ' + element
            composition[i].metadata.set_item("Sample.elements", ([element]))
            composition[i].metadata.set_item(
                "Sample.xray_lines", ([xray_line]))
            if plot_result and \
                    composition[i].axes_manager.signal_dimension == 0:
                print("%s (%s): Composition = %.2f %s percent"
                      % (element, xray_line, composition[i].data,
                         composition_units))
        if plot_result and composition[i].axes_manager.signal_dimension != 0:
            utils.plot.plot_signals(composition, **kwargs)

        if store_in_mp:
            for i, compo in enumerate(composition):
                self._set_result(
                    xray_line=xray_lines[i], result='quant',
                    data_res=compo.data, plot_result=False,
                    store_in_mp=store_in_mp)
            if method == 'zeta':
                self.metadata.set_item("Sample.mass_thickness", mass_thickness)
        return composition

    def get_absorption_corrections(self, weight_fraction='auto',
                                   thickness='auto', density='auto'):
        """
        Compute the absoprtion corrections for each X-ray-lines

        Parameters
        ----------
        weight_fraction: {list of float or signals.Signal or 'auto'}
            Set the weight fraction
            If 'auto', take the weight fraction stored in metadata.Sample.quant
        thickness: {float or 'auto'}
            Set the thickness in nm
            If 'auto', take the thickness stored in metadata.Sample
        density: {float or signals.Signal or 'auto'}
            Set the density. If 'auto', obtain from the compo_at.
            If 'auto', calculate the density with the weight fraction. see
            get_sample_density
        """
        if weight_fraction == 'auto' and 'weight_fraction' in self.metadata.Sample:
            weight_fraction = self.metadata.Sample.quant
        else:
            raise ValueError("Weight fraction needed")
        if thickness == 'auto'and 'thickness' in self.metadata.Sample:
            thickness = self.metadata.Sample.thickness
        else:
            raise ValueError("thickness needed")
        mac_sample = self.get_sample_mass_absorption_coefficient(
            weight_fraction=weight_fraction)
        TOA = self.get_take_off_angle()
        if density == 'auto':
            density = self.get_sample_density(weight_fraction=weight_fraction)
        abs_corr = utils_eds.absorption_correction(
            mac_sample,
            density,
            thickness,
            TOA)
        return abs_corr

    def quantification_absorption_corrections(self,
                                              intensities='integrate',
                                              kfactors='auto',
                                              thickness='auto',
                                              max_iter=50,
                                              atol=1e-3,
                                              plot_result=True,
                                              store_in_mp=True,
                                              all_data=False,
                                              **kwargs):
        """
        Quantification with absorption correction

        using Cliff-Lorimer
        Store the result in metadata.Sample.quant

        Parameters
        ----------
        intensities: {'integrate','model',list of signal}
            If 'integrate', integrate unde the peak using get_lines_intensity
            if 'model', generate a model and fit it
            Else a list of intensities (signal or image or spectrum)
        kfactors: {list of float | 'auto'}
            the list of kfactor, compared to the first
            elements. eg. kfactors = [1.2, 2.5]
            for kfactors_name = ['Al_Ka/Cu_Ka', 'Al_Ka/Nb_Ka']
            if 'auto', take the kfactors stored in metadata
        thickness: {float or 'auto'}
            Set the thickness in nm
            If 'auto', take the thickness stored in metadata.Sample
        plot_result: bool
            If true (default option), plot the result.
        all_data: bool
            if True return only the data in a spectrum
        kwargs
            The extra keyword arguments for get_lines_intensity
        """
        xray_lines = self.metadata.Sample.xray_lines
        elements = self.metadata.Sample.elements
        if thickness == 'auto'and 'thickness' in self.metadata.Sample:
            thickness = self.metadata.Sample.thickness
        else:
            raise ValueError("thickness needed")
        # beam_energy = self._get_beam_energy()
        if intensities == 'integrate':
            intensities = self.get_lines_intensity(**kwargs)
        elif intensities == 'model':
            print 'not checked'
            from hyperspy.hspy import create_model
            m = create_model(self)
            m.multifit()
            intensities = m.get_line_intensities(plot_result=False,
                                                 store_in_mp=False)
        if kfactors == 'auto':
            kfactors = self.metadata.Sample.kfactors
        TOA = self.get_take_off_angle()
        if all_data is False:
            weight_fractions = utils.stack(intensities).as_spectrum(0)
            weight_fractions.map(utils_eds.quantification_absorption_corrections_thin_film,
                                 elements=elements,
                                 xray_lines=xray_lines,
                                 kfactors=kfactors,
                                 TOA=TOA,
                                 thickness=thickness,
                                 max_iter=max_iter,
                                 atol=atol,)
            weight_fractions.metadata._HyperSpy.Stacking_history.axis = -1
            weight_fractions = weight_fractions.split()
            for xray_line, weight_fraction in zip(xray_lines, weight_fractions):
                weight_fraction.metadata.General.title = (
                    'Weight fraction of %s from %s' %
                    (xray_line,
                     self.metadata.General.title))
            if store_in_mp:
                self.metadata.Sample.quant = weight_fractions
            return weight_fractions
        else:
            from hyperspy import signals
            data_res = utils_eds.quantification_absorption_corrections_thin_film(
                elements=elements,
                xray_lines=xray_lines,
                intensities=[intensity.data for intensity in intensities],
                kfactors=kfactors,
                TOA=TOA,
                thickness=thickness,
                max_iter=max_iter,
                atol=atol, all_data=True)
            data_res = signals.Spectrum(data_res).as_spectrum(0)
            return data_res
        # res=[]
        # for xray_line, data in zip(xray_lines,data_res[-1]):
            # res.append(self._set_result(xray_line=xray_line, result='quant',
            # data_res=data,
            # plot_result=plot_result,
            # store_in_mp=store_in_mp))
        # if store_in_mp is False:
            # return res
        # return data_res

    def compute_continuous_xray_absorption(self,
                                           thickness=100,
                                           weight_fraction='auto',
                                           density='auto'):
        """Contninous X-ray Absorption within thin film sample

        Depth distribution of X-ray production is assumed constant

        Parameters
        ----------
        thickness: float
            The thickness in nm
        weight_fraction: list of float
            The sample composition. If 'auto', takes value in metadata.
            If not there, use and equ-composition.
        density: float or 'auto'
            Set the density. in g/cm^3
            if 'auto', calculated from weight_fraction

        See also
        --------
        utils.misc.eds.model.continuous_xray_absorption
        edsmodel.add_background
        """
        spec = self._get_signal_signal()
        spec.metadata.General.title = 'Absorption model (Thin film)'
        if spec.axes_manager.signal_axes[0].units == 'eV':
            units_factor = 1000.
        else:
            units_factor = 1.

        elements = self.metadata.Sample.elements
        TOA = self.get_take_off_angle()
        if weight_fraction == 'auto':
            if 'weight_fraction' in self.metadata.Sample:
                weight_fraction = self.metadata.Sample.weight_fraction
            else:
                weight_fraction = []
                for elm in elements:
                    weight_fraction.append(1. / len(elements))

        if density == 'auto':
            density = self.get_sample_density(weight_fraction=weight_fraction)

        # energy_axis = spec.axes_manager.signal_axes[0]
        # eng = np.linspace(energy_axis.low_value,
            # energy_axis.high_value,
            # energy_axis.size) / units_factor
        eng = spec.axes_manager.signal_axes[0].axis / units_factor
        eng = eng[np.searchsorted(eng, 0.0):]
        spec.data = np.append(np.array([0.0] * (len(spec.data) - len(eng))),
                              physical_model.xray_absorption_thin_film(
                                  energy=eng,
                                  weight_fraction=weight_fraction,
                                  elements=elements,
                                  density=density,
                                  thickness=thickness,
                                  TOA=TOA))
        return spec

    def compute_3D_absorption_correction(self, weight_fraction='auto',
                                         tilt=None,
                                         xray_lines='auto',
                                         thickness='auto',
                                         density='auto',
                                         mask=None,
                                         plot_result=False,
                                         store_result=False,
                                         parallel=None):
        """
        Correct the intensities from absorption knowing the composition in 3D

        Parameters
        ----------
        weight_fraction: list of image or array
            The fraction of elements in the sample by weigh.
            If 'auto' look in quant
        tilt: list of float
            If not None, the weight_fraction is tilted
        thickness: float
            The thickness of each indiviual voxel (square). If 'auto' axes
            manager. in nm.
        density: array
            The density to correct of the sample. If 'auto' use the
            weight_fraction to calculate it. in gm/cm^3
        mask: bool array
            A mask to be applied to the correction absorption
        plot_resut: bool
            plot the result

        Return
        -------
        If store_result True store the result in quant_enh.
        Elif tilt = None return an array of abs_corr
        Else return an array of abs_corr adn an array of tilted intensities
        """
        if hasattr(xray_lines, '__iter__') is False:
            xray_lines = self.metadata.Sample.xray_lines
        elements = self.metadata.Sample.elements
        elevation_angle = self.metadata.Acquisition_instrument.\
            TEM.Detector.EDS.elevation_angle
        azimuth_angle = \
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
        elements = self.metadata.Sample.elements
        if weight_fraction == 'auto':
            weight_fraction = self.metadata.Sample.quant
            weight_fraction = utils.stack(weight_fraction)
            weight_fraction = weight_fraction.data
        ax = self.axes_manager
        if thickness == 'auto':
            thickness = ax.navigation_axes[0].scale * 1e-7
        else:
            thickness = thickness * 1e-7
        if hasattr(tilt, '__iter__'):
            dim = weight_fraction.shape
            arg = {"weight_fraction": weight_fraction,
                   "xray_lines": xray_lines,
                   "elements": elements,
                   "thickness": thickness,
                   "density": density,
                   "elevation_angle": elevation_angle,
                   "mask_el": mask}
            if parallel is None:
                abs_corr = np.zeros([len(xray_lines)] +
                                    [len(tilt)] + list(dim[1:]))
                for i, ti in enumerate(tilt):
                    print ti
                    if hasattr(azimuth_angle, '__iter__'):
                        azim = azimuth_angle[i]
                        if hasattr(azimuth_angle[i], '__iter__'):
                            abs_corr[:, i] = physical_model.\
                                absorption_correction_matrix2(
                                azimuth_angle=azim, tilt=ti, **arg)
                        else:
                            abs_corr[:, i] = physical_model.\
                                absorption_correction_matrix(
                                azimuth_angle=azim, tilt=ti, **arg)
                    else:
                        abs_corr[:, i] = physical_model.\
                            absorption_correction_matrix(
                            azimuth_angle=azim, tilt=ti, **arg)
                return abs_corr
            else:
                from hyperspy.misc import multiprocessing
                pool, pool_type = multiprocessing.pool(parallel)
                args = []
                for ti, azim in zip(tilt, azimuth_angle):
                    args.append({'tilt': ti, 'azimuth_angle': azim})
                    args[-1].update(arg)
                if hasattr(azimuth_angle[0], '__iter__'):
                    abs_corr = np.array(pool.map_sync(
                        multiprocessing.absorption_correction_matrix2, args))
                else:
                    abs_corr = np.array(pool.map_sync(
                        multiprocessing.absorption_correction_matrix, args))
                abs_corr = np.rollaxis(abs_corr, 1, 0)
                return abs_corr
        elif tilt is None:
            abs_corr = physical_model.absorption_correction_matrix(
                weight_fraction=weight_fraction,
                xray_lines=xray_lines,
                elements=elements,
                thickness=thickness,
                density=density,
                azimuth_angle=azimuth_angle,
                elevation_angle=elevation_angle,
                mask_el=mask)

            if store_result:
                for i, xray_line in enumerate(xray_lines):
                    self._set_result(xray_line, "intensities_corr",
                                     abs_corr[i],
                                     plot_result=plot_result)

            else:

                return abs_corr

    def tilt_3D_results(self, tilt, intensities='auto',
                        parallel=None):
        """
        Parameters
        ----------
        intensities: list of image or array
            The intensities to correct of the sample. If 'auto' look in
            intensites
        tilt: list of float

        Return
        ------
        np.array
        """
        if intensities == 'auto':
            intensities = self.metadata.Sample.intensities
            intensities = utils.stack(intensities)
            intensities = intensities.data
        x_ax, z_ax = 3, 1
        dim = intensities.shape
        arg = {"axes": (x_ax, z_ax), "order": 3,
               "reshape": False, "mode": 'reflect'}
        if parallel is None:
            tilt_intensities = np.zeros([dim[0]] + [len(tilt)] + list(dim[1:]))
            for i, ti in enumerate(tilt):
                tilt_intensities[:, i] = ndimage.rotate(intensities, angle=-ti,
                                                        **arg)
        else:
            from hyperspy.misc import multiprocessing
            pool, pool_type = multiprocessing.pool(parallel)
            args = []
            arg['input'] = intensities
            for ti in tilt:
                args.append(arg.copy())
                args[-1]['angle'] = -ti
            tilt_intensities = np.array(pool.map_sync(multiprocessing.rotate,
                                                      args))
            tilt_intensities = np.rollaxis(tilt_intensities, 1, 0)
        return tilt_intensities

    def tilt_3D_result(self,
                       result,
                       tilt):
        """
        Tilt a 3D result

        Parameters
        -----------
        result: 3D image
            The result to tilt
        tilt: list of float
            the tilt angle

        Return
        ------
        np.array

        """
        from scipy import ndimage
        result = result.data.copy()
        result = result.astype("float")
        tilt_result = []
        for i, ti in enumerate(tilt):
            tilt_result.append(ndimage.rotate(result, angle=-ti, axes=(2, 0),
                               order=0, reshape=False, mode='reflect'))
        tilt_result = np.array(tilt_result)
        return tilt_result

    def tomographic_reconstruction_result(self, result,
                                          algorithm='SART',
                                          tilt_stages='auto',
                                          iteration=1,
                                          relaxation=0.15,
                                          parallel=None,
                                          **kwargs):
        """
        Reconstruct all the 3D tomograms from the elements sinogram

        Parameters
        ----------
        result: str
            The result in metadata.Sample to be reconstructed
        algorithm: {'SART','FBP'}
            FBP, filtered back projection
            SART, Simultaneous Algebraic Reconstruction Technique
        tilt_stages: list or 'auto'
            the angles of the sinogram. If 'auto', takes the angles in
            Acquisition_instrument.TEM.tilt_stage
        iteration: int
            The numebr of iteration used for SART
        parallel : {None, int}
            If None or 1, does not parallelise multifit. If >1, will look for
            ipython clusters. If no ipython clusters are running, it will
            create multiprocessing cluster.

        Return
        ------
        The reconstructions as a 3D image

        Examples
        --------
        >>> adf_tilt = database.image3D('tilt_TEM')
        >>> adf_tilt.change_dtype('float')
        >>> rec = adf_tilt.tomographic_reconstruction()
        """

        if isinstance(result, str) is False:
            sinograms = result
            for i in range(len(sinograms)):
                sinograms[i].change_dtype('float')
        elif hasattr(self.metadata.Sample[result], 'metadata'):
            self.metadata.Sample[result].change_dtype('float')
            sinograms = self.metadata.Sample[result].split()
        else:
            sinograms = self.metadata.Sample[result]
            for i in range(len(sinograms)):
                sinograms[i].change_dtype('float')
        if tilt_stages == 'auto':
            tilt_stages = sinograms[0].axes_manager[0].axis
        if hasattr(relaxation, '__iter__') is False:
            relaxation = [relaxation] * len(sinograms)
        if hasattr(iteration, '__iter__') is False:
            iteration = [iteration] * len(sinograms)
        if parallel is None:
            rec = []
            for i, sinogram in enumerate(sinograms):
                rec.append(sinogram.tomographic_reconstruction(
                           algorithm=algorithm,
                           tilt_stages=tilt_stages,
                           iteration=iteration[i],
                           relaxation=relaxation[i],
                           parallel=parallel))
        else:
            rec = utils.stack(sinograms)
            from hyperspy.misc import multiprocessing
            pool, pool_type = multiprocessing.pool(parallel)
            kwargs.update({'theta': tilt_stages})
            data = []
            for i, sinogram in enumerate(sinograms):
                kwargs['relaxation'] = relaxation[i]
                data.append([sinogram.to_spectrum().data, iteration[i],
                             kwargs.copy()])
            res = pool.map_sync(multiprocessing.isart, data)
            if pool_type == 'mp':
                pool.close()
                pool.join()
            rec.data = np.rollaxis(np.array(res), 2, 1)
            rec.axes_manager[0].scale = rec.axes_manager[2].scale
            rec.axes_manager[0].offset = rec.axes_manager[2].offset
            rec.axes_manager[0].units = rec.axes_manager[2].units
            rec.axes_manager[0].name = 'z'
            rec.get_dimensions_from_data()
            rec = rec.split()
        return rec

    def vacuum_mask(self, threshold=1.0, closing=True, opening=False):
        """
        Generate mask of the vacuum region

        Parameters
        ----------
        threshold: float
            For a given pixel, maximum value in the energy axis below which the
            pixel is considered as vacuum.
        closing: bool
            If true, applied a morphologic closing to the mask
        opnening: bool
            If true, applied a morphologic opening to the mask

        Examples
        --------
        >>> # Simulate a spectrum image with vacuum region
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> s_vac = hs.signals.Simulation(np.ones_like(s.data, dtype=float))*0.005
        >>> s_vac.add_poissonian_noise()
        >>> si = hs.stack([s]*3 + [s_vac])
        >>> si.vacuum_mask().data
        array([False, False, False,  True], dtype=bool)

        Return
        ------
        mask: signal
            The mask of the region
        """
        from scipy.ndimage.morphology import binary_dilation, binary_erosion
        mask = (self.max(-1) <= threshold)
        if closing:
            mask.data = binary_dilation(mask.data, border_value=0)
            mask.data = binary_erosion(mask.data, border_value=1)
        if opening:
            mask.data = binary_erosion(mask.data, border_value=1)
            mask.data = binary_dilation(mask.data, border_value=0)
        return mask

    def decomposition(self,
                      normalize_poissonian_noise=True,
                      navigation_mask=1.0,
                      closing=True,
                      *args,
                      **kwargs):
        """
        Decomposition with a choice of algorithms

        The results are stored in self.learning_results

        Parameters
        ----------
        normalize_poissonian_noise : bool
            If True, scale the SI to normalize Poissonian noise
        navigation_mask : None or float or boolean numpy array
            The navigation locations marked as True are not used in the
            decompostion. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threshold.
        closing: bool
            If true, applied a morphologic closing to the maks obtained by
            vacuum_mask.
        algorithm : 'svd' | 'fast_svd' | 'mlpca' | 'fast_mlpca' | 'nmf' |
            'sparse_pca' | 'mini_batch_sparse_pca'
        output_dimension : None or int
            number of components to keep/calculate
        centre : None | 'variables' | 'trials'
            If None no centring is applied. If 'variable' the centring will be
            performed in the variable axis. If 'trials', the centring will be
            performed in the 'trials' axis. It only has effect when using the
            svd or fast_svd algorithms
        auto_transpose : bool
            If True, automatically transposes the data to boost performance.
            Only has effect when using the svd of fast_svd algorithms.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm
        var_func : function or numpy array
            If function, it will apply it to the dataset to obtain the
            var_array. Alternatively, it can a an array with the coefficients
            of a polynomial.
        polyfit :
        reproject : None | signal | navigation | both
            If not None, the results of the decomposition will be projected in
            the selected masked area.

        Examples
        --------
        >>> s = hs.datasets.example_signals.EDS_TEM_Spectrum()
        >>> si = hs.stack([s]*3)
        >>> si.change_dtype(float)
        >>> si.decomposition()

        See also
        --------
        vacuum_mask
        """
        if isinstance(navigation_mask, float):
            navigation_mask = self.vacuum_mask(navigation_mask, closing).data
        super(EDSSpectrum, self).decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask, *args, **kwargs)
        self.learning_results.loadings = np.nan_to_num(
            self.learning_results.loadings)

    def zfactors_from_kfactors(self, kfactors='auto'):
        """
        Provide Zeta factors from the k-factors

        Parameters
        ----------
        zfactors: list of float
            The list of kfactor in same order as intensities. Note that
            intensities provided by hyperspy are sorted by the aplhabetical
            order of the X-ray lines. eg. kfactors =[0.982, 1.32, 1.60] for
            ['Al_Ka','Cr_Ka', 'Ni_Ka'].
        """
        print "Not working"
        if kfactors == 'auto':
            kfactors = self.metadata.Sample.kfactors
        self.metadata.Sample.set_item(
            'zfactors',
            np.array(self.metadata.Sample.kfactors) / constants.N_A)

    def get_dose(self, beam_current='auto', real_time='auto'):
        """
        Return the total electron dose.  given by i*t*N, i the current, t the
        acquisition time, and N the number of electron by unit electric charge.

        Parameters
        ----------
        beam_current:
        real_time:
        """
        parameters = self.metadata.Acquisition_instrument.TEM
        if beam_current == 'auto':
            beam_current = parameters.beam_current
        if real_time == 'auto':
            real_time = parameters.Detector.EDS.real_time
        return real_time * beam_current * 1e-9 / constants.e
