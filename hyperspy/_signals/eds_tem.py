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

import traits.api as t
import numpy as np
from scipy import ndimage

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
                self.metadata.set_item("Acquisition_instrument.TEM",
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
            mp.set_item("Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                        preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa" not in mp:
            mp.set_item("Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa",
                        preferences.EDS.eds_mn_ka)
        if "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item("Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
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
                "Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa",
                energy_resolution_MnKa)

        if set([beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa]) == {None}:
            self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        tem_par = TEMParametersUI()
        mapping = {
            'Acquisition_instrument.TEM.beam_energy': 'tem_par.beam_energy',
            'Acquisition_instrument.TEM.tilt_stage': 'tem_par.tilt_stage',
            'Acquisition_instrument.TEM.Detector.EDS.live_time': 'tem_par.live_time',
            'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle': 'tem_par.azimuth_angle',
            'Acquisition_instrument.TEM.Detector.EDS.elevation_angle': 'tem_par.elevation_angle',
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa': 'tem_par.energy_resolution_MnKa', }
        for key, value in mapping.iteritems():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        tem_par.edit_traits()

        mapping = {
            'Acquisition_instrument.TEM.beam_energy': tem_par.beam_energy,
            'Acquisition_instrument.TEM.tilt_stage': tem_par.tilt_stage,
            'Acquisition_instrument.TEM.Detector.EDS.live_time': tem_par.live_time,
            'Acquisition_instrument.TEM.Detector.EDS.azimuth_angle': tem_par.azimuth_angle,
            'Acquisition_instrument.TEM.Detector.EDS.elevation_angle': tem_par.elevation_angle,
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa': tem_par.energy_resolution_MnKa, }

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
        """

        self.original_metadata = ref.original_metadata.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units
        ax_m.offset = ax_ref.offset

        # if hasattr(self.original_metadata, 'CHOFFSET'):
        #ax_m.scale = ref.original_metadata.CHOFFSET
        # if hasattr(self.original_metadata, 'OFFSET'):
        #ax_m.offset = ref.original_metadata.OFFSET
        # if hasattr(self.original_metadata, 'XUNITS'):
        #ax_m.units = ref.original_metadata.XUNITS
        # if hasattr(self.original_metadata, 'CHOFFSET'):
        # if self.original_metadata.XUNITS == 'keV':
        #ax_m.scale = ref.original_metadata.CHOFFSET / 1000

        # Setup metadata
        if 'Acquisition_instrument.TEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.TEM
        elif 'Acquisition_instrument.SEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.SEM
        else:
            raise ValueError("The reference has no metadata.Acquisition_instrument.TEM"
                             "\n nor metadata.Acquisition_instrument.SEM ")

        mp = self.metadata

        mp.Acquisition_instrument.TEM = mp_ref.deepcopy()

        # if hasattr(mp_ref, 'tilt_stage'):
        #mp.Acquisition_instrument.SEM.tilt_stage = mp_ref.tilt_stage
        # if hasattr(mp_ref, 'beam_energy'):
        #mp.Acquisition_instrument.SEM.beam_energy = mp_ref.beam_energy
        # if hasattr(mp_ref.EDS, 'energy_resolution_MnKa'):
        #mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa = mp_ref.EDS.energy_resolution_MnKa
        # if hasattr(mp_ref.EDS, 'azimuth_angle'):
        #mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle = mp_ref.EDS.azimuth_angle
        # if hasattr(mp_ref.EDS, 'elevation_angle'):
        #mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle = mp_ref.EDS.elevation_angle

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
            atomic_percent = np.array(utils.material.weight_to_atomic(
                el_binary, [0.5, 0.5]))
            std_met.set_elements(el_binary)
            std_met.set_lines(xray_binary)
            std.append(utils_eds.simulate_one_spectrum_TEM(nTraj,
                                                           dose=dose, mp=std_met.metadata,
                                                           detector=detector, compo_at=atomic_percent,
                                                           gateway=gateway))
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
        kfactors_name = []
        for i, (std, el, xray) in enumerate(zip(self.metadata.Sample.standard_spec,
                                                self.metadata.Sample.elements,
                                                self.metadata.Sample.xray_lines)):
            intens = std.get_lines_intensity([xray] + [common_xray], **kwargs)
            kfactor = intens[1].data / intens[0].data
            if i == 0:
                kfactor0 = kfactor
                kfactor0_name = xray
            else:
                kfactors.append(kfactor / kfactor0)
                kfactors_name.append(xray + '/' + kfactor0_name)
        self.metadata.Sample.kfactors = kfactors
        self.metadata.Sample.kfactors_name = kfactors_name

    def get_kfactors_from_brucker(self, common_line='first',
                                  microscope_name='osiris_200'):
        """
        Get the kfactor from the brucker database

        Parameters
        ----------
        common_line: str
            The line for the common element to use ('Al_Ka').
        microscope_name: str
            name of the microscope

        Examples
        ---------
        >>> s = database.spec1D('TEM')
        >>> s.set_elements(["Ni", "Cr",'Al'])
        >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
        >>> s.get_kfactors_from_osiris_database()
        """
        kfactors = []
        kfactors_name = []
        xray_lines = self.metadata.Sample.xray_lines
        kfactors_db = database.kfactors_brucker(xray_lines=xray_lines,
                                                microscope_name=microscope_name)[0]
        if common_line == 'first':
            xray_line0 = xray_lines[0]
            kfactor0 = kfactors_db[0]
        else:

            xray_line0 = xray_lines[xray_lines.index(common_line)]
            kfactor0 = kfactors_db[xray_lines.index(common_line)]

        for iline, xray_line in enumerate(xray_lines):
            kfactor = kfactors_db[iline]
            #kfactorer = kfactors_db[elem][line]['kfactor_error']
            if xray_line != xray_line0:
                kfactors.append(kfactor / kfactor0)
                kfactors_name.append(xray_line + '/' + xray_line0)
        self.metadata.Sample.kfactors = kfactors
        self.metadata.Sample.kfactors_name = kfactors_name

    def quant_cliff_lorimer(self,
                            intensities='integrate',
                            kfactors='auto',
                            plot_result=True,
                            store_in_mp=True,
                            **kwargs):
        """
        Quantification using Cliff-Lorimer

        Store the result in metadata.Sample.quant

        Parameters
        ----------
        intensities: {'integrate','model',list of signal}
            If 'integrate', integrate unde the peak using get_lines_intensity
            if 'model', generate a model and fit it
            Else a list of intensities (signal or image or spectrum)
        kfactors: {list of float | 'auto'}
            the list of kfactor, compared to the first
            elements. eg. kfactors = [1.47,1.72]
            for kfactors_name = ['Cr_Ka/Al_Ka', 'Ni_Ka/Al_Ka']
            with kfactors_name in alphabetical order
            if 'auto', take the kfactors stored in metadata
        plot_result: bool
          If true (default option), plot the result.
        kwargs
            The extra keyword arguments for get_lines_intensity

        Examples
        ---------
        >>> s = database.spec3D('TEM')
        >>> s.set_elements(["Al", "Cr", "Ni"])
        >>> s.set_lines(["Al_Ka","Cr_Ka", "Ni_Ka"])
        >>> kfactors = [s.metadata.Sample.kfactors[2],
        >>>         s.metadata.Sample.kfactors[6]]
        >>> s.quant_cliff_lorimer(kfactors=kfactors)

        See also
        --------
        get_kfactors_from_standard, simulate_two_elements_standard,
            get_lines_intensity

        """
        #from hyperspy import signals

        xray_lines = self.metadata.Sample.xray_lines
        #beam_energy = self._get_beam_energy()
        if intensities == 'integrate':
            intensities = self.get_lines_intensity(**kwargs)
        elif intensities == 'model':
            from hyperspy.hspy import create_model
            m = create_model(self)
            m.multifit()
            intensities = m.get_line_intensities(plot_result=False,
                                                 store_in_mp=False)
        if kfactors == 'auto':
            kfactors = self.metadata.Sample.kfactors
        data_res = utils_eds.quantification_cliff_lorimer(
            kfactors=kfactors,
            intensities=[intensity.data for intensity in intensities])
        res = []
        for xray_line, data in zip(xray_lines, data_res):
            res.append(self._set_result(xray_line=xray_line, result='quant',
                                        data_res=data,
                                        plot_result=plot_result,
                                        store_in_mp=store_in_mp))
        if store_in_mp is False:
            return res

    def get_kfactors_from_first_principles(self,
                                           detector_efficiency=None,
                                           gateway='auto'):
        """
        Get the kfactors from first principles

        Save them in metadata.Sample.kfactors

        Parameters
        ----------
        detector_efficiency: signals.Spectrum

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
        xrays = self.metadata.Sample.xray_lines
        beam_energy = self._get_beam_energy()
        kfactors = []
        kfactors_name = []
        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()
        for i, xray in enumerate(xrays):
            if i != 0:
                kfactors.append(utils_eds.get_kfactors([xray, xrays[0]],
                                                       beam_energy=beam_energy,
                                                       detector_efficiency=detector_efficiency,
                                                       gateway=gateway))
                kfactors_name.append(xray + '/' + xrays[0])
        self.metadata.Sample.kfactors = kfactors
        self.metadata.Sample.kfactors_name = kfactors_name

    def get_two_windows_intensities(self, bck_position):
        """
        Quantified for giorgio, 21.05.2014

        Parameters
        ----------
        bck_position: list
            The position of the bck to substract eg [[1.2,1.4],[2.5,2.6]]

        Examples
        --------
        >>> s = database.spec3D('TEM')
        >>> s.set_elements(["Ni", "Cr",'Al'])
        >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
        >>> intensities = s.get_two_windows_intensities(
        >>>      bck_position=[[1.2,3.0],[5.0,5.7],[5.0,9.5]])
        """
        if 'Sample.xray_lines' in self.metadata:
            xray_lines = self.metadata.Sample.xray_lines
        else:
            print('Set the Xray lines with set_lines')
        intensities = []
        t = self.deepcopy()
        for i, Xray_line in enumerate(xray_lines):
            line_energy, line_FWHM = self._get_line_energy(Xray_line,
                                                           FWHM_MnKa='auto')
            det = line_FWHM
            img = self[..., line_energy - det:line_energy + det
                       ].integrate1D(-1)
            img1 = self[..., bck_position[i][0] - det:bck_position[i][0] + det
                        ].integrate1D(-1)
            img2 = self[..., bck_position[i][1] - det:bck_position[i][1] + det
                        ].integrate1D(-1)
            img = img - (img1 + img2) / 2
            img.metadata.General.title = (
                'Intensity of %s at %.2f %s from %s' %
                (Xray_line,
                 line_energy,
                 self.axes_manager.signal_axes[0].units,
                 self.metadata.General.title))
            intensities.append(img.as_image([0, 1]))

            t[..., line_energy - det:line_energy + det] = 10
            t[..., bck_position[i][0] - det:bck_position[i][0] + det] = 10
            t[..., bck_position[i][1] - det:bck_position[i][1] + det] = 10
        t.plot()
        return intensities

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

    def quantification_cliff_lorimer(self,
                                     intensities,
                                     kfactors):
        """
        Quantification using Cliff-Lorimer

        Parameters
        ----------
        kfactors: list of float
            the list of kfactor, compared to the first
            elements. eg. kfactors = [1.47,1.72]
            for kfactors_name = ['Cr_Ka/Al_Ka', 'Ni_Ka/Al_Ka']
            with kfactors_name in alphabetical order
        intensities: list of signal.Signals
            the intensities for each X-ray lines.

        Examples
        ---------
        >>> #s is a signals.EDSTEMSpectrum
        >>> s.set_elements(["Al", "Cr", "Ni"])
        >>> s.set_lines(["Al_Ka","Cr_Ka", "Ni_Ka"])
        >>> kfactors = [1.47,1.72]
        >>> intensities = s.get_lines_intensity()
        >>> res = s.quantification_cliff_lorimer(intensities,kfactors)
        >>> utils.plot.plot_signals(res)
        """

        xray_lines = self.metadata.Sample.xray_lines
        data_res = utils_eds.quantification_cliff_lorimer(
            kfactors=kfactors,
            intensities=[intensity.data for intensity in intensities])
        spec_res = []
        for xray_line, data, intensity in zip(
                xray_lines, data_res, intensities):
            element, line = utils_eds._get_element_and_line(xray_line)
            spec_res.append(intensity.deepcopy())
            spec_res[-1].data = data
            spec_res[-1].metadata.General.title = 'Weight fraction of ' + \
                element
        return spec_res

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

    def correct_intensities_from_absorption(self, weight_fraction='auto',
                                            intensities='auto',
                                            tilt=None,
                                            thickness='auto',
                                            density='auto',
                                            mask=None,
                                            plot_result=False,
                                            store_result=False):
        """
        Correct the intensities from absorption knowing the composition in 3D

        Parameters
        ----------
        weight_fraction: list of image or array
            The fraction of elements in the sample by weigh.
            If 'auto' look in quant
        intensities: list of image or array
            The intensities to correct of the sample. If 'auto' look in
            intensites
        tilt: list of float
            If not None, the weight_fraction is tilted
        thickness: float
            The thickness of each indiviual voxel (square). If 'auto' axes
            manager
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
        xray_lines = self.metadata.Sample.xray_lines
        elements = self.metadata.Sample.elements
        elevation_angle = self.metadata.Acquisition_instrument.\
            TEM.Detector.EDS.elevation_angle
        azimuth_angle = \
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
        elements = self.metadata.Sample.elements
        xray_lines = self.metadata.Sample.xray_lines
        if weight_fraction == 'auto':
            weight_fraction = self.metadata.Sample.quant
            weight_fraction = utils.stack(weight_fraction)
            weight_fraction = weight_fraction.data
        if intensities == 'auto':
            intensities = self.metadata.Sample.intensities
            intensities = utils.stack(intensities)
            intensities = intensities.data
        ax = self.axes_manager
        if thickness == 'auto':
            thickness = ax.navigation_axes[0].scale * 1e-7
        else:
            thickness = thickness * 1e-7
        if hasattr(tilt, '__iter__'):
            x_ax, z_ax = 3, 1
            dim = intensities.shape
            tilt_intensities = np.zeros([dim[0]] + [len(tilt)] + list(dim[1:]))
            abs_corr = np.zeros([dim[0]] + [len(tilt)] + list(dim[1:]))
            azim = azimuth_angle
            for i, ti in enumerate(tilt):
                print ti
                if hasattr(azimuth_angle, '__iter__'):
                    azim = azimuth_angle[i]
                tilt_intensities[:, i] = ndimage.rotate(intensities, angle=-ti,
                                                        axes=(x_ax, z_ax),
                                                        order=3, reshape=False,
                                                        mode='reflect')
                abs_corr[:, i] = physical_model.absorption_correction_matrix(
                    weight_fraction=weight_fraction,
                    xray_lines=xray_lines,
                    elements=elements,
                    thickness=thickness,
                    density=density,
                    azimuth_angle=azim,
                    elevation_angle=elevation_angle,
                    mask_el=mask,
                    tilt=ti)
            return abs_corr, tilt_intensities
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
                    data = intensities[i] / abs_corr[i]

                    self._set_result(xray_line, "intensities_corr", data,
                                     plot_result=plot_result)

            else:

                return abs_corr

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

        if hasattr(self.metadata.Sample[result], 'metadata'):
            self.change_dtype('float')
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
