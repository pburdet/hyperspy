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

from hyperspy._signals.eds import EDSSpectrum
from hyperspy.decorators import only_interactive
from hyperspy.gui.eds import TEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.misc.eds import utils as utils_eds
from hyperspy import utils

# TEM spectrum is just a copy of the basic function of SEM spectrum.


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
            'Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa': tem_par.elevation_angle, }

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

    def simulate_binary_standard(self,
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
                + common_element + '_binary'
            std[-1].metadata.Sample.weight_percent = [0.5, 0.5]
        self.metadata.Sample.standard_spec = std

    def get_kfactors_from_standard(self,
                                   common_line='Ka',
                                   **kwargs):
        """
        Exctract the kfactor from binary standard

        Store the kfactor in metadata.sample.kfactors

        Parameters
        ----------

        common_line: str
            The line for the common element to use.

        kwargs
        The extra keyword arguments for get_lines_intensity

        See also
        -------

        simulate_binary_standard, get_lines_intensity

        """
        std_title = self.metadata.Sample.standard_spec[0
                                                       ].metadata.General.title
        if 'binary' not in std_title:
            raise ValueError(
                "Binary standard are needed.")
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
                kfactors.append(kfactor0 / kfactor)
                kfactors_name.append(kfactor0_name + '/' + xray)
        self.metadata.Sample.kfactors = kfactors
        self.metadata.Sample.kfactors_name = kfactors_name

    def quant_cliff_lorimer(self,
                            kfactors='auto',
                            plot_result=True,
                            **kwargs):
        """

        Parameters
        ----------

        kfactors: {list of float | 'auto'}
            the list of kfactor, compared to the first
            elements. eg. kfactors = [1.2, 2.5]
            for kfactors_name = ['Al_Ka/Cu_Ka', 'Al_Ka/Nb_Ka']
            if 'auto', take the kfactors stored in metadata

        plot_result: bool
          If true (default option), plot the result.

        kwargs
            The extra keyword arguments for get_lines_intensity

        See also
        --------

        get_kfactors_from_standard, simulate_binary_standard,
            get_lines_intensity

        """

        xrays = self.metadata.Sample.xray_lines
        beam_energy = self.metadata.Acquisition_instrument.TEM.beam_energy
        intensities = self.get_lines_intensity(**kwargs)

        if kfactors == 'auto':
            kfactors = self.metadata.Sample.kfactors
        ab = []
        for i, kab in enumerate(kfactors):
            # ab = Ia/Ib * kab
            ab.append(intensities[0].data / intensities[i + 1].data * kab)
        # Ca = ab /(1 + ab + ab/ac + ab/ad + ...)
        composition = np.ones(ab[0].shape)
        for i, ab1 in enumerate(ab):
            if i == 0:
                composition += ab[0]
            else:
                composition += (ab[0] / ab1)
        composition = ab[0] / composition
        # Cb = Ca / ab
        for i, xray in enumerate(xrays):
            if i == 0:
                self._set_result(xray_line=xray, result='quant',
                                 data_res=np.nan_to_num(composition),
                                 plot_result=plot_result, store_in_mp=True)
            else:
                self._set_result(xray_line=xray, result='quant',
                                 data_res=np.nan_to_num(composition / ab[i - 1]),
                                 plot_result=plot_result, store_in_mp=True)

    def get_kfactors_from_first_principles(self,
                                           detector_efficiency=None,
                                           gateway='auto'):
        """
        Get the kfactors from first principles

        Parameters
        ----------
        detector_efficiency: signals.Spectrum

        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.

        See also
        --------
        utils_eds.get_detector_properties, simulate_binary_standard,
        get_link_to_jython

        """
        xrays = self.metadata.Sample.xray_lines
        beam_energy = self.metadata.Acquisition_instrument.TEM.beam_energy
        kfactors = []
        kfactors_name = []
        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()
        for i, xray in enumerate(xrays):
            if i != 0:
                kfactors.append(utils_eds.get_kfactors([xrays[0], xray],
                                                       beam_energy=beam_energy,
                                                       detector_efficiency=detector_efficiency,
                                                       gateway=gateway))
                kfactors_name.append(xrays[0] + '/' + xray)
        self.metadata.Sample.kfactors = kfactors
        self.metadata.Sample.kfactors_name = kfactors_name
        
        
    def quant_cliff_lorimer_simple(self,
                            intensities,
                            kfactors):
            """
            Quantified for giorgio, 21.05.2014

            Parameters
            ----------
            kfactors: list of float 
                the list of kfactor, compared to the first
                elements. eg. kfactors = [1.2, 2.5]
                for kfactors_name = ['Al_Ka/Cu_Ka', 'Al_Ka/Nb_Ka']
                
            Examples
            ---------
            >>> s = utils_eds.database_3Dspec('TEM')
            >>> s.set_elements(["Ni", "Cr",'Al'])
            >>> s.set_lines(["Ni_Ka", "Cr_Ka", "Al_Ka"])
            >>> kfactors = [s.metadata.Sample.kfactors[2],
            >>>         s.metadata.Sample.kfactors[6]]
            >>> intensities = s.get_two_windows_intensities(
            >>>      bck_position=[[1.2,3.0],[5.0,5.7],[5.0,9.5]])
            >>> res = s.quant_cliff_lorimer_simple(intensities,kfactors)
            >>> utils.plot.plot_signals(res)
            """

            xrays = self.metadata.Sample.xray_lines
            beam_energy = self.metadata.Acquisition_instrument.TEM.beam_energy
            #intensities = self.get_lines_intensity(**kwargs)

            ab = []
            for i, kab in enumerate(kfactors):
                # ab = Ia/Ib * kab
                ab.append(intensities[0].data / intensities[i + 1].data * kab)
                #signals.Image(ab[-1]).plot()
            # Ca = ab /(1 + ab + ab/ac + ab/ad + ...)
            composition = np.ones(ab[0].shape)
            for i, ab1 in enumerate(ab):
                if i == 0:
                    composition += ab[0]
                else:
                    composition += (ab[0] / ab1)
            composition = ab[0] / composition
            res_compo = []
            # Cb = Ca / ab
            for i, xray in enumerate(xrays):
                if i == 0:
                    data_res=composition            
                else:
                    data_res=composition / ab[i - 1]
                data_res = np.nan_to_num(data_res)
                res_compo.append(intensities[i].deepcopy())
                res_compo[-1].data = data_res
                res_compo[-1].metadata.General.title = 'Composition ' + xray
            return res_compo   
            
    def get_two_windows_intensities(self,bck_position):
        """
        Quantified for giorgio, 21.05.2014
        
        Parameters
        ----------
        bck_position: list
            The position of the bck to substract eg [[1.2,1.4],[2.5,2.6]]
        """   
        if 'Sample.xray_lines' in self.metadata:
            xray_lines = self.metadata.Sample.xray_lines
            print xray_lines
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
            img = img-(img1+img2)/2
            img.metadata.General.title = (
                    'Intensity of %s at %.2f %s from %s' %
                    (Xray_line,
                     line_energy,
                     self.axes_manager.signal_axes[0].units,
                     self.metadata.General.title))
            intensities.append(img)
            
            t[..., line_energy - det:line_energy + det] = 10
            t[..., bck_position[i][0] - det:bck_position[i][0] + det] = 10
            t[..., bck_position[i][1] - det:bck_position[i][1] + det] = 10
        t.plot()
        return intensities
