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

import numpy as np
import traits.api as t
import os
import codecs
import subprocess
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import execnet
import copy

from hyperspy._signals.eds import EDSSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.decorators import only_interactive
from hyperspy.io import load
import hyperspy.components as components
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds import image_eds
from hyperspy import utils
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.config_dir import config_path, os_name, data_path
from hyperspy.drawing.utils import animate_legend
from hyperspy.misc.eds import physical_model


class EDSSEMSpectrum(EDSSpectrum):
    _signal_type = "EDS_SEM"

    def __init__(self, *args, **kwards):
        EDSSpectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        if 'Acquisition_instrument.SEM.Detector.EDS' not in self.metadata:
            if 'Acquisition_instrument.TEM' in self.metadata:
                self.metadata.set_item(
                    "Acquisition_instrument.SEM",
                    self.metadata.Acquisition_instrument.TEM)
                del self.metadata.Acquisition_instrument.TEM
        self._set_default_param()

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
        >>> ref = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> s = hs.signals.EDSSEMSpectrum(
        >>>     hs.datasets.example_signals.EDS_SEM_Spectrum().data)
        >>> print s.axes_manager[0].scale
        >>> s.get_calibration_from(ref)
        >>> print s.axes_manager[0].scale
        1.0
        0.01

        """

        self.original_metadata = ref.original_metadata.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units
        ax_m.offset = ax_ref.offset

        # Setup metadata
        if 'Acquisition_instrument.SEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.SEM
        elif 'Acquisition_instrument.TEM' in ref.metadata:
            mp_ref = ref.metadata.Acquisition_instrument.TEM
        else:
            raise ValueError(
                "The reference has no metadata.Acquisition_instrument.TEM"
                "\n nor metadata.Acquisition_instrument.SEM ")

        mp = self.metadata

        mp.Acquisition_instrument.SEM = mp_ref.deepcopy()

        if hasattr(mp_ref.Detector.EDS, 'live_time'):
            mp.Acquisition_instrument.SEM.Detector.EDS.live_time = \
                mp_ref.Detector.EDS.live_time / nb_pix

    def _load_from_TEM_param(self):
        """Transfer metadata.Acquisition_instrument.TEM to
        metadata.Acquisition_instrument.SEM

        """

        mp = self.metadata
        if mp.has_item('Acquisition_instrument.SEM') is False:
            mp.add_node('Acquisition_instrument.SEM')
        if mp.has_item('Acquisition_instrument.SEM.Detector.EDS') is False:
            mp.Acquisition_instrument.SEM.add_node('EDS')
        mp.Signal.signal_type = 'EDS_SEM'

        # Transfer
        if 'Acquisition_instrument.TEM' in mp:
            mp.Acquisition_instrument.SEM = mp.Acquisition_instrument.TEM
            del mp.Acquisition_instrument.TEM

    def _set_default_param(self):
        """Set to value to default (defined in preferences)

        """
        mp = self.metadata
        if "Acquisition_instrument.SEM.tilt_stage" not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.tilt_stage",
                preferences.EDS.eds_tilt_stage)
        if "Acquisition_instrument.SEM.Detector.EDS.elevation_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.elevation_angle",
                preferences.EDS.eds_detector_elevation)
        if "Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa" \
                not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Detector.EDS."
                "energy_resolution_MnKa",
                preferences.EDS.eds_mn_ka)
        if "Acquisition_instrument.SEM.Detector.EDS.azimuth_angle" not in mp:
            mp.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.azimuth_angle",
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
        >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
        >>> print('Default value %s eV' %
        >>>       s.metadata.Acquisition_instrument.
        >>>       SEM.Detector.EDS.energy_resolution_MnKa)
        >>> s.set_microscope_parameters(energy_resolution_MnKa=135.)
        >>> print('Now set to %s eV' %
        >>>       s.metadata.Acquisition_instrument.
        >>>       SEM.Detector.EDS.energy_resolution_MnKa)
        Default value 130.0 eV
        Now set to 135.0 eV

        """
        md = self.metadata

        if beam_energy is not None:
            md.set_item("Acquisition_instrument.SEM.beam_energy", beam_energy)
        if live_time is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.live_time",
                live_time)
        if tilt_stage is not None:
            md.set_item("Acquisition_instrument.SEM.tilt_stage", tilt_stage)
        if azimuth_angle is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.azimuth_angle",
                azimuth_angle)
        if elevation_angle is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS.elevation_angle",
                elevation_angle)
        if energy_resolution_MnKa is not None:
            md.set_item(
                "Acquisition_instrument.SEM.Detector.EDS."
                "energy_resolution_MnKa",
                energy_resolution_MnKa)

        if {beam_energy, live_time, tilt_stage, azimuth_angle,
                elevation_angle, energy_resolution_MnKa} == {None}:
            self._are_microscope_parameters_missing()

    @only_interactive
    def _set_microscope_parameters(self):
        from hyperspy.gui.eds import SEMParametersUI
        tem_par = SEMParametersUI()
        mapping = {
            'Acquisition_instrument.SEM.beam_energy': 'tem_par.beam_energy',
            'Acquisition_instrument.SEM.tilt_stage': 'tem_par.tilt_stage',
            'Acquisition_instrument.SEM.Detector.EDS.live_time':
            'tem_par.live_time',
            'Acquisition_instrument.SEM.Detector.EDS.azimuth_angle':
            'tem_par.azimuth_angle',
            'Acquisition_instrument.SEM.Detector.EDS.elevation_angle':
            'tem_par.elevation_angle',
            'Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa':
            'tem_par.energy_resolution_MnKa', }

        for key, value in mapping.iteritems():
            if self.metadata.has_item(key):
                exec('%s = self.metadata.%s' % (value, key))
        tem_par.edit_traits()

        mapping = {
            'Acquisition_instrument.SEM.beam_energy': tem_par.beam_energy,
            'Acquisition_instrument.SEM.tilt_stage': tem_par.tilt_stage,
            'Acquisition_instrument.SEM.Detector.EDS.live_time':
            tem_par.live_time,
            'Acquisition_instrument.SEM.Detector.EDS.azimuth_angle':
            tem_par.azimuth_angle,
            'Acquisition_instrument.SEM.Detector.EDS.elevation_angle':
            tem_par.elevation_angle,
            'Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa':
            tem_par.energy_resolution_MnKa, }

        for key, value in mapping.iteritems():
            if value != t.Undefined:
                self.metadata.set_item(key, value)
        self._are_microscope_parameters_missing()

    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in metadata. If not, in interactive mode
        raises an UI item to fill the values

        """
        import hyperspy.gui.messages as messagesui
        must_exist = (
            'Acquisition_instrument.SEM.beam_energy',
            'Acquisition_instrument.SEM.Detector.EDS.live_time', )

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

    def link_standard(self, std_folder, std_file_extension='msa'):
        """
        Seek for standard spectra (spectrum recorded on known composition
        sample) in the std_file folder and link them to the analyzed
        elements of 'metadata.Sample.elements'. A standard
        spectrum is linked if its file name contains the elements name.
        "C.msa", "Co.msa" but not "Co4.msa".

        Store the standard spectra in 'metadata.Sample.standard_spec'


        Parameters
        ----------------
        std_folder: path name
            The path of the folder containing the standard file.

        std_file_extension: extension name
            The name of the standard file extension.

        See also
        --------
        set_elements, add_elements

        """

        if not hasattr(self.metadata, 'Sample'):
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.metadata.Sample, 'elements'):
            raise ValueError("Add elements first, see 'set_elements'")
        # if not hasattr(self.metadata.Sample, 'xray_lines'):
            #raise ValueError("Add lines first, see 'set_lines'")

        std_tot = load(
            std_folder +
            "//*." +
            std_file_extension,
            signal_type='EDS_SEM')
        mp = self.metadata
        mp.Sample.standard_spec = []
        for element in mp.Sample.elements:
        # for xray_line in mp.Sample.xray_lines:
            #element, line = utils_eds._get_element_and_line(xray_line)
            test_file_exist = False
            for std in std_tot:
                mp_std = std.metadata
                if 'General.original_filename' in mp_std:
                    filename = mp_std.General.original_filename
                else:
                    filename = mp_std.General.title
                if element + "." in filename or element + '_std' == filename:
                    test_file_exist = True
                    # print("Standard file for %s : %s" % (element,
                    #  filename))
                    mp_std.General.title = element + "_std"
                    mp.Sample.standard_spec.append(std)
            if test_file_exist == False:
                print("\nStandard file for %s not found\n" % element)

    def _get_kratio(self, xray_lines, plot_result):
        """
        Calculate the k-ratio without deconvolution
        """
        from hyperspy.hspy import create_model
        width_windows = 0.75
        mp = self.metadata

        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            std = self.get_result(element, 'standard_spec')
            mp_std = std.metadata
            line_energy = self._get_line_energy(xray_line)
            diff_ltime = mp.Acquisition_instrument.SEM.Detector.EDS.live_time / \
                mp_std.Acquisition_instrument.SEM.Detector.EDS.live_time
            # Fit with least square
            m = create_model(
                self.top_hat(line_energy, width_windows),
                auto_background=False, auto_add_lines=False)
            fp = components.ScalableFixedPattern(std.top_hat(line_energy,
                                                             width_windows))
            fp.set_parameters_not_free(['xscale', 'shift'])
            m.append(fp)
            m.multifit(fitter='leastsq')
            # store k-ratio
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result(xray_line, 'kratios',
                                 fp.yscale.value / diff_ltime, plot_result)
            else:
                self._set_result(xray_line, 'kratios',
                                 fp.yscale.as_signal().data / diff_ltime, plot_result)

    # do it with EDS model
    def get_kratio(self, deconvolution=None, plot_result=True):
        """
        Calculate the k-ratios by least-square fitting of the standard
        sepectrum after background substraction with a top hat filtering

        Return a display of the resutls and store them in
        'metadata.Sample.k_ratios'

        Parameters
        ----------
        plot_result : bool
            If True (default option), plot the k-ratio map.

        See also
        --------
        set_elements, link_standard, top_hat

        """

        if not hasattr(self.metadata, 'Sample'):
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.metadata.Sample, 'elements'):
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.metadata.Sample, 'standard_spec'):
            raise ValueError("Add Standard, see 'link_standard'")

        mp = self.metadata
        mp.Sample.kratios = list(np.zeros(len(mp.Sample.xray_lines)))
        xray_lines = list(mp.Sample.xray_lines)

        if deconvolution is not None:
            for deconvo in deconvolution:
                if len(deconvo) == 3:
                    self._deconvolve_kratio(deconvo[0], deconvo[1], deconvo[2],
                                            plot_result=plot_result)
                else:
                    self._deconvolve_kratio(deconvo[0], deconvo[1],
                                            deconvo[2], top_hat_applied=deconvo[3], plot_result=plot_result)
                for xray_line in deconvo[0]:
                    xray_lines.remove(xray_line)
        if len(xray_lines) > 0:
            self._get_kratio(xray_lines, plot_result)

    def _deconvolve_kratio(self,
                           xray_lines,
                           elements,
                           width_energy,
                           top_hat_applied=True,
                           plot_result=True):
        """
        Calculate the k-ratio, applying a fit on a larger region with
        selected X-ray lines
        """

        from hyperspy.hspy import create_model
        line_energy = np.mean(width_energy)
        width_windows = [line_energy - width_energy[0],
                         width_energy[1] - line_energy]
        if top_hat_applied:
            m = create_model(self.top_hat(line_energy, width_windows), auto_background=False,
                             auto_add_lines=False)
        else:
            m = create_model(self[..., width_energy[0]:width_energy[1]], auto_background=False,
                             auto_add_lines=False)
        mp = self.metadata

        diff_ltime = []
        fps = []
        for element in elements:
            std = self.get_result(element, 'standard_spec')
            if top_hat_applied:
                fp = components.ScalableFixedPattern(std.top_hat(line_energy,
                                                                 width_windows))
            else:
                fp = components.ScalableFixedPattern(
                    std[width_energy[0]:width_energy[1]])
            fp.set_parameters_not_free(['offset', 'xscale', 'shift'])
            fps.append(fp)
            m.append(fps[-1])
            diff_ltime.append(mp.Acquisition_instrument.SEM.Detector.EDS.live_time /
                              std.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time)
        m.multifit(fitter='leastsq')
        i = 0
        for xray_line in xray_lines:
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result(xray_line, 'kratios',
                                 fps[i].yscale.value / diff_ltime[i], plot_result)
            else:
                self._set_result(xray_line, 'kratios',
                                 fps[i].yscale.as_signal().data /
                                 diff_ltime[i],
                                 plot_result)
            i += 1

    def check_kratio(self, xray_lines, width_energy='auto',
                     top_hat_applied=False,
                     plot_all_standard=False,
                     plot_legend=True,
                     new_figure=True):
        """
        Compare the spectrum to the sum of the standard spectra scaled
        in y with the k-ratios. The residual is ploted as well.

        Works only for spectrum.

        Parameters
        ----------

        xray_lines: list of string
            the X-ray lines to display.

        width_windows: 'auto' | [min energy, max energy]
            Set the width of the display windows If 'auto'
            (default option), the display is adjusted to the higest/lowest
            energy line.

        top_hat_applied: boolean
            If True, apply the top hat to all spectra

        plot_all_standard: boolean
            If True, plot all standard spectra

        """

        if width_energy == 'auto':
            [line_energy, resolution] = zip(*[self._get_line_energy(xray_line, 130)
                                              for xray_line in xray_lines])
            width_energy = [np.min(line_energy) - np.min(resolution) * 2,
                            np.max(line_energy) + np.max(resolution) * 2]

        line_energy = np.mean(width_energy)
        width_windows = [line_energy - width_energy[0], width_energy[1]
                         - line_energy]

        mp = self.metadata
        if new_figure:
            fig = plt.figure()
        if top_hat_applied:
            self_data = self.top_hat(line_energy, width_windows).data
        else:
            self_data = self[width_energy[0]:width_energy[1]].data
        plt.plot(self_data)
        leg_plot = ["Spec"]
        line_energies = []
        intensities = []
        spec_sum = np.zeros(len(self.top_hat(line_energy,
                                             width_windows).data))
        for xray_line in xray_lines:
            element, line = utils_eds._get_element_and_line(xray_line)
            line_energy = self._get_line_energy(xray_line)

            width_windows = [line_energy - width_energy[0], width_energy[1] -
                             line_energy]

            std_spec = self.get_result(element, 'standard_spec')
            kratio = self.get_result(xray_line, 'kratios').data
            diff_ltime = mp.Acquisition_instrument.SEM.Detector.EDS.live_time /\
                std_spec.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time
            if top_hat_applied:
                std_data = std_spec.top_hat(line_energy,
                                            width_windows).data * kratio * diff_ltime
            else:
                std_data = std_spec[width_energy[0]:width_energy[1]].data\
                    * kratio * diff_ltime
            if plot_all_standard:
                plt.plot(std_data)
                leg_plot.append(xray_line)
            line_energies.append((line_energy - width_energy[0]) /
                                 self.axes_manager[0].scale - self.axes_manager[0].offset)
            intensities.append(std_data[int(line_energies[-1])])
            spec_sum = spec_sum + std_data
        plt.plot(spec_sum)
        plt.plot(self_data - spec_sum)
        leg_plot.append("Sum")
        leg_plot.append("Residual")
        if plot_legend:
            plt.legend(leg_plot)
        print("Tot residual: %s" % np.abs(self_data - spec_sum).sum())
        for i in range(len(line_energies)):
            plt.annotate(xray_lines[i], xy=(line_energies[i],
                                            intensities[i]))
        if new_figure:
            fig.show()
            # return fig
            animate_legend(fig)

    # shouldn't be needed
    # In fact, why not
   

    def quant(
            self, plot_result=True, enh=False, enh_param=[
                0, 0.001, 0.01, 49, 1],
            compiler=0):
        """
        Quantify using stratagem, a commercial software. A licence is
        needed.

        k-ratios needs to be calculated before. Return a display of the
        results and store them in 'metadata.Sample.quants'

        Parameters
        ----------
        plot_result: bool
            If true (default option), plot the result.

        enh: bool
            If True, used the enhanced quantification (need 3D data)

        enh_param: list of float
            Parameter needed for the enh quantifiacation:
            [method, limit_kratio_0,limit_comp_same,iter_max]
            1. The method to use
            1. Limit to consider a k-ratio equivalent to null
            2. Limit to consider two compositions equivalent
            3. Number maximum of iteration of the thin-film quantification

        compiler: (0|1|2)
            use different the same script in another folder
            ('.hyperspy/stratquant1' or '.hyperspy/stratquant') for processing
            in parallel.

        Return
        ------

        A signal containing the mass fraction (composition) unnormalized.

        See also
        --------
        set_elements, link_standard, top_hat, get_kratio

        """
        mp = self.metadata
        if enh is False:
            if len(mp.Sample.xray_lines) != len(mp.Sample.elements):
                raise ValueError("Only one X-ray lines should be " +
                                 "attributed per element.")
                return 0
            if compiler == 0:
                foldername = os.path.join(config_path, 'strata_quant//')
            else:
                foldername = os.path.join(config_path,
                                          'strata_quant' + str(compiler) + '//')

            self._write_nbData_tsv(foldername + 'essai')
        elif enh is True and self.axes_manager.navigation_dimension == 3:
            if mp.has_item('elec_distr') is False:
                raise ValueError(" Simulate an electron distribution first " +
                                 "with simulate_electron_distribution.")
                return 0
            if compiler == 0:
                foldername = os.path.join(config_path, 'strata_quant_enh//')
            else:
                foldername = os.path.join(config_path,
                                          'strata_quant_enh' + str(compiler) + '//')
            if mp.has_item('enh_param') is False:
                mp.add_node('enh_param')
            mp.enh_param['method'] = enh_param[0]
            mp.enh_param['limit_kratio_0'] = enh_param[1]
            mp.enh_param['limit_comp_same'] = enh_param[2]
            mp.enh_param['iter_max'] = enh_param[3]
            mp.enh_param['extra_param'] = enh_param[4]
            self._write_nbData_ehn_tsv(foldername + 'essai')
        else:
            raise ValueError("Ehnanced quantification needs 3D data.")
            return 0
        self._write_donnee_tsv(foldername + 'essai')
        p = subprocess.Popen(foldername + 'Debug//essai.exe')
        p.wait()
        self._read_result_tsv(foldername + 'essai', plot_result, enh=enh)

    def _read_result_tsv(self, foldername, plot_result, enh):
        encoding = 'latin-1'
        mp = self.metadata
        f = codecs.open(foldername + '//result.tsv', encoding=encoding,
                        errors='replace')
        #dim = list(self.data.shape)
        xray_lines = mp.Sample.xray_lines
        elements = mp.Sample.elements
        nbElem = len(elements)
        #dim = list(self.axes_manager.navigation_shape)[::-1]
        dim = np.copy(self.get_result(xray_lines[0],
                                      'kratios').data.shape).tolist()
        raw_data = []
        for el in elements:
            raw_data.append([])
        for line in f.readlines():
            for i in range(nbElem):
                raw_data[i].append(float(line.split()[3 + i]))
        f.close()
        i = 0
        if enh:
            mp.Sample.quant_enh = list(np.zeros(nbElem))
        else:
            mp.Sample.quant = list(np.zeros(nbElem))
        for el in elements:
            if (self.axes_manager.navigation_dimension == 0):
                data_quant = raw_data[i][0]
            elif (self.axes_manager.navigation_dimension == 1):
                data_quant = np.array(raw_data[i]).reshape((dim[0]))
            elif (self.axes_manager.navigation_dimension == 2):
                data_quant = np.array(raw_data[i]).reshape((dim[1], dim[0])).T
            elif (self.axes_manager.navigation_dimension == 3):
                data_quant = np.array(raw_data[i]).reshape((dim[2], dim[1],
                                                           dim[0])).T
            if enh:
                data_quant = data_quant[::, ::-1]
                self._set_result(el, 'quant_enh', data_quant, plot_result)
            else:
                self._set_result(el, 'quant', data_quant, plot_result)
            i += 1

    def read_enh_ouput(self, compiler=0):
        """
        read the iter, rho, error
        """
        from hyperspy import signals
        if compiler == 0:
            foldername = os.path.join(config_path, 'strata_quant_enh//essai')
        else:
            foldername = os.path.join(config_path,
                                      'strata_quant_enh' + str(compiler) + '//essai')
        encoding = 'latin-1'
        mp = self.metadata
        f = codecs.open(foldername + '//result.tsv', encoding=encoding,
                        errors='replace')

        xray_lines = mp.Sample.xray_lines

        dim = np.copy(self.get_result(xray_lines[0],
                                      'kratios').data.shape).tolist()
        raw_data = []
        for el in [-3, -2, -1]:
            raw_data.append([])
        for line in f.readlines():
            for i in range(3):
                raw_data[i].append(float(line.split()[[-3, -2, -1][i]]))
        f.close()
        axes_res = self.axes_manager.deepcopy()
        axes_res.remove(-1)

        data_tot = []
        for i in range(3):
            data_quant = np.array(raw_data[i]).reshape((dim[2], dim[1],
                                                       dim[0])).T
            data_quant = data_quant[::, ::-1]
            data_quant = signals.Image(data_quant)
            data_quant.axes_manager = axes_res
            data_tot.append(data_quant)
        data_tot = utils.stack(data_tot, new_axis_name='iter_rho_error')
        data_tot = data_tot.as_image([0, 1])
        data_tot.get_dimensions_from_data()
        return data_tot

    def _write_donnee_tsv(self, foldername):
        encoding = 'latin-1'
        mp = self.metadata
        xray_lines = mp.Sample.xray_lines
        elements = mp.Sample.elements
        f = codecs.open(foldername + '//donnee.tsv', 'w',
                        encoding=encoding, errors='ignore')
        dim = np.copy(self.get_result(xray_lines[0],
                                      'kratios').data.shape).tolist()
        #dim = np.copy(self.axes_manager.navigation_shape).tolist()
        # dim = np.copy(self.get_result(xray_lines[0],
        #    'kratios').axes_manager.navigation_shape).tolist()
        # dim.reverse()
        if self.axes_manager.navigation_dimension == 0:
            f.write("1_1\r\n")
            for i in range(len(xray_lines)):
                f.write("%s\t" % mp.Sample.kratios[i].data)
        elif self.axes_manager.navigation_dimension == 1:
            for x in range(dim[0]):
                y = 0
                f.write("%s_%s\r\n" % (x + 1, y + 1))
                for xray_line in xray_lines:
                    f.write("%s\t" % self.get_result(xray_line,
                                                     'kratios').data[x])
                f.write('\r\n')
        elif self.axes_manager.navigation_dimension == 2:
            for x in range(dim[1]):
                for y in range(dim[0]):
                    f.write("%s_%s\r\n" % (x + 1, y + 1))
                    for xray_line in xray_lines:
                        f.write("%s\t" % self.get_result(xray_line,
                                                         'kratios').data[y, x])
                    f.write('\r\n')
        elif self.axes_manager.navigation_dimension == 3:
            for x in range(dim[2]):
                for y in range(dim[1]):
                    f.write("%s_%s\r\n" % (x + 1, y + 1))
                    for z in range(dim[0]):
                        # for elm in elements: # Inverse order line
                        #    for xray_line in xray_lines[::-1]:  #
                        #        if elm + '_' in xray_line:#
                        # f.write("%s\t" % self.get_result(elm +
                        # xray_line[2::],
                        for xray_line in xray_lines:
                            f.write(
                                "%s\t" %
                                self.get_result(xray_line, 'kratios').data[z, y, x])
                        f.write('\r\n')
        f.close()

    def _write_nbData_tsv(self, foldername):
        encoding = 'latin-1'
        mp = self.metadata
        f = codecs.open(foldername + '//nbData.tsv', 'w',
                        encoding=encoding, errors='ignore')

        xray_lines = mp.Sample.xray_lines
        dim = np.copy(self.get_result(xray_lines[0],
                                      'kratios').data.shape).tolist()
        dim.reverse()
        #dim = np.copy(self.axes_manager.navigation_shape).tolist()
        # dim.reverse()
        dim.append(1)
        dim.append(1)
        dim.append(1)
        # if dim[0] == 0:
        #    dim[0] =1
        f.write("nbpixel_x\t%s\r\n" % dim[0])
        f.write('nbpixel_y\t%s\r\n' % dim[1])
        f.write('nbpixel_z\t%s\r\n' % dim[2])
        #f.write('pixelsize_z\t%s' % self.axes_manager[0].scale*1000)
        f.write('pixelsize_z\t100\r\n')
        f.write('nblayermax\t5\r\n')
        f.write('Limitkratio0\t0.00001\r\n')
        f.write('Limitcompsame\t0.00001\r\n')
        f.write('Itermax\t49\r\n')
        f.write('\r\n')
        f.write('HV\t%s\r\n' % mp.Acquisition_instrument.SEM.beam_energy)
        f.write(
            'Elevation\t%s\r\n' %
            mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle)
        f.write(
            'azimuth\t%s\r\n' %
            mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle)
        f.write('tilt\t%s\r\n' % mp.Acquisition_instrument.SEM.tilt_stage)
        f.write('\r\n')
        f.write('nbelement\t%s\r\n' % len(xray_lines))
        elements = 'Element'
        z_el = 'Z'
        line_el = 'line'
        for xray_line in xray_lines:
            el, line = utils_eds._get_element_and_line(xray_line)
            elements = elements + '\t' + el
            z_el = z_el + '\t' + \
                str(elements_db[el]['General_properties']['Z'])
            if line == 'Ka':
                line_el = line_el + '\t0'
            if line == 'La':
                line_el = line_el + '\t1'
            if line == 'Ma':
                line_el = line_el + '\t2'
        f.write('%s\r\n' % elements)
        f.write('%s\r\n' % z_el)
        f.write('%s\r\n' % line_el)
        f.close()

    def _write_nbData_ehn_tsv(self, foldername):
        encoding = 'latin-1'
        mp = self.metadata
        f = codecs.open(foldername + '//nbData.tsv', 'w',
                        encoding=encoding, errors='ignore')
        xray_lines = mp.Sample.xray_lines
        dim = np.copy(self.get_result(xray_lines[0],
                                      'kratios').data.shape).tolist()
        dim.reverse()
        #dim = np.copy(self.axes_manager.navigation_shape).tolist()
        distr_dic = self.metadata.elec_distr
        scale = []
        for ax in self.axes_manager.navigation_axes:
            scale.append(ax.scale * 1000)

        elements = mp['Sample']['elements']
        limit_x = distr_dic['limit_x']
        dx0 = distr_dic['dx0']
        dx_increment = distr_dic['dx_increment']
        stat = distr_dic['distr']
        #pixSize = self.axes_manager[2].scale
        pixLat = int((limit_x[1] - limit_x[0]) / dx0 + 1)

        distres = []
        for el, elm in enumerate(elements):
            distres.append([])
            for i, distr in enumerate(stat[el]):
                length = int(
                    (limit_x[1] - limit_x[0]) / (dx0 * (dx_increment * i + 1)))
                distr = distr[int(pixLat / 2. - round(length / 2.)):
                              int(pixLat / 2. + int(length / 2.))]
                if sum(distr) != 0:
                    distres[el].append([x / sum(distr) for x in distr])

        f.write(
            "v2_\t%s\t%s\t0.1\r\n" %
            (mp.enh_param['method'], mp.enh_param['extra_param']))
        f.write("nbpixel_xyz\t%s\t%s\t%s\r\n" % (dim[0], dim[1], dim[2]))
        f.write(
            'pixelsize_xyz\t%s\t%s\t%s\r\n' %
            (scale[0], scale[1], scale[2]))
        f.write('nblayermax\t%s\r\n' % max(distr_dic.max_slice_z))
        f.write('Limitkratio0\t%s\r\n' % mp.enh_param['limit_kratio_0'])
        f.write('Limitcompsame\t%s\r\n' % mp.enh_param['limit_comp_same'])
        f.write('Itermax\t%s\r\n' % mp.enh_param['iter_max'])
        f.write('\r\n')
        f.write('HV\t%s\r\n' % mp.Acquisition_instrument.SEM.beam_energy)
        f.write(
            'Elevation\t%s\r\n' %
            mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle)
        f.write(
            'azimuth\t%s\r\n' %
            mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle)
        f.write('tilt\t%s\r\n' % mp.Acquisition_instrument.SEM.tilt_stage)
        f.write('\r\n')
        f.write('nbelement\t%s\t%s\r\n' % (len(elements), len(xray_lines)))
        el_str = 'Element'
        z_el = 'Z'
        line_el = 'line'
        for elm in elements:
            el_str = el_str + '\t' + elm
            z_el = z_el + '\t' + \
                str(elements_db[elm]['General_properties']['Z'])
            i = 0
            line_el = line_el + '\t'
            # for xray_line in xray_lines[::-1]:   #Inverse order line
            for xray_line in xray_lines:
                if elm + '_' in xray_line:
                    tmp, line = utils_eds._get_element_and_line(xray_line)
                    if i == 1:
                        line_el = line_el + '_'
                    if line == 'Ka':
                        line_el = line_el + '0'
                    if line == 'La':
                        line_el = line_el + '1'
                    if line == 'Ma':
                        line_el = line_el + '2'
                    i = 1
            #el, line = utils_eds._get_element_and_line(xray_line)
            # if line == 'Ka':
                #line_el = line_el + '\t0'
            # if line== 'La':
                #line_el = line_el + '\t1'
            # if line == 'Ma':
                #line_el = line_el + '\t2'
        f.write('%s\r\n' % el_str)
        f.write('%s\r\n' % z_el)
        f.write('%s\r\n' % line_el)
        f.write('\r\n')
        f.write('DistrX_Min_Max_Dx_IncF\t%s\t%s\t%s\t%s\r\n' %
                (limit_x[0] * 1000, limit_x[1] * 1000, dx0 * 1000, dx_increment))
        f.write('DistrZ_Size_nbforelems')
        for slice_z in distr_dic.max_slice_z:
            f.write('\t%s' % slice_z)
        f.write('\r\n')
        f.write('\r\n')
        for el in range(len(elements)):
            for z in range(distr_dic.max_slice_z[el]):
                for x in distres[el][z]:
                    f.write("%s\t" % x)
                f.write('\r\n')
            f.write('\r\n')
        f.close()

        elements = mp['Sample']['elements']
        limit_x = distr_dic['limit_x']
        dx0 = distr_dic['dx0']
        dx_increment = distr_dic['dx_increment']
        stat = distr_dic['distr']
        #pixSize = self.axes_manager[2].scale
        pixLat = int((limit_x[1] - limit_x[0]) / dx0 + 1)

    #.as_signal slow
    # background, better physical model...
    # link with EDSmodel
    def simulate_model(self, elemental_map='random'):
        """
        Simulate a model, given by

        Parameters
        ----------
        elemental_map: {'random',None, signals.Image}
            map
            
        Example
        -------
        >>> s = signals.EDSSEMSpectrum(range(1024))
        >>> s.axes_manager[-1].scale = 0.01
        >>> s.axes_manager[-1].units = "keV"
        >>> s.axes_manager[-1].offset = -0.1
        >>> s.set_microscope_parameters(beam_energy=15, live_time=10)
        >>> s.set_elements(['Al', 'Zn'])
        >>> s.simulate_model()
        """
        from hyperspy._signals.image import Image
        from hyperspy.hspy import create_model

        live_time = self.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time
        beam_energy = self._get_beam_energy()
        FWHM_MnKa = self.metadata.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa
        energy_axis = self.axes_manager.signal_axes[0]
        elements = self.metadata.Sample.elements

        #tilt = np.radians(mp.Acquisition_instrument.SEM.tilt_stage)
        #elevation = mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle
        # azim = mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle)

        if "counts_rate" in self.metadata.Acquisition_instrument.SEM.Detector.EDS:
            counts_rate = self.metadata.Acquisition_instrument.SEM.Detector.EDS.counts_rate
        else:
            counts_rate = 10000
        elemental_map_shape = list(
            [len(elements)] + list(self.data.shape[:-1]))

        if "weight_percents" in self.metadata.Sample:
            weight_percents = self.metadata.Sample.weight_percents
            elemental_map = None
        else:
            weight_percents = [100] * len(elements)

        if isinstance(elemental_map, Image):
            if list(elemental_map.data.shape) != elemental_map_shape:
                raise ValueError(
                    "elemental _map doesn't have the good size")
                return
            elemental_map = elemental_map.data
        elif elemental_map == 'random':
            elemental_map = np.random.random(elemental_map_shape)
        elif elemental_map is None:
            elemental_map = np.ones(elemental_map_shape)

        m = create_model(self, auto_background=False,
                         auto_add_lines=False)
        for i, (element, weight_percent) in enumerate(zip(elements, weight_percents)):

            for line in utils.material.elements[element].Atomic_properties.Xray_lines.keys():
                line_energy = utils.material.elements[element
                                                      ].Atomic_properties.Xray_lines[line].energy_keV
                ratio_line = utils.material.elements[element
                                                     ].Atomic_properties.Xray_lines[line].weight
                if line_energy < beam_energy:
                    g = components.Gaussian()
                    g.centre.value = line_energy
                    g.sigma.value = utils_eds.get_FWHM_at_Energy(
                        FWHM_MnKa,
                        line_energy) / 2.355
                    g.A.value = live_time * counts_rate * \
                        weight_percent / 100 * ratio_line
                    #g.A.value = live_time*ratio_line*weight_percent/100
                    m.append(g)
                    g.A.map['values'][:] = g.A.value * elemental_map[i]
                    g.A.map['is_set'][:] = True

        if set(self.data.flatten()) == set([0]):
            self.data = m.as_signal().data
            self.add_poissonian_noise()
        else:
            s = self.deepcopy()
            s.data = m.as_signal().data
            s.add_poissonian_noise()
            return s

    def simulate_electron_distribution(self,
                                       nb_traj,
                                       limit_x,
                                       dx0,
                                       dx_increment,
                                       detector='Si(Li)',
                                       plot_result=False,
                                       gateway='auto'):
        # works on the angle. Define always the same..
        """"
        Simulate a the electron distribution in each layer z using DTSA-II

        Parameters
        ----------
        nb_traj: int
            number of electron trajectories.
        limit_x: list of float
            Parameters to define the grid system for the simulation :
            Min and max in x [mum].
        dx0: float
            Parameter to define the grid system for the simulation :
            voxel size iny for the upper layer [mum].
        dx_increment: list of float
            Parameter to define the grid system for the simulation :
            Increment in y voxel at each subsequent layer.
        detector: str
            Give the detector name defined in DTSA-II.
        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.
        plot_result: bool
            If true (default option), plot the result.

        Return
        ------
        The number of electron in each place of the grid and the position
        of each grid.

        Notes
        -----
        Micron need to be used

        """

        mp = self.metadata
        #dic = self.deepcopy().metadata.as_dictionary()
        if hasattr(mp.Sample, 'elements') is False:
            raise ValueError('Elements needs to be defined')
            return 0

        elements = list(mp.Sample.elements)
        e0 = self._get_beam_energy()
        tilt = np.radians(mp.Acquisition_instrument.SEM.tilt_stage)

        # Units!!
        pixSize = [dx0 * 1.0e-6, 0.2 * 1.0e-6,
                   self.axes_manager[2].scale * 1.0e-6]
        nblayer = []
        for el in elements:
            nblayer.append(utils.eds.electron_range(el, e0, tilt=tilt))

        nblayermax = int(round(max(nblayer) / self.axes_manager[2].scale))
        pixLat = [int((limit_x[1] - limit_x[0]) / dx0 + 1), nblayermax]
        dev = (limit_x[1] + limit_x[0]) * 1.0e-6

        if 1 == 0:
            # AlZn nTraj = 20000
            dx_increment = 0.75
            # pixLat = nbx nbz
            limit_x = [-250, 300]
            pixSize = [4 * 1.0e-9, 200 * 1.0e-9, 40 * 1.0e-9]  # y,x,z
            # (max -min)/y +1 (ou 0.5), maxeldepth #nb de pixel
            pixLat = [138, 7]
            dev = 50 * 1.0e-9  # min+max, deviation du centre
        if 1 == 0:
            # TiFeNi
            limit_x = [-350, 450]
            dx_increment = 0.5
            pixSize = (8 * 1.0e-9, 200 * 1.0e-9, 100 * 1.0e-9)
            pixLat = (100, 5)
            dev = 100 * 1.0e-9

        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()
        channel = gateway.remote_exec("""
            import dtsa2
            import math
            epq = dtsa2.epq
            epu = dtsa2.epu
            nm = dtsa2.nm

            elements = """ + str(elements) + """
            elms = []
            for element in elements:
                elms.append(getattr(dtsa2.epq.Element,element))
            e0 = """ + str(e0) + """
            #needs to be changed
            tilt = -""" + str(tilt) + """
            #tiltD = tilt
            #if tilt < 0:
                #tilt cannot be negative
                #tiltD = -tiltD

            nTraj = """ + str(nb_traj) + """
            IncrementF = """ + str(dx_increment) + """
            pixSize = """ + str(pixSize) + """
            pixLat = """ + str(pixLat) + """
            dev = """ + str(dev) + """
            pixTot = pixLat[0]*pixLat[1]

            det = dtsa2.findDetector('""" + detector + """')
            origin = epu.Math2.multiply(1.0e-3,
                epq.SpectrumUtils.getSamplePosition(det.getProperties()))
            z0 = origin[2]

            for el in elms :

                # Create a simulator and initialize it
                monteb = nm.MonteCarloSS()
                monteb.setBeamEnergy(epq.ToSI.keV(e0))

                # top substrat
                mat=epq.MaterialFactory.createPureElement(el)
                monteb.addSubRegion(monteb.getChamber(),
                    mat, nm.MultiPlaneShape.createSubstrate(
                    [math.sin(tilt),0.0,-math.cos(tilt)], origin) )

                #Create Shape
                center0=epu.Math2.plus(origin,
                    [-math.cos(tilt)*(pixLat[0]-1)/2.0*pixSize[0]-math.sin(tilt)*pixSize[2]/2,0,
                    -math.sin(tilt)*(pixLat[0]-1)/2.0*pixSize[0]+math.cos(tilt)*pixSize[2]/2])
                stats=range(pixTot)
                k=0
                for z in range (pixLat[1]):
                    center0=epu.Math2.plus(origin,
                        [-math.cos(tilt)*((pixLat[0]-1)*pixSize[0]*(1+IncrementF*z)-dev)/
                        2.0-math.sin(tilt)*pixSize[2]*(1/2+z),0,
                        -math.sin(tilt)*((pixLat[0]-1)*pixSize[0]*(1+IncrementF*z)-dev)/
                        2.0+math.cos(tilt)*pixSize[2]*(1/2+z)])
                    for y in range (pixLat[0]):
                        stats[k] = nm.ElectronStatsListener(monteb,
                            -5e-3, 5e-3, 1,-5e-3, 5e-3, 1,-5e-3, 5e-3, 1)
                        cub1=nm.MultiPlaneShape.createBlock([pixSize[0],
                            pixSize[1],pixSize[2]],center0, 0.0, -tilt,0.0)

                        stats[k].setShape(cub1)
                        monteb.addActionListener(stats[k])
                        k=k+1
                        center0=epu.Math2.plus(center0,
                            [pixSize[0]*math.cos(tilt)*(1+IncrementF*z),
                            0.0,pixSize[0]*math.sin(tilt)*(1+IncrementF*z)])

                # Reset the detector and run the electrons
                det.reset()
                monteb.runMultipleTrajectories(nTraj)

                k = 0
                for z in range (pixLat[1]):
                    for x in range (pixLat[0]):
                        channel.send(stats[k].getSpatialDistribution().maxValue())
                        k=k+1

        """)

        datas = []
        for i, item in enumerate(channel):
            datas.append(item)

        i = 0
        stat = np.zeros([len(elements), pixLat[1], pixLat[0]])
        for el, elm in enumerate(elements):
            for z in range(pixLat[1]):
                for x in range(pixLat[0]):
                    stat[el, z, x] = datas[i]
                    i = i + 1
        distres = []
        xdatas = []
        for el, elm in enumerate(elements):
            distres.append([])
            xdatas.append([])
            if plot_result:
                f = plt.figure()
                leg = []
            for i, distr in enumerate(stat[el]):
                length = int(
                    (limit_x[1] - limit_x[0]) / (dx0 * (dx_increment * i + 1)))
                distr = distr[int(pixLat[0] / 2. - round(length / 2.)):
                              int(pixLat[0] / 2. + int(length / 2.))]
                if sum(distr) != 0:
                    xdata = []
                    for x in range(length):
                        xdata.append(
                            limit_x[0] + x * dx0 * (dx_increment * i + 1))
                    if plot_result:
                        leg.append(
                            'z slice ' + str(pixSize[2] * 1.0e6 * i) + ' ${\mu}m$')
                        plt.plot(xdata, distr)
                    distres[el].append([x / sum(distr) for x in distr])
                    xdatas[el].append(xdata)

            if plot_result:
                plt.legend(leg, loc=2)
                plt.title(elm + ': Electron depth distribution')
                plt.xlabel('x position [${\mu}m$]')
                plt.ylabel('nb electrons')
                animate_legend()

        if mp.has_item('elec_distr') is False:
            mp.add_node('elec_distr')
        mp.elec_distr['limit_x'] = limit_x
        mp.elec_distr['dx0'] = dx0
        mp.elec_distr['dx_increment'] = dx_increment
        max_slice_z = []
        for el, elm in enumerate(elements):
            max_slice_z.append(len(distres[el]))
        mp.elec_distr['max_slice_z'] = max_slice_z
        mp.elec_distr['distr'] = stat
        mp.elec_distr['nb_traj'] = nb_traj

        return distres, xdatas

    def plot_electron_distribution(self, elements='all', max_depth='auto'):
        """Retrieve and plot the electron distribution from
        simulate_electron_distribution
        """

        mp = self.metadata

        if mp.has_item('elec_distr') is False:
            raise ValueError(" Simulate an electron distribution first " +
                             "with simulate_electron_distribution.")
            return 0

        elem_list = mp['Sample']['elements']
        limit_x = mp.elec_distr['limit_x']
        dx0 = mp.elec_distr['dx0']
        dx_increment = mp.elec_distr['dx_increment']
        stat = mp.elec_distr['distr']
        nb_traj = mp.elec_distr['nb_traj']

        pixSize = self.axes_manager[2].scale
        pixLat = int((limit_x[1] - limit_x[0]) / dx0 + 1)

        for el, elm in enumerate(elem_list):

            if elements == 'all' or elements == elm:
                f = plt.figure()
                leg = []
                for i, distr in enumerate(stat[el]):
                    if i > max_depth:
                        break
                    length = int(
                        (limit_x[1] - limit_x[0]) / (dx0 * (dx_increment * i + 1)))
                    distr = distr[int(pixLat / 2. - round(length / 2.)):
                                  int(pixLat / 2. + int(length / 2.))]
                    xdata = []
                    for x in range(length):
                        xdata.append(
                            limit_x[0] + x * dx0 * (dx_increment * i + 1))
                    leg.append('z slice ' + str(pixSize * i) + ' ${\mu}m$')
                    plt.plot(xdata, distr)

                plt.legend(leg, loc=1)
                plt.title(elm + ': Electron depth distribution (nb traj :'
                          + str(nb_traj) + ')')
                plt.xlabel('x position [${\mu}m$]')
                plt.ylabel('nb electrons / sum electrons in the layer')
                animate_legend()

        if elements != 'all':
            return f



    def quant_with_DTSA(self,
                        detector='Si(Li)',
                        gateway='auto'):
        """calcul the composition from a set of kratios.

        Parameters
        ----------

        detector: str
            Give the detector name defined in DTSA-II

        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.
        """
        mp = self.metadata
        if hasattr(mp.Sample, 'xray_lines'):
            xray_lines = mp.Sample.xray_lines
            xrts = []
            elements = []
            for xray_line in xray_lines:
                el, li = utils_eds._get_element_and_line(xray_line)
                elements.append(el)
                if li == 'Ka':
                    xrts.append(u'K\u03b1')
                elif li == 'La':
                    xrts.append(u'L\u03b1')
                elif li == 'Ma':
                    xrts.append(u'M\u03b1')
                else:
                    raise ValueError('xray_lines not translated')
                    return 0
        else:
            raise ValueError('xray_lines need to be defined')
            return 0

        if hasattr(mp.Sample, 'kratios') is False:
            raise ValueError('kratios need to be defined')
            return 0

        e0 = self._get_beam_energy()
        tilt = np.radians(mp.Acquisition_instrument.SEM.tilt_stage)
        elevation = np.radians(
            mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle)
        azim = np.radians(
            90 -
            mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle)

        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()

        dim = self.get_result(xray_lines[0], 'kratios').data.shape

        def _quant_with_dtsa(kratios):
            channel = gateway.remote_exec("""
                import dtsa2
                import math
                epq = dtsa2.epq

                lim_kratio=0.0001

                #Element and k-ratios
                kratiosI = """ + str(kratios) + """
                elmsI = """ + str(elements) + """
                xrtsI = """ + str(xrts) + """
                elms = []
                kratios = []
                xrts = []
                for i, elm in enumerate(elmsI):
                    if kratiosI[i] > lim_kratio:
                        elms.append(getattr(epq.Element,elm))
                        kratios.append(kratiosI[i])
                        xrts.append(xrtsI[i])

                #Microscope parameters
                e0 =""" + str(e0) + """
                tilt = """ + str(tilt) + """
                elevation = """ + str(elevation) + """
                azim = """ + str(azim) + """

                det = dtsa2.findDetector('""" + detector + """')

                #Define spectrum properties
                specprops = epq.SpectrumProperties()
                specprops.setNumericProperty(epq.SpectrumProperties.BeamEnergy,e0)
                specprops.setDetectorPosition(elevation, azim, 0.005, 2e-5)

                specprops.setSampleShape(
                    epq.SpectrumProperties.SampleShape,
                    epq.SampleShape.Bulk([0.0,math.sin(tilt),-math.cos(tilt)]))

                #Define quantification
                quant = epq.CompositionFromKRatios()
                kratiosSet =  epq.KRatioSet()

                for i, elm in enumerate(elms):
                    transSet = epq.XRayTransitionSet(elm,xrts[i])
                    quant.addStandard(transSet, epq.Composition(elm), specprops)
                    kratiosSet.addKRatio(transSet, kratios[i])

                quant.setConvergenceCriterion(0.001)
                quant.setMaxIterations(50)

                #Compute
                has_converged = True
                quant.compute(kratiosSet,specprops)
                #try:
                #    quant.compute(kratiosSet,specprops)
                #except:
                #    has_converged = False
                #    print "do not converge"

                #get result
                comp = quant.getResult()
                a =  quant.getCorrectionAlgorithm()

                for i, elm in enumerate(elmsI):
                    if has_converged == False:
                        channel.send(kratiosI[i])
                    elif kratiosI[i] > lim_kratio:
                        elm_epq = getattr(epq.Element,elm)
                        channel.send(comp.weightFraction(elm_epq, 0))
                    else:
                        channel.send(0)
                for i, elm in enumerate(elmsI):
                    if has_converged == False:
                        for j in range(4):
                            channel.send(1)
                    elif kratiosI[i] > lim_kratio:
                        elm_epq = getattr(epq.Element,elm)
                        for j in range(4):
                            channel.send(a.relativeZAF(comp,
                                epq.XRayTransitionSet(elm_epq,xrtsI[i]).getWeighiestTransition(),
                                specprops)[j])
                    else:
                        for j in range(4):
                            channel.send(1)
            """)

            comp = []
            ZAF = []
            for i, item in enumerate(channel):
                if i < len(elements):
                    comp.append(item)
                else:
                    ZAF.append(item)

            ZAF = np.array(ZAF)
            ZAF = np.reshape(ZAF, [len(elements), 4])

            return comp, ZAF

        if len(dim) == 0:
            kratios = []
            for xray_line in xray_lines:
                kratios.append(
                    float(self.get_result(xray_line, 'kratios').data))

            comp, ZAF = _quant_with_dtsa(kratios)

            return comp, ZAF
        elif len(dim) == 2:
            comp_tot = []
            ZAF_tot = []
            for y in range(dim[0]):
                for x in range(dim[1]):
                    kratios = []
                    for krat in mp.Sample.kratios:
                        kratios.append(float(krat[x, y].data[0]))
                    comp, ZAF = _quant_with_dtsa(kratios)
                    ZAF_tot.append(ZAF)
                    comp_tot.append(comp)

            return comp_tot, ZAF_tot

        else:
            raise ValueError('Dimension for suported yet')
            return 0
    # to be improved, colors are the same

    def plot_3D_iso_surface_result(self, elements, result, thresholds,
                                   outline=True,
                                   colors=None,
                                   figure=None,
                                   tv_denoise=False,
                                   **kwargs):
        """
        Generate an iso-surface in Mayavi.

        Parameters
        ----------

        elements: str or list
            The element to select.
        result: str
            The name of the result, or an image in 3D.
        threshold: float or list
            The threshold value(s) used to generate the contour(s).
            Between 0 (min intensity) and 1 (max intensity).
        colors: 'auto' or list of (r,g,b)
            'Auto' generate different color
        figure: None or mayavi.core.scene.Scene
            If None, generate a new scene/figure.
        outline: bool
            If True, draw an outline.
        tv_denoise:
            denoise the data
        kwargs:
            other keyword arguments of mlab.pipeline.iso_surface (eg.
            'color=(R,G,B)','name=','opacity=','transparent=',...)

        Examples
        --------

        >>> s = database.result3D()
        >>> fig,src,iso = s.plot_3D_iso_surface_result(['Hf','Ta','Ni'],'quant',
        >>>     [0.8,0.8,0.3])
        >>> # Change the threshold of the second iso-surface
        >>> iso[1].contour.contours = [0.1,]

        Return
        ------

        figure: mayavi.core.scene.Scene

        srcs: list of mayavi.sources.array_source.ArraySource

        isos: list of mayavi.modules.iso_surface.IsoSurface

        """
        if isinstance(elements, list):
            if isinstance(thresholds, list) is False:
                thresholds = [thresholds] * len(elements)
        elif isinstance(thresholds, list):
            if isinstance(elements, list) is False:
                elements = [elements] * len(thresholds)
        else:
            elements = [elements]
            thresholds = [thresholds]
        if isinstance(colors, list) is False:
            colors = [colors] * len(elements)

        srcs = []
        isos = []

        for i, el in enumerate(elements):

            if tv_denoise:
                import skimage.filter
                img = self.get_result(el, result).deepcopy()
                img.data = skimage.filter.denoise_tv_chambolle(
                    img.data,
                    weight=0.5,
                    n_iter_max=3)
                #img = utils_eds.tv_denoise(img)
            else:
                img = self.get_result(el, result)
            figure, src, iso = img.plot_3D_iso_surface(
                threshold=thresholds[i], outline=outline, figure=figure,
                color=colors[i], **kwargs)
            outline = False
            srcs.append(src)
            isos.append(iso)

        if len(elements) == 1:
            return figure, src, iso
        else:
            return figure, srcs, isos

    def add_standards_to_signal(self, std_names, dtype=None):
        """
        Add to the data extra lines containing the selected standard.

        The added standard have Poisson noise

        Parameters
        ----------

        std_names: list of string or 'all'

        dtype: If dtype == None, get the highest dtype between spec and
            self.

        Example
        -------

        >>> s = database.spec3D('SEM')
        >>> from hyperspy.misc.config_dir import config_path
        >>> s.add_elements(['Hf','Ta'])
        >>> s.link_standard(config_path+'/database/SEM/std_RR')
        >>> s2 = s.add_standards_to_signal(['Hf'])

        To show that works

        >>> s2.change_dtype('float')
        >>> s2.decomposition(True)
        >>> s3 = s2.get_decomposition_model(5)
        >>> s3[102:134,125:152].get_lines_intensity(
        >>>     plot_result=True,lines_deconvolution='standard')

        """

        mp = self.metadata
        if mp.has_item('Sample') is False:
            if mp.Sample.has_item('standard_spec') is False:
                raise ValueError(
                    "The Sample.standard_spec needs to be set")

        if std_names == 'all':
            if mp.Sample.has_item('elements'):
                std_names = mp.Sample.elements
            else:
                raise ValueError(
                    "With std_names = 'all', " +
                    "the Sample.elements need to be set")

        dim_nav = list(self.axes_manager.navigation_shape)
        mean_counts = self.data.mean() * self.axes_manager.signal_shape[0]
        spec_result = self.deepcopy()

        for std in std_names:
            std_spec = self.get_result(std, 'standard_spec').deepcopy()
            fact = mean_counts / std_spec.data.sum()
            std_noise = std_spec * fact

            for dim in [1] + dim_nav[1:]:
                std_noise = utils.stack([std_noise] * dim)
                del std_noise.original_metadata.stack_elements
            std_noise.add_poissonian_noise()
            if dtype is not None:
                std_noise.change_dtype(dtype)
            spec_result = utils.stack([spec_result, std_noise], axis=0)

        del spec_result.original_metadata.stack_elements

        return spec_result

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
        spec = self._get_signal_signal()
        spec.metadata.General.title = 'Absorption model (PHD model)'
        if spec.axes_manager.signal_axes[0].units == 'eV':
            units_factor = 1000.
        else:
            units_factor = 1.
        beam_energy = self._get_beam_energy() / units_factor
        elements = self.metadata.Sample.elements
        TOA = self.get_take_off_angle()
        if weight_fraction == 'auto':
            if 'weight_fraction' in self.metadata.Sample:
                weight_fraction = self.metadata.Sample.weight_fraction
            else:
                weight_fraction = []
                for elm in elements:
                    weight_fraction.append(1. / len(elements))

        eng = spec.axes_manager.signal_axes[0].axis / units_factor
        eng = eng[np.searchsorted(eng, 0.0):]
        spec.data = np.append(np.array([0.0] * (len(spec.data) - len(eng))),
                              physical_model.xray_absorption_bulk(
                                  energy=eng,
                                  weight_fraction=weight_fraction,
                                  elements=elements,
                                  beam_energy=beam_energy,
                                  TOA=TOA))
        return spec

    # def check_total(self):
        #img_0 = self.get_result(xray_lines[0],'kratios')

        #data_total = np.zeros_like(img_0.data)
        # for xray_line in xray_lines:
            #data_total += self.get_result(xray_line,'kratios').data

        #img_total = img_0.deepcopy
        #img_total.data = data_total
        # return img_total


      # should use get_lines_intensity
    # def deconvolve_intensity(self, width_windows='all', plot_result=True):
        #"""
        # Calculate the intensity by fitting standard spectra to the spectrum.

        # Deconvoluted intensity is thus obtained compared to
        # get_intensity_map

        # Needs standard to be set

        # Parameters
        #----------

        # width_windows: 'all' | [min energy, max energy]
            # Set the energy windows in which the fit is done. If 'all'
            #(default option), the whole energy range is used.

        # plot_result : bool
            # If True (default option), plot the intensity maps.

        # See also
        #--------

        #set_elements, link_standard, get_intensity_map


        #"""
        # print 'This is obsolete, use get_lines_intensity instead'
        #from hyperspy.hspy import create_model
        #m = create_model(self)
        #mp = self.metadata

        #elements = mp.Sample.elements

        #fps = []
        # for element in elements:
            #std = self.get_result(element, 'standard_spec')
            #fp = components.ScalableFixedPattern(std)
            #fp.set_parameters_not_free(['offset', 'xscale', 'shift'])
            # fps.append(fp)
            # m.append(fps[-1])
        # if width_windows != 'all':
            #m.set_signal_range(width_windows[0], width_windows[1])
        # m.multifit(fitter='leastsq')
        #mp.Sample.intensities = list(np.zeros(len(elements)))
        #i = 0
        # for element in elements:
            # if (self.axes_manager.navigation_dimension == 0):
                # self._set_result(element, 'intensities',
                                 # fps[i].yscale.value, plot_result)
                # if plot_result and i == 0:
                    # m.plot()
                    #plt.title('Fitted standard')
            # else:
                # self._set_result(element, 'intensities',
                                 # fps[i].yscale.as_signal().data, plot_result)
            #i += 1



