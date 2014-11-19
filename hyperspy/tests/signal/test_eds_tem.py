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


import numpy as np
from nose.tools import assert_true, assert_equal

from hyperspy.signals import EDSTEMSpectrum, Simulation
from hyperspy.defaults_parser import preferences
from hyperspy.misc.eds import database
from hyperspy.components import Gaussian


class Test_metadata:

    def setUp(self):
        # Create an empty spectrum
        s = EDSTEMSpectrum(np.ones((4, 2, 1024)))
        s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.TEM.beam_energy = 15.0
        self.signal = s

    def test_sum_live_time(self):
        s = self.signal
        sSum = s.sum(0)
        assert_equal(
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 *
            2)

    def test_rebin_live_time(self):
        s = self.signal
        dim = s.axes_manager.shape
        s = s.rebin([dim[0] / 2, dim[1] / 2, dim[2]])
        assert_equal(
            s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 *
            2 *
            2)

    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al', 'Ni'])
        assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(['Al', 'Ni'])
        assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(["Fe", ])
        assert_equal(s.metadata.Sample.elements, ['Al', "Fe", 'Ni'])
        s.set_elements(['Al', 'Ni'])
        assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])

    def test_default_param(self):
        s = self.signal
        mp = s.metadata
        assert_equal(mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa,
                     preferences.EDS.eds_mn_ka)

    def test_SEM_to_TEM(self):
        s = self.signal[0, 0]
        signal_type = 'EDS_SEM'
        mp = s.metadata
        mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa = 125.3
        sSEM = s.deepcopy()
        sSEM.set_signal_type(signal_type)
        mpSEM = sSEM.metadata
        results = [
            mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa]
        results.append(signal_type)
        resultsSEM = [
            mpSEM.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa]
        resultsSEM.append(mpSEM.Signal.signal_type)
        assert_equal(results, resultsSEM)

    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSTEMSpectrum(np.ones((1024)))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        assert_equal(s.axes_manager.signal_axes[0].scale,
                     energy_axis.scale)


class Test_quantification:

    def setUp(self):
        s = EDSTEMSpectrum(np.ones([2, 1024]))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.set_microscope_parameters(beam_energy=200,
                                    live_time=3.1, tilt_stage=0.0,
                                    azimuth_angle=None, elevation_angle=35,
                                    energy_resolution_MnKa=130)
        elements = ['Al', 'Zn']
        xray_lines = ['Al_Ka', 'Zn_Ka']
        intensities = [300, 500]
        for i, xray_line in enumerate(xray_lines):
            gauss = Gaussian()
            line_energy, FWHM = s._get_line_energy(xray_line, FWHM_MnKa='auto')
            gauss.centre.value = line_energy
            gauss.A.value = intensities[i]
            gauss.sigma.value = FWHM
            s.data[:] += gauss.function(energy_axis.axis)

        s.set_elements(elements)
        s.add_lines(xray_lines)
        self.signal = s

    def test_quant_lorimer_simple(self):
        s = self.signal
        kfactors = [2.0009344042484134]
        intensities = s.get_lines_intensity()
        res = s.quantification_cliff_lorimer(intensities, kfactors)
        assert_true(np.allclose(res[0].data, np.array(
                    [0.2270779, 0.2270779]), atol=1e-3))


class Test_vacum_mask:

    def setUp(self):
        s = Simulation(np.array([np.linspace(0.001,0.5,20)]*100).T)
        s.add_poissonian_noise()
        s = EDSTEMSpectrum(s.data)
        self.signal = s

    def test_vacuum_mask(self):
        s = self.signal
        assert_equal(s.vacuum_mask().data[0], True)
        assert_equal(s.vacuum_mask().data[-1], False)

# class Test_get_lines_intentisity:
#    def setUp(self):
# Create an empty spectrum
#        s = EDSTEMSpectrum(np.ones((4,2,1024)))
#        energy_axis = s.axes_manager.signal_axes[0]
#        energy_axis.scale = 0.01
#        energy_axis.offset = -0.10
#        energy_axis.units = 'keV'
#        self.signal = s
#
#    def test(self):
#        s = self.signal
#        s.set_elements(['Al','Ni'],['Ka','La'])
#        sAl = s.get_lines_intensity(plot_result=True)[0]
#        assert_true(np.allclose(s[...,0].data*15.0, sAl.data))
