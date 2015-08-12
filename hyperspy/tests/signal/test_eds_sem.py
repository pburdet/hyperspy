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


import numpy as np
from nose.tools import assert_true, assert_equal, assert_not_equal
from matplotlib.testing.decorators import cleanup
# from matplotlib.testing.decorators import image_comparison
import nose.tools


from hyperspy.signals import EDSSEMSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.components import Gaussian
from hyperspy.misc.elements import elements_db as elements_EDS
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds import database
from hyperspy import utils
from hyperspy.misc.config_dir import config_path


class Test_metadata:

    def setUp(self):
        # Create an empty spectrum
        s = EDSSEMSpectrum(np.ones((4, 2, 1024)))
        s.axes_manager.signal_axes[0].scale = 1e-3
        s.axes_manager.signal_axes[0].units = "keV"
        s.axes_manager.signal_axes[0].name = "Energy"
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 15.0
        s.metadata.Acquisition_instrument.SEM.tilt_stage = -38
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle = 63
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.elevation_angle = 35
        self.signal = s

    def test_sum_live_time(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum(0)
        nose.tools.assert_equal(
            sSum.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time,
            3.1 *
            2)
        # Check that metadata is unchanged
        print old_metadata, s.metadata      # Capture for comparison on error
        nose.tools.assert_dict_equal(old_metadata.as_dictionary(),
                                     s.metadata.as_dictionary(),
                                     "Source metadata changed")

    def test_rebin_live_time(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        dim = s.axes_manager.shape
        s = s.rebin([dim[0] / 2, dim[1] / 2, dim[2]])
        nose.tools.assert_equal(
            s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time,
            3.1 *
            2 *
            2)
        # Check that metadata is unchanged
        print old_metadata, self.signal.metadata    # Captured on error
        nose.tools.assert_dict_equal(old_metadata.as_dictionary(),
                                     self.signal.metadata.as_dictionary(),
                                     "Source metadata changed")

    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al', 'Ni'])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(['Al', 'Ni'])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(["Fe", ])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', "Fe", 'Ni'])
        s.set_elements(['Al', 'Ni'])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])

    def test_add_lines(self):
        s = self.signal
        s.add_lines(lines=())
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, [])
        s.add_lines(("Fe_Ln",))
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, ["Fe_Ln"])
        s.add_lines(("Fe_Ln",))
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, ["Fe_Ln"])
        s.add_elements(["Ti", ])
        s.add_lines(())
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Fe_Ln', 'Ti_La'])
        s.set_lines((), only_one=False, only_lines=False)
        nose.tools.assert_equal(s.metadata.Sample.xray_lines,
                                ['Fe_La', 'Fe_Lb3', 'Fe_Ll', 'Fe_Ln', 'Ti_La',
                                 'Ti_Lb3', 'Ti_Ll', 'Ti_Ln'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 0.4
        s.set_lines((), only_one=False, only_lines=False)
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, ['Ti_Ll'])

    def test_add_lines_auto(self):
        s = self.signal
        s.axes_manager.signal_axes[0].scale = 1e-2
        s.set_elements(["Ti", "Al"])
        s.set_lines(['Al_Ka'])
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Al_Ka', 'Ti_Ka'])

        del s.metadata.Sample.xray_lines
        s.set_elements(['Al', 'Ni'])
        s.add_lines()
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Al_Ka', 'Ni_Ka'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 10.0
        s.set_lines([])
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Al_Ka', 'Ni_La'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 200
        s.set_elements(['Au', 'Ni'])
        s.set_lines([])
        nose.tools.assert_equal(s.metadata.Sample.xray_lines,
                                ['Au_La', 'Ni_Ka'])

    def test_default_param(self):
        s = self.signal
        mp = s.metadata
        nose.tools.assert_equal(
            mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa,
            preferences.EDS.eds_mn_ka)

    def test_SEM_to_TEM(self):
        s = self.signal[0, 0]
        signal_type = 'EDS_TEM'
        mp = s.metadata
        mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa = \
            125.3
        sTEM = s.deepcopy()
        sTEM.set_signal_type(signal_type)
        mpTEM = sTEM.metadata
        results = [
            mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa,
            signal_type]
        resultsTEM = [
            mpTEM.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa,
            mpTEM.Signal.signal_type]
        nose.tools.assert_equal(results, resultsTEM)

    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSSEMSpectrum(np.ones(1024))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        nose.tools.assert_equal(s.axes_manager.signal_axes[0].scale,
                                energy_axis.scale)

    def test_take_off_angle(self):
        s = self.signal
        nose.tools.assert_equal(s.get_take_off_angle(), 12.886929785732487)


class Test_get_lines_intentisity:

    def setUp(self):
        # Create an empty spectrum
        s = EDSSEMSpectrum(np.zeros((2, 2, 3, 100)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.04
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        g = Gaussian()
        g.sigma.value = 0.05
        g.centre.value = 1.487
        s.data[:] = g.function(energy_axis.axis)
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 15.0
        self.signal = s

    def test_eV(self):
        s = self.signal
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 40
        energy_axis.units = 'eV'

        sAl = s.get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_windows=5)[0]
        nose.tools.assert_true(np.allclose(24.99516, sAl.data[0, 0, 0],
                                           atol=1e-3))

    def test(self):
        s = self.signal
        sAl = s.get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_windows=5)[0]
        nose.tools.assert_true(
            np.allclose(24.99516, sAl.data[0, 0, 0], atol=1e-3))
        sAl = s[0].get_lines_intensity(["Al_Ka"],
                                       plot_result=False,
                                       integration_windows=5)[0]
        nose.tools.assert_true(
            np.allclose(24.99516, sAl.data[0, 0], atol=1e-3))
        sAl = s[0, 0].get_lines_intensity(["Al_Ka"],
                                          plot_result=False,
                                          integration_windows=5)[0]
        nose.tools.assert_true(np.allclose(24.99516, sAl.data[0], atol=1e-3))
        sAl = s[0, 0, 0].get_lines_intensity(["Al_Ka"],
                                             plot_result=False,
                                             integration_windows=5)[0]
        nose.tools.assert_true(np.allclose(24.99516, sAl.data, atol=1e-3))
        s.axes_manager[-1].offset = 1.0
        sC = s.get_lines_intensity(["C_Ka"], plot_result=False)
        nose.tools.assert_equal(len(sC), 0)
        nose.tools.assert_true(sAl.metadata.Sample.elements, ["Al"])
        nose.tools.assert_true(sAl.metadata.Sample.xray_lines, ["Al_Ka"])

#    def test_model_deconvolution(self):
#        s = self.signal
#
#        sAl = s[0].get_lines_intensity(["Al_Ka"],
#                                       plot_result=False,
#                                       lines_deconvolution='model')[0]
#        nose.tools.assert_true(np.allclose(19.22613, sAl.data[0, 0],
#                                           atol=1e-3))
#        # Why?
#        # assert_true(np.allclose(0.75061671, sAl.data[0, 0], atol=1e-3))
#        sAl = s[0, 0].get_lines_intensity(["Al_Ka"],
#                                          plot_result=False,
#                                          lines_deconvolution='model')[0]
#        nose.tools.assert_true(np.allclose(19.22613, sAl.data[0], atol=1e-3))
#        # assert_true(np.allclose(0.75061671, sAl.data[0], atol=1e-3))
#        sAl = s[0, 0, 0].get_lines_intensity(["Al_Ka"],
#                                             plot_result=False,
#                                             lines_deconvolution='model')[0]
#        nose.tools.assert_true(np.allclose(19.22613, sAl.data, atol=1e-3))
#        # assert_true(np.allclose(0.75061671, sAl.data, atol=1e-3))

#    def test_model_deconvolution_std(self):
#        s = self.signal
#        std = s[0, 0, 0]
#        std.metadata.General.title = 'Al_std'
#        s.metadata.set_item('Sample.standard_spec', [std * 0.99])
#        sAl = s[0].get_lines_intensity(["Al_Ka"],
#                                       plot_result=False,
#                                       lines_deconvolution='standard')[0]
#        nose.tools.assert_true(np.allclose(25.252525252525249, sAl.data[0, 0],
#                                           atol=1e-3))
#        sAl = s[0, 0].get_lines_intensity(["Al_Ka"],
#                                          plot_result=False,
#                                          lines_deconvolution='standard')[0]
#        nose.tools.assert_true(np.allclose(25.252525252525249, sAl.data[0],
#                                           atol=1e-3))
#        sAl = s[0, 0, 0].get_lines_intensity(["Al_Ka"],
#                                             plot_result=False,
#                                             lines_deconvolution='standard')[0]
#        nose.tools.assert_true(np.allclose(25.252525252525249, sAl.data,
#                                           atol=1e-3))
                                           
    def test_background_substraction(self):
        s = self.signal
        intens = s.get_lines_intensity(["Al_Ka"], plot_result=False)[0].data
        s += 1.
        nose.tools.assert_true(np.allclose(s.estimate_background_windows(
            xray_lines=["Al_Ka"])[0, 0], 1.25666201, atol=1e-3))
        nose.tools.assert_true(np.allclose(s.get_lines_intensity(
            ["Al_Ka"], background_windows=s.estimate_background_windows(
                [4, 4], xray_lines=["Al_Ka"]), plot_result=False)[0].data,
            intens, atol=1e-3))

    def test_estimate_integration_windows(self):
        s = self.signal
        nose.tools.assert_true(np.allclose(
            s.estimate_integration_windows(3.0, ["Al_Ka"]),
            [[1.371, 1.601]], atol=1e-2))

    def test_with_signals_examples(self):
        from hyperspy.misc.example_signals_loading import \
            load_1D_EDS_SEM_spectrum as EDS_SEM_Spectrum
        s = EDS_SEM_Spectrum()
        np.allclose(utils.stack(s.get_lines_intensity()).data,
                    np.array([84163, 89063, 96117, 96700, 99075]))


class Test_quantification:

    def setUp(self):
        s = utils.eds.xray_lines_model(
            elements=['Al', 'Zn'], beam_energy=15, counts_rate=1000,
            weight_percents=[70, 30],
            energy_resolution_MnKa=130, live_time=3.1,
            energy_axis={'units': 'keV', 'size': 1024,
                         'scale': 1e-2, 'name': 'Energy', 'offset': -0.1})
        s.set_signal_type('EDS_SEM')
        s = utils.stack([utils.stack([utils.stack([s]*3)]*2)]*2)
        s.add_lines()
        stdAl = utils.eds.xray_lines_model(
            elements=['Al'], beam_energy=15, counts_rate=1000, live_time=31.,
            weight_percents=[100], energy_resolution_MnKa=130,
            energy_axis={'units': 'keV', 'size': 1024,
                         'scale': 1e-2, 'name': 'Energy', 'offset': -0.1})
        stdAl.metadata.General.title = 'Al_std'
        stdAl.set_signal_type('EDS_SEM')
        stdZn = utils.eds.xray_lines_model(
            elements=['Zn'], beam_energy=15, counts_rate=1000, live_time=31.,
            weight_percents=[100], energy_resolution_MnKa=130,
            energy_axis={'units': 'keV', 'size': 1024,
                         'scale': 1e-2, 'name': 'Energy', 'offset': -0.1})
        stdZn.set_signal_type('EDS_SEM')
        stdZn.metadata.General.title = 'Zn_std'
        s.metadata.Sample.standard_spec = [stdAl, stdZn]
        self.signal = s

    @cleanup
    def test_kratio(self):
        s = self.signal

        s1 = s.deepcopy()[0, 0, 0]
        s1.get_kratio(plot_result=False)
        res = np.array([s1.get_result('Al_Ka', 'kratios').data,
                        s1.get_result('Zn_La', 'kratios').data])
        nose.tools.assert_true(np.allclose(res, np.array([70.,  30.])))

        # s1.check_kratio(('Al_Ka', 'Zn_La'))

        s1 = s.deepcopy()[0, 0]
        s1.get_kratio(plot_result=False)
        res = np.array([s1.get_result('Al_Ka', 'kratios').data[0],
                        s1.get_result('Zn_La', 'kratios').data[0]])
        nose.tools.assert_true(np.allclose(res, np.array([70.,  30.])))

        s1 = s.deepcopy()[0]
        s1.get_kratio(plot_result=False)
        res = np.array([s1.get_result('Al_Ka', 'kratios').data[0, 0],
                        s1.get_result('Zn_La', 'kratios').data[0, 0]])
        nose.tools.assert_true(np.allclose(res, np.array([70.,  30.])))

        s.get_kratio(plot_result=False)
        res = np.array([s.get_result('Al_Ka', 'kratios').data[0, 0, 0],
                        s.get_result('Zn_La', 'kratios').data[0, 0, 0]])
        nose.tools.assert_true(np.allclose(res, np.array([70.,  30.])))

        s.get_kratio([[["Zn_La", 'Al_Ka'], ["Zn", 'Al'], [0.8, 1.75]]],
                     plot_result=False)
        res = np.array([s.get_result('Al_Ka', 'kratios').data[0, 0, 0],
                        s.get_result('Zn_La', 'kratios').data[0, 0, 0]])
        nose.tools.assert_true(np.allclose(res, np.array([70.,  30.])))

    def test_quant(self):
        s = self.signal

        s1 = s.deepcopy()[0, 0, 0]
        s1.get_kratio(plot_result=False)
        s1.quant(plot_result=False)
        res = np.array([s1.get_result('Al', 'quant').data,
                        s1.get_result('Zn', 'quant').data])
        nose.tools.assert_true(np.allclose(res, np.array([1.,  1.])))

        s1 = s.deepcopy()[0, 0]
        s1.get_kratio(plot_result=False)
        s1.quant(plot_result=False)
        res = np.array([s1.get_result('Al', 'quant').data[0],
                        s1.get_result('Zn', 'quant').data[0]])
        nose.tools.assert_true(np.allclose(res, np.array([1.,  1.])))

        s1 = s.deepcopy()[0]
        s1.get_kratio(plot_result=False)
        s1.quant(plot_result=False)
        res = np.array([s1.get_result('Al', 'quant').data[0, 0],
                        s1.get_result('Zn', 'quant').data[0, 0]])
        nose.tools.assert_true(np.allclose(res, np.array([1.,  1.])))

        s.get_kratio(plot_result=False)
        s.quant(plot_result=False)
        res = np.array([s.get_result('Al', 'quant').data[0, 0, 0],
                        s.get_result('Zn', 'quant').data[0, 0, 0]])
        nose.tools.assert_true(np.allclose(res, np.array([1.,  1.])))

# Should go in is own file


class Test_simulation:

    def setUp(self):
        # careful with auto param
        s = EDSSEMSpectrum(np.ones(1024))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 15.0

        s.set_elements(('Al', 'Zn'))
        s.add_lines()

        self.signal = s

    def test_simu_1_spec(self):
        s = self.signal
        gateway = utils_eds.get_link_to_jython()
        s.simulate_model()
        utils_eds.simulate_one_spectrum(nTraj=10,
                                        mp=s.metadata, gateway=gateway)
        utils_eds.simulate_Xray_depth_distribution(10,
                                                   mp=s.metadata, gateway=gateway)
        # test to save time, test TEM here
        s.set_signal_type('EDS_TEM')
        utils_eds.simulate_one_spectrum_TEM(nTraj=10,
                                            mp=s.metadata, gateway=gateway)
        s.get_kfactors_from_first_principles(gateway=gateway)
        nose.tools.assert_equal(s.metadata.Sample.kfactors[0], 0.9608483657921267)
                                # 1.9798201626244532)
        s.simulate_two_elements_standard(nTraj=10, gateway=gateway)
        s.get_kfactors_from_standard()
        #s.quantification()
        s.set_signal_type('EDS_SEM')


class Test_electron_distribution:

    def setUp(self):
        # careful with auto param
        s = EDSSEMSpectrum(np.ones((2, 2, 3, 1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0

        nav_axis = s.axes_manager.navigation_axes
        units_name = '${\mu}m$'
        EDS_scale = np.array([0.050, 0.050, 0.100])
        for i, ax in enumerate(nav_axis):
            ax.units = units_name
            ax.scale = EDS_scale[i]

        s.set_elements(('Al', 'Zn'))
        s.add_lines()

        self.signal = s

    def test_electron_distribution(self):
        s = self.signal
        s.simulate_electron_distribution(nb_traj=10,
                                         limit_x=[-0.250, 0.300], dx0=0.004, dx_increment=0.75)


class Test_convolve_sum:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones((2, 2, 3, 1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 15.0

        self.signal = s

    # def test_running_sum(self):
        #s = self.signal
        # s.running_sum()

        #assert_equal(s[0, 0, 0, 0].data[0], 4.)

        #s = self.signal
        #s = s[0]

        # s.running_sum()
        #assert_equal(s[0, 0, 0].data[0], 16.)

        #assert_equal(s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time, 49.6)

    def test_convolve_sum(self):
        s = self.signal
        res = s.convolve_sum()

        nose.tools.assert_equal(res[0, 0, 0, 0].data[0], 9.)

        s = self.signal
        s = s[0]

        res = s.convolve_sum(size=4)
        assert_equal(res[0, 0, 0].data[0], 16.)

        nose.tools.assert_equal(
            res.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time,
            49.6)


class Test_plot_Xray_lines:

    def setUp(self):
        s = utils_eds.simulate_model()
        self.signal = s

    @cleanup
    def test_plot_Xray_lines(self):
        s = self.signal

        s.plot(True)
        # s.plot_Xray_lines(only_lines=('a'))
        # s.plot_Xray_lines(only_lines=('a,Kb'))

    # @image_comparison(baseline_images=['plot_Xray_lines2'])
    # def test_plot_Xray_lines2(self):
        #s = self.signal
        # s.plot_Xray_lines()

class Test_tools_bulk:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.units = 'keV'
        s.set_elements(['Al', 'Zn'])
        s.add_lines()
        self.signal = s

    def test_electron_range(self):
        s = self.signal
        mp = s.metadata
        elec_range = utils.eds.electron_range(
            mp.Sample.elements[0],
            mp.Acquisition_instrument.SEM.beam_energy,
            density='auto',
            tilt=mp.Acquisition_instrument.SEM.tilt_stage)
        nose.tools.assert_equal(elec_range, 0.41350651162374225)

    def test_xray_range(self):
        s = self.signal
        mp = s.metadata
        xr_range = utils.eds.xray_range(
            mp.Sample.xray_lines[0],
            mp.Acquisition_instrument.SEM.beam_energy,
            density=4.37499648818)
        nose.tools.assert_true(np.allclose(xr_range, 0.1900207, atol=1e-3))
        # assert_equal(xr_range, 0.1900368800933955)
        # assert_equal(xr_range, 0.1900207, atol=1e-3)


class Test_decomposition_model_from:

    def setUp(self):
        s = database.spec3D()
        s.change_dtype('float')
        s = s[:4, :6, :10]
        self.signal = s

    def test_decomposition_model_from_2D(self):
        s = self.signal
        s2 = s.deepcopy()
        dim = s.axes_manager.shape
        s2 = s2.rebin((dim[0] / 2, dim[1] / 2, dim[2]))
        s2.decomposition(True)
        a = s.get_decomposition_model_from(s2, components=3)
        nose.tools.assert_true(a.axes_manager.shape == s.axes_manager.shape)

    def test_decomposition_model_from_3D(self):
        s = self.signal
        s = utils.stack([s, s])
        s2 = s.deepcopy()
        dim = s.axes_manager.shape
        s2 = s2.rebin((dim[0] / 2, dim[1] / 2, dim[2], dim[3]))
        s2.decomposition(True)
        a = s.get_decomposition_model_from(s2, components=3)
        nose.tools.assert_true(a.axes_manager.shape == s.axes_manager.shape)


class Test_add_standards_to_signal:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones([3, 4, 5, 1024]))
        self.signal = s

    def test_add_standards_to_signal_3D(self):
        s = self.signal
        dim = s.axes_manager.shape
        elements = ['Hf', 'Ta']
        s.add_elements(elements)
        s.link_standard(config_path + '/database/SEM/std_RR')
        res = s.add_standards_to_signal('all')
        dim = np.array(s.axes_manager.navigation_shape)
        dim_res = np.array(res.axes_manager.navigation_shape)
        nose.tools.assert_true(np.all(dim_res == dim + [len(elements), 0, 0]))

    def test_add_standards_to_signal_2D(self):
        s = self.signal[0]
        dim = s.axes_manager.shape
        elements = ['Hf', 'Ta']
        s.add_elements(elements)
        s.link_standard(config_path + '/database/SEM/std_RR')
        res = s.add_standards_to_signal('all')
        dim = np.array(s.axes_manager.navigation_shape)
        dim_res = np.array(res.axes_manager.navigation_shape)
        nose.tools.assert_true(np.all(dim_res == dim + [len(elements), 0]))


class Test_energy_units:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0
        s.axes_manager.signal_axes[0].units = 'keV'
        s.set_microscope_parameters(energy_resolution_MnKa=130)
        self.signal = s

    def test_beam_energy(self):
        s = self.signal
        nose.tools.assert_equal(s._get_beam_energy(), 5.0)
        s.axes_manager.signal_axes[0].units = 'eV'
        nose.tools.assert_equal(s._get_beam_energy(), 5000.0)
        s.axes_manager.signal_axes[0].units = 'keV'

    def test_line_energy(self):
        s = self.signal
        nose.tools.assert_equal(s._get_line_energy('Al_Ka'), 1.4865)
        s.axes_manager.signal_axes[0].units = 'eV'
        nose.tools.assert_equal(s._get_line_energy('Al_Ka'), 1486.5)
        s.axes_manager.signal_axes[0].units = 'keV'

        nose.tools.assert_equal(s._get_line_energy('Al_Ka', FWHM_MnKa='auto'),
                                (1.4865, 0.07661266213883969))
        nose.tools.assert_equal(s._get_line_energy('Al_Ka', FWHM_MnKa=128),
                                (1.4865, 0.073167615787314))


class Test_MAC:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.units = 'keV'
        s.set_elements(['Al', 'Zn'])
        s.add_lines()
        self.signal = s

    def test_MAC_sample(self):
        s = self.signal
        # assert_true(np.allclose(xr_range, 0.1900207, atol=1e-3))
        assert_true(np.allclose(
            s.get_sample_mass_absorption_coefficient(
                weight_fraction=[0.5, 0.5]),
                [2587.4161643905127, 1239.4598146508552]))


class Test_detector_efficiency:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones((5)))
        s.axes_manager.signal_axes[0].scale = 0.5
        s.axes_manager.signal_axes[0].units = "keV"
        self.signal = s

    def test_beam_energy(self):
        s = self.signal
        det = s.detetector_efficiency_from_layers(
            elements=['C', 'Al', 'Si', 'O'], thicknesses_layer=[50., 30.,
                                                                40., 40.],
            thickness_detector=0.45, cutoff_energy=0.1)
        nose.tools.assert_true(
            np.allclose(det.data, np.array([0., 0.75500789, 0.95501759,
                                            0.98541387, 0.95350419]),
                        atol=1e-3))
