import numpy as np
import nose.tools

from hyperspy.hspy import *


def test_spectrum_binned_default():
    s = signals.Spectrum([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_image_binned_default():
    s = signals.Image(np.empty((2, 2)))
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_image_simulation_binned_default():
    s = signals.ImageSimulation(np.empty([2, 2]))
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_dielectric_function_binned_default():
    s = signals.DielectricFunction([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_signal_binned_default():
    s = signals.Signal([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_simulation_binned_default():
    s = signals.Simulation([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_spectrum_simulation_binned_default():
    s = signals.SpectrumSimulation([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_eels_spectrum_binned_default():
    s = signals.EELSSpectrum([0])
    nose.tools.assert_true(s.metadata.Signal.binned)


def test_eds_tem_binned_default():
    s = signals.EDSTEMSpectrum([0])
    nose.tools.assert_true(s.metadata.Signal.binned)


def test_eds_sem_binned_default():
    s = signals.EDSSEMSpectrum([0])
    nose.tools.assert_true(s.metadata.Signal.binned)


class TestModelBinned:

    def setUp(self):
        s = signals.Spectrum([1])
        s.axes_manager[0].scale = 0.1
        m = create_model(s)
        m.append(components.Offset())
        m[0].offset.value = 1
        self.m = m

    def test_unbinned(self):
        self.m.spectrum.metadata.Signal.binned = False
        nose.tools.assert_equal(self.m(), 1)

    def test_binned(self):
        self.m.spectrum.metadata.Signal.binned = True
        nose.tools.assert_equal(self.m(), 0.1)
