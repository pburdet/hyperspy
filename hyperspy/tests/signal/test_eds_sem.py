# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from nose.tools import assert_true, assert_equal, assert_not_equal

from hyperspy.signals import EDSSEMSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.components import Gaussian
from hyperspy.misc.eds.elements import elements as elements_EDS
from hyperspy.misc.eds import utils as utils_eds
from hyperspy import utils


class Test_mapped_parameters:
    def setUp(self):
        # Create an empty spectrum
        s = EDSSEMSpectrum(np.ones((4,2,1024)))
        s.axes_manager.signal_axes[0].scale = 1e-3
        s.axes_manager.signal_axes[0].units = "keV"
        s.axes_manager.signal_axes[0].name = "Energy"
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 15.0 
        s.mapped_parameters.SEM.tilt_stage = -38
        s.mapped_parameters.SEM.EDS.azimuth_angle = 63
        s.mapped_parameters.SEM.EDS.elevation_angle = 35      
        self.signal = s
        
    def test_sum_live_time(self):
        s = self.signal
        sSum = s.sum(0)
        assert_equal(sSum.mapped_parameters.SEM.EDS.live_time, 3.1*2)
    
    def test_rebin_live_time(self):
        s = self.signal
        dim = s.axes_manager.shape
        s = s.rebin([dim[0]/2,dim[1]/2,dim[2]])
        assert_equal(s.mapped_parameters.SEM.EDS.live_time, 3.1*2*2)
 
    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al','Ni'])
        assert_equal(s.mapped_parameters.Sample.elements, ['Al','Ni'])
        s.add_elements(['Al','Ni'])
        assert_equal(s.mapped_parameters.Sample.elements, ['Al','Ni'])
        s.add_elements(["Fe",])
        assert_equal(s.mapped_parameters.Sample.elements, ['Al',"Fe", 'Ni'])
        s.set_elements(['Al','Ni'])
        assert_equal(s.mapped_parameters.Sample.elements, ['Al','Ni'])
    
    def test_add_lines(self):
        s = self.signal
        s.add_lines(lines=())
        assert_equal(s.mapped_parameters.Sample.Xray_lines, [])
        s.add_lines(("Fe_Ln",))
        assert_equal(s.mapped_parameters.Sample.Xray_lines, ["Fe_Ln"])
        s.add_lines(("Fe_Ln",))
        assert_equal(s.mapped_parameters.Sample.Xray_lines, ["Fe_Ln"])
        s.add_elements(["Ti",])
        s.add_lines(())
        assert_equal(s.mapped_parameters.Sample.Xray_lines, ['Fe_Ln', 'Ti_La'])
        s.set_lines((), only_one=False, only_lines=False)
        assert_equal(s.mapped_parameters.Sample.Xray_lines,
                     ['Fe_La', 'Fe_Lb3', 'Fe_Ll', 'Fe_Ln', 'Ti_La', 
                     'Ti_Lb3', 'Ti_Ll', 'Ti_Ln'])
        s.mapped_parameters.SEM.beam_energy = 0.4
        s.set_lines((), only_one=False, only_lines=False)
        assert_equal(s.mapped_parameters.Sample.Xray_lines, ['Ti_Ll'])
#        s.add_lines()
#        results.append(mp.Sample.Xray_lines[1])
#        mp.SEM.beam_energy = 10.0
#        s.set_elements(['Al','Ni'])
#        results.append(mp.Sample.Xray_lines[1])
#        s.add_elements(['Fe'])
#        results.append(mp.Sample.Xray_lines[1])    
#        assert_equal(results, ['Al_Ka','Ni','Ni_Ka','Ni_La','Fe_La'])
        
    def test_default_param(self):
        s = self.signal
        mp = s.mapped_parameters
        assert_equal(mp.SEM.EDS.energy_resolution_MnKa,
            preferences.EDS.eds_mn_ka)
            
    def test_SEM_to_TEM(self):
        s = self.signal[0,0]
        signal_type = 'EDS_TEM'
        mp = s.mapped_parameters
        mp.SEM.EDS.energy_resolution_MnKa = 125.3
        sTEM = s.deepcopy()
        sTEM.set_signal_type(signal_type)        
        mpTEM = sTEM.mapped_parameters            
        results = [mp.SEM.EDS.energy_resolution_MnKa]
        results.append(signal_type)        
        resultsTEM = [mpTEM.TEM.EDS.energy_resolution_MnKa]
        resultsTEM.append(mpTEM.signal_type)        
        assert_equal(results,resultsTEM )
        
    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSSEMSpectrum(np.ones((1024)))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        assert_equal(s.axes_manager.signal_axes[0].scale,
            energy_axis.scale)
            
    def test_take_off_angle(self):
        s = self.signal
        assert_equal(s.get_take_off_angle(),12.886929785732487)
        
        
class Test_get_intentisity_map:
    def setUp(self):
        # Create an empty spectrum
        s = EDSSEMSpectrum(np.zeros((2,2,3,100)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.04
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        g = Gaussian()
        g.sigma.value = 0.05
        g.centre.value = 1.487
        s.data[:] = g.function(energy_axis.axis)
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 15.0               
        self.signal = s
    
    def test(self):        
        s = self.signal
        sAl = s.get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_window_factor=5)[0]
        assert_true(np.allclose(1, sAl.data[0,0,0], atol=1e-3))
        sAl = s[0].get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_window_factor=5)[0]
        assert_true(np.allclose(1, sAl.data[0,0], atol=1e-3))
        sAl = s[0,0].get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_window_factor=5)[0]
        assert_true(np.allclose(1, sAl.data[0], atol=1e-3))
        sAl = s[0,0,0].get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_window_factor=5)[0]
        assert_true(np.allclose(1, sAl.data, atol=1e-3))
        
        
class Test_quantification:
    def setUp(self):
        s = EDSSEMSpectrum(np.ones((2,2,3,1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 15.0 
        
        gauss = Gaussian()
        line_energy = elements_EDS['Al']['Xray_energy']['Ka']               
        gauss.centre.value = line_energy
        gauss.A.value = 500
        FWHM_MnKa = s.mapped_parameters.SEM.EDS.energy_resolution_MnKa        
        gauss.sigma.value = utils_eds.get_FWHM_at_Energy(FWHM_MnKa,line_energy)

        gauss2 = Gaussian()
        line_energy = elements_EDS['Zn']['Xray_energy']['La']
        gauss2.centre.value = line_energy
        gauss2.A.value = 300
        FWHM_MnKa = s.mapped_parameters.SEM.EDS.energy_resolution_MnKa        
        gauss2.sigma.value = utils_eds.get_FWHM_at_Energy(FWHM_MnKa,line_energy)

        s.data[:] = (gauss.function(energy_axis.axis) + 
                     gauss2.function(energy_axis.axis))  
                     
        s.set_elements(('Al','Zn'))
        s.add_lines()

        stdAl=s[0,0,0].deepcopy()
        gauss.A.value = 12000
        stdAl.mapped_parameters.SEM.EDS.live_time = 31
        stdAl.data[:] = gauss.function(energy_axis.axis)
        stdAl.mapped_parameters.title = 'Al_std'

        stdZn=s[0,0,0].deepcopy()
        gauss2.A.value = 13000
        stdZn.mapped_parameters.SEM.EDS.live_time = 32
        stdZn.data[:] = gauss2.function(energy_axis.axis)
        stdZn.mapped_parameters.title = 'Zn_std'

        s.mapped_parameters.Sample.standard_spec = [stdAl,stdZn]
        self.signal = s
    
    def test_kratio(self):
        s = self.signal
        
        s1 = s.deepcopy()[0,0,0]
        s1.get_kratio(plot_result=False)
        res = np.array([s1.get_result('Al_Ka','kratios').data,
            s1.get_result('Zn_La','kratios').data])        
        assert_true(np.allclose(res,
            np.array([0.4166665022647609, 0.23821329009859846]))) 
            
        s1.check_kratio(('Al_Ka','Zn_La'))
            
        s1 = s.deepcopy()[0,0]
        s1.get_kratio(plot_result=False)
        res = np.array([s1.get_result('Al_Ka','kratios').data[0],
            s1.get_result('Zn_La','kratios').data[0]])        
        assert_true(np.allclose(res,
            np.array([0.4166665022647609, 0.23821329009859846]))) 
            
        s1 = s.deepcopy()[0]
        s1.get_kratio(plot_result=False)
        res = np.array([s1.get_result('Al_Ka','kratios').data[0,0],
            s1.get_result('Zn_La','kratios').data[0,0]])        
        assert_true(np.allclose(res,
            np.array([0.4166665022647609, 0.23821329009859846])))
        
        s.get_kratio(plot_result=False)
        res = np.array([s.get_result('Al_Ka','kratios').data[0,0,0],
            s.get_result('Zn_La','kratios').data[0,0,0]])        
        assert_true(np.allclose(res,
            np.array([0.4166665022647609, 0.23821329009859846])))
            
        s.get_kratio([[["Zn_La",'Al_Ka'],["Zn",'Al'],[0.8,1.75]]],
            plot_result=False)
        res = np.array([s.get_result('Al_Ka','kratios').data[0,0,0],
                    s.get_result('Zn_La','kratios').data[0,0,0]])
        np.allclose(res,
            np.array([ 0.41666667,  0.2382134 ]))
            
    def test_quant(self):
        s = self.signal
        
        s1 = s.deepcopy()[0,0,0]
        s1.get_kratio(plot_result=False)
        s1.quant(plot_result=False)
        res = np.array([s1.get_result('Al','quant').data,
            s1.get_result('Zn','quant').data])        
        assert_true(np.allclose(res,
            np.array([ 0.610979,  0.246892])))
            
        s1 = s.deepcopy()[0,0]
        s1.get_kratio(plot_result=False)
        s1.quant(plot_result=False)
        res = np.array([s1.get_result('Al','quant').data[0],
            s1.get_result('Zn','quant').data[0]])        
        assert_true(np.allclose(res,
            np.array([ 0.610979,  0.246892])))
            
        s1 = s.deepcopy()[0]
        s1.get_kratio(plot_result=False)
        s1.quant(plot_result=False)
        res = np.array([s1.get_result('Al','quant').data[0,0],
            s1.get_result('Zn','quant').data[0,0]])        
        assert_true(np.allclose(res,
            np.array([ 0.610979,  0.246892])))
                    
        s.get_kratio(plot_result=False)
        s.quant(plot_result=False)
        res = np.array([s.get_result('Al','quant').data[0,0,0],
            s.get_result('Zn','quant').data[0,0,0]])        
        assert_true(np.allclose(res,
            np.array([ 0.610979,  0.246892])))
            
#Should go in is own file            
class Test_simulation:    
    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 15.0 
 
        s.set_elements(('Al','Zn'))
        s.add_lines()


        self.signal = s
        
    def test_simu_1_spec(self):
        s = self.signal
        gateway= utils_eds.get_link_to_jython()
        utils_eds.simulate_one_spectrum(nTraj=10,
            mp=s.mapped_parameters,gateway=gateway)
        utils_eds.simulate_Xray_depth_distribution(10,
            mp=s.mapped_parameters,gateway=gateway)
        
class Test_electron_distribution:    
    def setUp(self):
        s = EDSSEMSpectrum(np.ones((2,2,3,1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 5.0 
        
        nav_axis = s.axes_manager.navigation_axes
        units_name = '${\mu}m$'
        EDS_scale = np.array([0.050,0.050,0.100])
        for i, ax in enumerate(nav_axis):
            ax.units = units_name
            ax.scale = EDS_scale[i]
 
        s.set_elements(('Al','Zn'))
        s.add_lines()


        self.signal = s
        
    def test_electron_distribution(self):
        s = self.signal
        s.simulate_electron_distribution(nb_traj=10,
            limit_x=[-0.250, 0.300], dx0=0.004, dx_increment=0.75)
        
class Test_running_sum:        
    def setUp(self):
        s = EDSSEMSpectrum(np.ones((2,2,3,1024)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 15.0
        
        self.signal = s 
        
    def test_running_sum(self):
        s = self.signal
        s.running_sum()
        
        assert_equal(s[0,0,0,0].data[0], 4.)
        
        s = self.signal
        s = s[0]
        
        s.running_sum()
        assert_equal(s[0,0,0].data[0], 16.)
        
        assert_equal(s.mapped_parameters.SEM.EDS.live_time,49.6)
        
class Test_plot_Xray_lines:        
    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.mapped_parameters.SEM.EDS.live_time = 3.1
        s.mapped_parameters.SEM.beam_energy = 15.0
        
        s.set_elements(('Al','Zn'))
        s.add_lines()
        
        self.signal = s 
        
    def test_plot_Xray_lines(self):
        s = self.signal
        
        s.plot_Xray_line()        
        s.plot_Xray_line(line_to_plot='all')
        s.plot_Xray_line(line_to_plot='a')
        s.plot_Xray_line(line_to_plot='ab')
        
        

class Test_tools_bulk:
    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        s.mapped_parameters.SEM.beam_energy = 5.0
        s.set_elements(['Al','Zn'])
        s.add_lines()
        self.signal = s
    def test_range(self):
        s = self.signal
        mp = s.mapped_parameters
        elec_range = utils.eds.electron_range(mp.Sample.elements[0],
            mp.SEM.beam_energy,rho='auto',tilt=mp.SEM.tilt_stage)
        assert_equal(elec_range,0.41350651162374225)
        
        density = utils.eds.density_from_composition(mp.Sample.elements,[0.8,0.2])
        xr_range = utils.eds.xray_range(mp.Sample.Xray_lines[0],
            mp.SEM.beam_energy,rho=density)
        assert_equal(xr_range,0.19002078834049554)
        
    

