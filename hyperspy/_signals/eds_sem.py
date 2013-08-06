# -*- coding: utf-8 -*-
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
from __future__ import division

import numpy as np
import traits.api as t
import os
import codecs
import subprocess
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import execnet


from hyperspy._signals.eds import EDSSpectrum
from hyperspy.gui.eds import SEMParametersUI
from hyperspy.defaults_parser import preferences
import hyperspy.gui.messages as messagesui
from hyperspy.decorators import only_interactive
from hyperspy.io import load
import hyperspy.components as components
from hyperspy.misc.eds import utils as utils_eds
from hyperspy import utils
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.config_dir import config_path, os_name, data_path


class EDSSEMSpectrum(EDSSpectrum):
    _signal_type = "EDS_SEM"
    
    def __init__(self, *args, **kwards):
        EDSSpectrum.__init__(self, *args, **kwards)
        # Attributes defaults        
        if hasattr(self.mapped_parameters, 'SEM.EDS') == False: 
            self._load_from_TEM_param()
        self._set_default_param()
        

    def get_calibration_from(self, ref, nb_pix=1):
        """Copy the calibration and all metadata of a reference.

        Primary use: To add a calibration to ripple file from INCA 
        software
                
        Parameters
        ----------
        ref : signal
            The reference contains the calibration in its 
            mapped_parameters 
        nb_pix : int
            The live time (real time corrected from the "dead time")
            is divided by the number of pixel (spectrums), giving an 
            average live time.          
        """
        
        
        self.original_parameters = ref.original_parameters.deepcopy()
        # Setup the axes_manager
        ax_m = self.axes_manager.signal_axes[0]
        ax_ref = ref.axes_manager.signal_axes[0]
        ax_m.scale = ax_ref.scale
        ax_m.units = ax_ref.units 
        ax_m.offset = ax_ref.offset
 
        
        # Setup mapped_parameters
        if hasattr(ref.mapped_parameters, 'SEM'):
            mp_ref = ref.mapped_parameters.SEM 
        elif hasattr(ref.mapped_parameters, 'TEM'):
            mp_ref = ref.mapped_parameters.TEM
        else:
            raise ValueError("The reference has no mapped_parameters.TEM"
            "\n nor mapped_parameters.SEM ")
            
        mp = self.mapped_parameters
        
        mp.SEM = mp_ref.deepcopy()
        
        if hasattr(mp_ref.EDS, 'live_time'):
            mp.SEM.EDS.live_time = mp_ref.EDS.live_time / nb_pix
                 

    
            
    def _load_from_TEM_param(self): 
        """Transfer mapped_parameters.TEM to mapped_parameters.SEM
        
        """      
         
        mp = self.mapped_parameters                     
        if mp.has_item('SEM') is False:
            mp.add_node('SEM')
        if mp.has_item('SEM.EDS') is False:
            mp.SEM.add_node('EDS') 
        mp.signal_type = 'EDS_SEM'
        
        #Transfer    
        if hasattr(mp,'TEM'):
            mp.SEM = mp.TEM
            del mp.__dict__['TEM']
        
    def _set_default_param(self): 
        """Set to value to default (defined in preferences)
        
        """  
        mp = self.mapped_parameters         
        if hasattr(mp.SEM, 'tilt_stage') is False:
            mp.SEM.tilt_stage = preferences.EDS.eds_tilt_stage
        if hasattr(mp.SEM.EDS, 'elevation_angle') is False:
            mp.SEM.EDS.elevation_angle = preferences.EDS.eds_detector_elevation
        if hasattr(mp.SEM.EDS, 'energy_resolution_MnKa') is False:
            mp.SEM.EDS.energy_resolution_MnKa = preferences.EDS.eds_mn_ka
        if hasattr(mp.SEM.EDS, 'azimuth_angle') is False:
            mp.SEM.EDS.azimuth_angle = preferences.EDS.eds_detector_azimuth 
                
               
    def set_microscope_parameters(self, beam_energy=None, live_time=None,
     tilt_stage=None, azimuth_angle=None, elevation_angle=None,
     energy_resolution_MnKa=None):
        """Set the microscope parameters that are necessary to quantify
        the spectrum.
        
        If not all of them are defined, raises in interactive mode 
        raises an UI item to fill the values
        
        Parameters
        ----------------
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
        mp_mic = self.mapped_parameters.SEM   
        
        if beam_energy is not None:
            mp_mic.beam_energy = beam_energy
        if live_time is not None:
            mp_mic.EDS.live_time = live_time
        if tilt_stage is not None:
            mp_mic.tilt_stage = tilt_stage
        if azimuth_angle is not None:
            mp_mic.EDS.azimuth_angle = azimuth_angle
        if tilt_stage is not None:
            mp_mic.EDS.elevation_angle = elevation_angle
        if energy_resolution_MnKa is not None:
            mp_mic.EDS.energy_resolution_MnKa  = energy_resolution_MnKa
            
        self._set_microscope_parameters()
            
    @only_interactive            
    def _set_microscope_parameters(self):    
        
        tem_par = SEMParametersUI()            
        mapping = {
        'SEM.beam_energy' : 'tem_par.beam_energy',        
        'SEM.tilt_stage' : 'tem_par.tilt_stage',
        'SEM.EDS.live_time' : 'tem_par.live_time',
        'SEM.EDS.azimuth_angle' : 'tem_par.azimuth_angle',
        'SEM.EDS.elevation_angle' : 'tem_par.elevation_angle',
        'SEM.EDS.energy_resolution_MnKa' : 'tem_par.energy_resolution_MnKa',}
       
        for key, value in mapping.iteritems():
            if self.mapped_parameters.has_item(key):
                exec('%s = self.mapped_parameters.%s' % (value, key))
        tem_par.edit_traits()
                  
        mapping = {
        'SEM.beam_energy' : tem_par.beam_energy,        
        'SEM.tilt_stage' : tem_par.tilt_stage,
        'SEM.EDS.live_time' : tem_par.live_time,
        'SEM.EDS.azimuth_angle' : tem_par.azimuth_angle,
        'SEM.EDS.elevation_angle' : tem_par.elevation_angle,
        'SEM.EDS.energy_resolution_MnKa' : tem_par.energy_resolution_MnKa,}
        
        
        for key, value in mapping.iteritems():
            if value != t.Undefined:
                exec('self.mapped_parameters.%s = %s' % (key, value))
        self._are_microscope_parameters_missing()
     
    def _are_microscope_parameters_missing(self):
        """Check if the EDS parameters necessary for quantification
        are defined in mapped_parameters. If not, in interactive mode 
        raises an UI item to fill the values
        
        """       
        
        must_exist = (
            'SEM.beam_energy',            
            'SEM.EDS.live_time', )

        
        missing_parameters = []
        for item in must_exist:
            exists = self.mapped_parameters.has_item(item)
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
        elements of 'mapped_parameters.Sample.elements'. A standard 
        spectrum is linked if its file name contains the elements name. 
        "C.msa", "Co.msa" but not "Co4.msa".
        
        Store the standard spectra in 'mapped_parameters.Sample.standard_spec'

        
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
        
        if not hasattr(self.mapped_parameters, 'Sample') :
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.mapped_parameters.Sample, 'elements'):
            raise ValueError("Add elements first, see 'set_elements'")
        

        std_tot = load(std_folder+"//*."+std_file_extension,signal_type 
          = 'EDS_SEM')
        mp = self.mapped_parameters        
        mp.Sample.standard_spec = []
        #for element in mp.Sample.elements:
        for Xray_line in mp.Sample.Xray_lines:
            element, line = utils_eds._get_element_and_line(Xray_line)  
            test_file_exist=False           
            for std in std_tot:    
                mp_std = std.mapped_parameters
                if element + "." in mp_std.original_filename:
                    test_file_exist=True
                    print("Standard file for %s : %s" % (element, 
                      mp_std.original_filename))
                    mp_std.title = element+"_std"
                    mp.Sample.standard_spec.append(std)
            if test_file_exist == False:
                print("\nStandard file for %s not found\n" % element)
        
    def top_hat(self, line_energy, width_windows = 1.):
        """
        Substact the background with a top hat filter. The width of the
        lobs are defined with the width of the peak at the line_energy.
        
        Parameters
        ----------------
        line_energy: float
            The energy in keV used to set the lob width calculate with
            FHWM_eds.
            
        width_windows: float or list(min,max)
            The width of the windows on which is applied the top_hat. 
            By default set to 1, which is equivalent to the size of the 
            filtering object. 
                       
        Notes
        -----
        See the textbook of Goldstein et al., Plenum publisher, 
        third edition p 399
        
        """
        offset = np.copy(self.axes_manager.signal_axes[0].offset)
        scale_s = np.copy(self.axes_manager.signal_axes[0].scale)
        FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        line_FWHM = utils_eds.FWHM(FWHM_MnKa, line_energy) 
        if np.ndim(width_windows) == 0:            
            det = [width_windows*line_FWHM,width_windows*line_FWHM]
        else :
            det = width_windows
        
        olob = int(round(line_FWHM/scale_s/2)*2)
        g = []
        for lob in range(-olob,olob):
            if abs(lob) > olob/2:
                g.append(-1./olob)
            else:
                g.append(1./(olob+1))    
        g = np.array(g)
   
        bornA = [int(round((line_energy-det[0]-offset)/scale_s)),\
        int(round((line_energy+det[1]-offset)/scale_s))]
  
        data_s = []
        for i in range(bornA[0],bornA[1]):
            data_s.append(self.data[...,i-olob:i+olob].dot(g))
            #data_s.append(self.data[...,i-olob:i+olob])
        data_s = np.array(data_s)
 
        dim = len(self.data.shape)
        #spec_th = EDSSEMSpectrum(np.rollaxis(data_s.dot(g),0,dim))
        spec_th = EDSSEMSpectrum(np.rollaxis(data_s,0,dim))

        return spec_th
        
    def _get_kratio(self,Xray_lines,plot_result):
        """
        Calculate the k-ratio without deconvolution
        """
        from hyperspy.hspy import create_model        
        width_windows=0.75 
        mp = self.mapped_parameters  
        
        for Xray_line in Xray_lines :
            element, line = utils_eds._get_element_and_line(Xray_line)  
            std = self.get_result(element,'standard_spec') 
            mp_std = std.mapped_parameters
            line_energy = elements_db[element]['Xray_energy'][line]
            diff_ltime = mp.SEM.EDS.live_time/mp_std.SEM.EDS.live_time
            #Fit with least square
            m = create_model(self.top_hat(line_energy,width_windows))
            fp = components.ScalableFixedPattern(std.top_hat(line_energy, 
              width_windows))
            fp.set_parameters_not_free(['offset','xscale','shift'])
            m.append(fp)          
            m.multifit(fitter='leastsq') 
            #store k-ratio
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result( Xray_line, 'kratios',\
                    fp.yscale.value/diff_ltime, plot_result)
            else:
                self._set_result( Xray_line, 'kratios',\
                    fp.yscale.as_signal().data/diff_ltime, plot_result)                  
               
    
        
    def get_kratio(self,deconvolution=None,plot_result=True):
        
        """
        Calculate the k-ratios by least-square fitting of the standard 
        sepectrum after background substraction with a top hat filtering
        
        Return a display of the resutls and store them in 
        'mapped_parameters.Sample.k_ratios'
        
        Parameters
        ----------
        plot_result : bool
            If True (default option), plot the k-ratio map.
        
        See also
        --------
        set_elements, link_standard, top_hat 
        
        """
        
        if not hasattr(self.mapped_parameters, 'Sample') :
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.mapped_parameters.Sample, 'elements'):
            raise ValueError("Add elements first, see 'set_elements'")
        if not hasattr(self.mapped_parameters.Sample, 'standard_spec') :
            raise ValueError("Add Standard, see 'link_standard'")

        mp = self.mapped_parameters         
        mp.Sample.kratios = list(np.zeros(len(mp.Sample.Xray_lines)))
        Xray_lines = list(mp.Sample.Xray_lines)
        
        if deconvolution is not None: 
            for deconvo in deconvolution:
                self._deconvolve_kratio(deconvo[0],deconvo[1],deconvo[2],
                  plot_result)
                for Xray_line in deconvo[0]:
                    Xray_lines.remove(Xray_line)
        if len(Xray_lines) > 0:     
            self._get_kratio(Xray_lines,plot_result)
    
    def _deconvolve_kratio(self,Xray_lines,elements,width_energy,\
        plot_result=True):
        """
        Calculate the k-ratio, applying a fit on a larger region with 
        selected X-ray lines
        """
        
        from hyperspy.hspy import create_model 
        line_energy = np.mean(width_energy)
        width_windows=[line_energy-width_energy[0],width_energy[1]-line_energy]
        
        m = create_model(self.top_hat(line_energy, width_windows))
        mp = self.mapped_parameters 
      
        diff_ltime =[]
        fps = []
        for element in elements:
            std = self.get_result(element,'standard_spec')
            fp = components.ScalableFixedPattern(std.top_hat(line_energy,
              width_windows))
            fp.set_parameters_not_free(['offset','xscale','shift'])
            fps.append(fp)    
            m.append(fps[-1])
            diff_ltime.append(mp.SEM.EDS.live_time/
              std.mapped_parameters.SEM.EDS.live_time)
        m.multifit(fitter='leastsq')
        i=0
        for Xray_line in Xray_lines:
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result( Xray_line, 'kratios',\
                    fps[i].yscale.value/diff_ltime[i], plot_result)
            else:
                self._set_result( Xray_line, 'kratios',\
                    fps[i].yscale.as_signal().data/diff_ltime[i], 
                      plot_result)
            i += 1

    def deconvolve_intensity(self,width_windows='all',plot_result=True):
        """
        Calculate the intensity by fitting standard spectra to the spectrum.
        
        Deconvoluted intensity is thus obtained compared to 
        get_intensity_map
        
        Needs standard to be set
        
        Parameters
        ----------
        
        width_windows: 'all' | [min energy, max energy]
            Set the energy windows in which the fit is done. If 'all'
            (default option), the whole energy range is used.
            
        plot_result : bool
            If True (default option), plot the intensity maps.
            
        See also
        --------
        
        set_elements, link_standard, get_intensity_map
        
        
        """
        from hyperspy.hspy import create_model 
        m = create_model(self)
        mp = self.mapped_parameters 
                
        elements = mp.Sample.elements
       
        fps = []
        for element in elements:
            std = self.get_result(element,'standard_spec')
            fp = components.ScalableFixedPattern(std)
            fp.set_parameters_not_free(['offset','xscale','shift'])
            fps.append(fp)    
            m.append(fps[-1])
        if width_windows != 'all':
            m.set_signal_range(width_windows[0],width_windows[1])
        m.multifit(fitter='leastsq')
        mp.Sample.intensities = list(np.zeros(len(elements)))
        i=0
        for element in elements:
            if (self.axes_manager.navigation_dimension == 0):
                self._set_result( element, 'intensities',\
                    fps[i].yscale.value, plot_result)
                if plot_result and i == 0:
                    m.plot()
                    plt.title('Fitted standard') 
            else:
                self._set_result( element, 'intensities',\
                    fps[i].yscale.as_signal().data, plot_result)
            i += 1
            

        
    
            
        
            
    def check_kratio(self,Xray_lines,width_energy='auto',
      top_hat_applied=False, plot_all_standard=False):
        """
        Compare the spectrum to the sum of the standard spectra scaled 
        in y with the k-ratios. The residual is ploted as well.
       
        Works only for spectrum.
        
        Parameters
        ----------        
        
        Xray_lines: list of string
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
        if width_energy=='auto':
            line_energy =[]
            for Xray_line in Xray_lines:
                element, line = utils_eds._get_element_and_line(Xray_line)  
                line_energy.append(elements_db[element]['Xray_energy'][line])
            width_energy = [0,0]
            width_energy[0] = np.min(line_energy)-utils_eds.FWHM(130,np.min(
              line_energy))*2
            width_energy[1] = np.max(line_energy)+utils_eds.FWHM(130,np.max(
              line_energy))*2
                
        line_energy = np.mean(width_energy)
        width_windows=[line_energy-width_energy[0],width_energy[1]\
            -line_energy]
            
        mp = self.mapped_parameters
        fig = plt.figure()
        if top_hat_applied:
            self_data = self.top_hat(line_energy, width_windows).data
        else:
            self_data = self[width_energy[0]:width_energy[1]].data
        plt.plot(self_data)
        leg_plot = ["Spec"]
        line_energies =[]
        intensities = []
        spec_sum = np.zeros(len(self.top_hat(line_energy, 
          width_windows).data))
        for Xray_line in Xray_lines:
            element, line = utils_eds._get_element_and_line(Xray_line)   
            line_energy = elements_db[element]['Xray_energy'][line]
            width_windows=[line_energy-width_energy[0],width_energy[1]-\
              line_energy]
            
            std_spec = self.get_result(element,'standard_spec')
            kratio = self.get_result(Xray_line,'kratios').data
            diff_ltime = mp.SEM.EDS.live_time/\
              std_spec.mapped_parameters.SEM.EDS.live_time
            if top_hat_applied:
                std_data = std_spec.top_hat(line_energy,
                width_windows).data*kratio*diff_ltime
            else:
                std_data = std_spec[width_energy[0]:width_energy[1]].data\
                        *kratio*diff_ltime
            if plot_all_standard:
                plt.plot(std_data)
                leg_plot.append(Xray_line)
            line_energies.append((line_energy-width_energy[0])/
              self.axes_manager[0].scale-self.axes_manager[0].offset)
            intensities.append(std_data[int(line_energies[-1])])
            spec_sum = spec_sum + std_data
        plt.plot(spec_sum)
        plt.plot(self_data-spec_sum)
        leg_plot.append("Sum")
        leg_plot.append("Residual")
        plt.legend(leg_plot)
        print("Tot residual: %s" % np.abs(self_data-spec_sum).sum())
        for i in range(len(line_energies)):
                plt.annotate(Xray_lines[i],xy = (line_energies[i],
                  intensities[i]))
        fig.show()
        
    def save_result(self, result, filename, Xray_lines='all',
      extension='hdf5'):
        """
        Save the result in a file (results stored in 
        'mapped_parameters.Sample')
        
        Parameters
        ----------
        result : string {'kratios'|'quant'|'intensities'}
            The result to save
            
        filename:
            the file path + the file name. The result and the Xray-lines
            is added at the end.
        
        Xray_lines: list of string
            the X-ray lines to save. If 'all' (default), save all X-ray lines
        
        Extension: 
            the extension in which the result is saved.
            
        See also
        -------        
        get_kratio, deconvolove_intensity, quant        
        
        """
        
        mp = self.mapped_parameters 
        if Xray_lines is 'all':
            if result == 'intensities':
                 Xray_lines = mp.Sample.elements
            else:
                Xray_lines = mp.Sample.Xray_lines
        for Xray_line in Xray_lines:
            if result == 'intensitiesS':
                res = self.intensity_map([Xray_line],plot_result=False)[0]
            else:
                res = self.get_result(Xray_line, result)
            if res.data.dtype == 'float64':
                a = 1
                res.change_dtype('float32')
                #res.change_dtype('uint32')
            res.save(filename=filename+"_"+result+"_"+Xray_line,
              extension = extension, overwrite = True) 
    
    
    def quant(self,plot_result=True,enh=False,enh_param=[0, 0.001,0.01,49]):        
        """
        Quantify using stratagem, a commercial software. A licence is 
        needed.
        
        k-ratios needs to be calculated before. Return a display of the 
        results and store them in 'mapped_parameters.Sample.quants'
        
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
        
        See also
        --------
        set_elements, link_standard, top_hat, get_kratio
        
        """
        mp = self.mapped_parameters
        if enh is False:
            foldername = os.path.join(config_path, 'strata_quant//')
            self._write_nbData_tsv(foldername + 'essai')
        elif enh is True and self.axes_manager.navigation_dimension == 3:
            if mp.has_item('elec_distr') is False:
                print("Error: Simulate an electron distribution first " +
                "with simulate_electron_distribution.")
                return 0 
            foldername = os.path.join(config_path, 'strata_quant_enh//')
            if mp.has_item('enh_param') is False:
                mp.add_node('enh_param')
            mp.enh_param['method'] = enh_param[0]
            mp.enh_param['limit_kratio_0'] = enh_param[1]
            mp.enh_param['limit_comp_same'] = enh_param[2]
            mp.enh_param['iter_max'] = enh_param[3]
            self._write_nbData_ehn_tsv(foldername + 'essai')
        else: 
            print("Error: Ehnanced quantification needs 3D data.")
            return 0 
        self._write_donnee_tsv(foldername + 'essai')
        p = subprocess.Popen(foldername + 'Debug//essai.exe')
        p.wait()
        self._read_result_tsv(foldername + 'essai',plot_result,enh=enh)
        
    def _read_result_tsv(self,foldername,plot_result,enh):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        
        f = codecs.open(foldername+'//result.tsv', encoding = encoding,
          errors = 'replace') 
        dim = list(self.data.shape)
        a = []
        for Xray_line in mp.Sample.Xray_lines:
            a.append([])        
        for line in f.readlines():
            for i in range(len(mp.Sample.Xray_lines)):
                a[i].append(float(line.split()[3+i]))            
        f.close()
        i=0
        if enh :
            mp.Sample.quant_enh = list(np.zeros(len(mp.Sample.Xray_lines)))
        else:
            mp.Sample.quant = list(np.zeros(len(mp.Sample.Xray_lines)))
        for Xray_line in mp.Sample.Xray_lines:  
            if (self.axes_manager.navigation_dimension == 0):
                data_quant=a[i][0]
            elif (self.axes_manager.navigation_dimension == 1):
                data_quant=np.array(a[i]).reshape((dim[0]))
            elif (self.axes_manager.navigation_dimension == 2):
                data_quant=np.array(a[i]).reshape((dim[1],dim[0])).T        
            elif (self.axes_manager.navigation_dimension == 3):                    
                data_quant=np.array(a[i]).reshape((dim[2],dim[1],
                  dim[0])).T
            if enh : 
                data_quant = data_quant[::,::-1]
                self._set_result( Xray_line, 'quant_enh',data_quant, plot_result)
            else:
                self._set_result( Xray_line, 'quant',data_quant, plot_result)        
            i += 1
        
    def _write_donnee_tsv(self, foldername):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        Xray_lines = mp.Sample.Xray_lines
        f = codecs.open(foldername+'//donnee.tsv', 'w', 
          encoding = encoding,errors = 'ignore') 
        dim = np.copy(self.axes_manager.navigation_shape).tolist()
        dim.reverse()
        if self.axes_manager.navigation_dimension == 0:
            f.write("1_1\r\n")
            for i in range(len(mp.Sample.Xray_lines)):
                f.write("%s\t" % mp.Sample.kratios[i].data)
        elif self.axes_manager.navigation_dimension == 1:
            for x in range(dim[0]):
                y = 0
                f.write("%s_%s\r\n" % (x+1,y+1))
                for Xray_line in Xray_lines:
                    f.write("%s\t" % self.get_result(Xray_line,
                      'kratios').data[x])
                f.write('\r\n')
        elif self.axes_manager.navigation_dimension == 2:
            for x in range(dim[1]):
                for y in range(dim[0]):
                    f.write("%s_%s\r\n" % (x+1,y+1))
                    for Xray_line in Xray_lines:
                        f.write("%s\t" % self.get_result(Xray_line,
                          'kratios').data[y,x])
                    f.write('\r\n')
        elif self.axes_manager.navigation_dimension == 3:
            for x in range(dim[2]):
                for y in range(dim[1]):
                    f.write("%s_%s\r\n" % (x+1,y+1))
                    for z in range(dim[0]):
                        for Xray_line in Xray_lines:
                            f.write("%s\t" % self.get_result(Xray_line,
                              'kratios').data[z,y,x])
                        f.write('\r\n')
        f.close()       
        
    
    def _write_nbData_tsv(self, foldername):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        f = codecs.open(foldername+'//nbData.tsv', 'w', 
          encoding = encoding,errors = 'ignore') 
        dim = np.copy(self.axes_manager.navigation_shape).tolist()
        #dim.reverse()
        dim.append(1)
        dim.append(1)
        if dim[0] == 0:
            dim[0] =1
        f.write("nbpixel_x\t%s\r\n" % dim[0])
        f.write('nbpixel_y\t%s\r\n' % dim[1])
        f.write('nbpixel_z\t%s\r\n' % dim[2])
        #f.write('pixelsize_z\t%s' % self.axes_manager[0].scale*1000)
        f.write('pixelsize_z\t100\r\n')
        f.write('nblayermax\t5\r\n')
        f.write('Limitkratio0\t0.001\r\n')
        f.write('Limitcompsame\t0.01\r\n')
        f.write('Itermax\t49\r\n')
        f.write('\r\n')
        f.write('HV\t%s\r\n'% mp.SEM.beam_energy)
        f.write('TOA\t%s\r\n'% utils_eds.TOA(self))
        f.write('azimuth\t%s\r\n'% mp.SEM.EDS.azimuth_angle)
        f.write('tilt\t%s\r\n'% -mp.SEM.tilt_stage)
        f.write('\r\n')
        f.write('nbelement\t%s\r\n'% len(mp.Sample.Xray_lines))
        elements = 'Element'
        z_el = 'Z'
        line_el = 'line'
        for Xray_line in mp.Sample.Xray_lines:
            el, line = utils_eds._get_element_and_line(Xray_line)  
            elements = elements + '\t' + el
            z_el = z_el + '\t' + str(elements_db[el]['Z'])
            if line == 'Ka':
                line_el = line_el + '\t0'
            if line== 'La':
                line_el = line_el + '\t1'
            if line == 'Ma':
                line_el = line_el + '\t2'    
        f.write('%s\r\n'% elements)
        f.write('%s\r\n'% z_el)
        f.write('%s\r\n'% line_el)
        f.close()
        
    def _write_nbData_ehn_tsv(self, foldername):
        encoding = 'latin-1'
        mp=self.mapped_parameters
        f = codecs.open(foldername+'//nbData.tsv', 'w', 
          encoding = encoding,errors = 'ignore') 
        dim = np.copy(self.axes_manager.navigation_shape).tolist()
        distr_dic = self.mapped_parameters.elec_distr
        scale = []
        for ax in self.axes_manager.navigation_axes:
            scale.append(ax.scale*1000)  
             
        elements = mp['Sample']['elements']
        limit_x = distr_dic['limit_x'] 
        dx0 = distr_dic['dx0'] 
        dx_increment  = distr_dic['dx_increment']    
        stat = distr_dic['distr']   
        #pixSize = self.axes_manager[2].scale
        pixLat = int((limit_x[1]-limit_x[0])/dx0+1) 
                
        distres = []
        for el, elm in enumerate(elements): 
            distres.append([]) 
            for i, distr in enumerate(stat[el]):                
                length = int((limit_x[1]-limit_x[0])/(dx0*(dx_increment*i+1)))
                distr = distr[int(pixLat/2.-round(length/2.)):
                    int(pixLat/2.+int(length/2.))] 
                if sum(distr) != 0:
                    distres[el].append([x/sum(distr) for x in distr])
            
        f.write("v2_\t%s\t2\t0.1\r\n" % mp.enh_param['method'])
        f.write("nbpixel_xyz\t%s\t%s\t%s\r\n" % (dim[0],dim[1],dim[2]))
        f.write('pixelsize_xyz\t%s\t%s\t%s\r\n' % (scale[0],scale[1],scale[2]))
        f.write('nblayermax\t%s\r\n' % max(distr_dic.max_slice_z))
        f.write('Limitkratio0\t%s\r\n' % mp.enh_param['limit_kratio_0']) 
        f.write('Limitcompsame\t%s\r\n' % mp.enh_param['limit_comp_same'])
        f.write('Itermax\t%s\r\n' % mp.enh_param['iter_max'])
        f.write('\r\n')
        f.write('HV\t%s\r\n'% mp.SEM.beam_energy)
        f.write('TOA\t%s\r\n'% utils_eds.TOA(self))
        f.write('azimuth\t%s\r\n'% mp.SEM.EDS.azimuth_angle)
        #Be carefull with that + or -
        f.write('tilt\t%s\r\n'% -mp.SEM.tilt_stage)
        f.write('\r\n')
        f.write('nbelement\t%s\r\n'% len(mp.Sample.Xray_lines))
        el_str = 'Element'
        z_el = 'Z'
        line_el = 'line'
        for Xray_line in mp.Sample.Xray_lines:
            el, line = utils_eds._get_element_and_line(Xray_line)  
            el_str = el_str + '\t' + el
            z_el = z_el + '\t' + str(elements_db[el]['Z'])
            if line == 'Ka':
                line_el = line_el + '\t0'
            if line== 'La':
                line_el = line_el + '\t1'
            if line == 'Ma':
                line_el = line_el + '\t2'    
        f.write('%s\r\n'% el_str)
        f.write('%s\r\n'% z_el)
        f.write('%s\r\n'% line_el)
        f.write('\r\n')
        f.write('DistrX_Min_Max_Dx_IncF\t%s\t%s\t%s\t%s\r\n' % 
            (limit_x[0]*1000,limit_x[1]*1000,dx0*1000,dx_increment))
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
        dx_increment  = distr_dic['dx_increment']    
        stat = distr_dic['distr']   
        #pixSize = self.axes_manager[2].scale
        pixLat = int((limit_x[1]-limit_x[0])/dx0+1) 
        
    #def check_total(self):
        #img_0 = self.get_result(Xray_lines[0],'kratios')
        
        #data_total = np.zeros_like(img_0.data) 
        #for Xray_line in Xray_lines:
            #data_total += self.get_result(Xray_line,'kratios').data
            
        #img_total = img_0.deepcopy
        #img_total.data = data_total
        #return img_total 
        
    def simulate_electron_distribution(self, 
        nb_traj, 
        limit_x, 
        dx0,
        dx_increment,
        detector='Si(Li)',
        plot_result=False,
        gateway='auto'):
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
        
        mp = self.mapped_parameters     
        dic = self.deepcopy().mapped_parameters.as_dictionary()           
        if hasattr(mp.Sample, 'elements') is False:
            print 'Elements needs to be defined'
            return 0
            
        elements = list(dic['Sample']['elements'])
        e0 = dic['SEM']['beam_energy']
        tilt = dic['SEM']['tilt_stage']
            
     
        #Units!! 
        pixSize = [dx0*1.0e-6, 0.2*1.0e-6,
            self.axes_manager[2].scale *1.0e-6]
        nblayer = []
        for el in elements:
            nblayer.append(utils_eds.electron_range(el,e0,tilt=tilt))            
            
        nblayermax = int(round(max(nblayer)/self.axes_manager[2].scale))
        pixLat = [int((limit_x[1]-limit_x[0])/dx0+1), nblayermax]
        dev = (limit_x[1] + limit_x[0])*1.0e-6

        if 1 == 0:
            #AlZn nTraj = 20000
            dx_increment = 0.75
            #pixLat = nbx nbz
            pixSize = [4*1.0e-9,200*1.0e-9,40*1.0e-9] #y,x,z
            pixLat = [138, 7] #(max -min)/y +1 (ou 0.5), maxeldepth #nb de pixel
            dev =  50*1.0e-9 #min+max, deviation du centre
        if 1 == 0:
            #TiFeNi
            dx_increment = 0.5
            pixSize = (8*1.0e-9,200*1.0e-9,100*1.0e-9)
            pixLat =(100, 5)
            dev =  100*1.0e-9

        
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
            tiltD = -""" + str(tilt) + """
            
            nTraj = """ + str(nb_traj) + """
            IncrementF = """ + str(dx_increment) + """
            pixSize = """ + str(pixSize) + """
            pixLat = """ + str(pixLat) + """
            dev = """ + str(dev) + """
            pixTot = pixLat[0]*pixLat[1]
            tilt = math.radians(tiltD) # tilt angle radian

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
            
        i=0        
        stat = np.zeros([len(elements),pixLat[1],pixLat[0]])
        for el, elm in enumerate(elements): 
            for z in range(pixLat[1]):
                    for x in range(pixLat[0]):
                        stat[el,z,x] = datas[i]
                        i = i+1
        distres = []
        xdatas = []
        for el, elm in enumerate(elements): 
            distres.append([]) 
            xdatas.append([])
            if plot_result:
                f = plt.figure()
                leg=[]
            for i, distr in enumerate(stat[el]):                
                length = int((limit_x[1]-limit_x[0])/(dx0*(dx_increment*i+1)))
                distr = distr[int(pixLat[0]/2.-round(length/2.)):
                    int(pixLat[0]/2.+int(length/2.))] 
                if sum(distr) != 0:
                    xdata =[]
                    for x in range(length):
                        xdata.append(limit_x[0]+x*dx0*(dx_increment*i+1))
                    if plot_result:     
                        leg.append('z slice ' + str(pixSize[2]*1.0e6*i)+ ' ${\mu}m$')
                        plt.plot(xdata,distr)
                    distres[el].append([x/sum(distr) for x in distr])
                    xdatas[el].append(xdata)
        
            if plot_result:
                plt.legend(leg,loc=2)
                plt.title(elm + ': Electron depth distribution')
                plt.xlabel('x position [${\mu}m$]')
                plt.ylabel('nb electrons')
                
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
        
    
        
    def plot_electron_distribution(self):
        """Retrieve and plot the electron distribution from 
        simulate_electron_distribution
        """
        
        mp = self.mapped_parameters
        
        if mp.has_item('elec_distr') is False:
            print("Error: Simulate an electron distribution first " +
            "with simulate_electron_distribution.")
            return 0 
        elements = mp['Sample']['elements']
        limit_x = mp.elec_distr['limit_x'] 
        dx0 = mp.elec_distr['dx0'] 
        dx_increment  = mp.elec_distr['dx_increment']    
        stat = mp.elec_distr['distr']
        nb_traj = mp.elec_distr['nb_traj'] 
        
        pixSize = self.axes_manager[2].scale
        pixLat = int((limit_x[1]-limit_x[0])/dx0+1)

        for el, elm in enumerate(elements): 
            
            f = plt.figure()
            leg=[]
            for i, distr in enumerate(stat[el]):                
                length = int((limit_x[1]-limit_x[0])/(dx0*(dx_increment*i+1)))
                distr = distr[int(pixLat/2.-round(length/2.)):
                    int(pixLat/2.+int(length/2.))]
                xdata =[]
                for x in range(length):
                    xdata.append(limit_x[0]+x*dx0*(dx_increment*i+1))     
                leg.append('z slice ' + str(pixSize*i)+ ' ${\mu}m$')
                plt.plot(xdata,distr)    
            
            plt.legend(leg,loc=2)
            plt.title(elm + ': Electron depth distribution (nb traj :' 
                + str(nb_traj) +')')
            plt.xlabel('x position [${\mu}m$]')
            plt.ylabel('nb electrons / sum electrons in the layer')
            
    def save(self, filename=None, overwrite=None, extension=None,
             **kwds):
        """Saves the signal in the specified format.

        The function gets the format from the extension.:
            - hdf5 for HDF5
            - rpl for Ripple (useful to export to Digital Micrograph)
            - msa for EMSA/MSA single spectrum saving.
            - Many image formats such as png, tiff, jpeg...

        If no extension is provided the default file format as defined 
        in the `preferences` is used.
        Please note that not all the formats supports saving datasets of
        arbitrary dimensions, e.g. msa only suports 1D data.
        
        Each format accepts a different set of parameters. For details 
        see the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None (default) and tmp_parameters.filename and 
            `tmp_paramters.folder` are defined, the
            filename and path will be taken from there. A valid
            extension can be provided e.g. "my_file.rpl", see `extension`.
        overwrite : None, bool
            If None, if the file exists it will query the user. If 
            True(False) it (does not) overwrites the file if it exists.
        extension : {None, 'hdf5', 'rpl', 'msa',common image extensions e.g. 'tiff', 'png'}
            The extension of the file that defines the file format. 
            If None, the extesion is taken from the first not None in the follwoing list:
            i) the filename 
            ii)  `tmp_parameters.extension`
            iii) `preferences.General.default_file_format` in this order. 
        """
        mp = self.mapped_parameters
        if hasattr(mp, 'Sample'):
            if hasattr(mp.Sample, 'standard_spec'):
                l_time = []
                for el in range(len(mp.Sample.elements)):
                    l_time.append(
                        mp.Sample.standard_spec[el].mapped_parameters.SEM.EDS.live_time)
                mp.Sample.standard_spec = utils.stack(mp.Sample.standard_spec)
                mp.Sample.standard_spec.mapped_parameters.SEM.EDS.live_time = l_time 
            for result in ['kratios','quant','quant_enh','intensities']:
                if hasattr(mp.Sample, result):
                    mp.Sample[result] = utils.stack(mp.Sample[result])
              
        
        super(EDSSEMSpectrum, self).save(filename, overwrite, extension)
        
    def align_result(self,results='all',reference=['kratios',0],starting_slice=0):
        """Align the results on the same alignement matrix.
        
        A reference stack with another signal can be used.
          
        Parameters
        ----------                  
        results: 'all' | List of string
            The list of result to be align. If 'all', all result are aligned.          
        reference: [result,elements] | image
            The reference is used to gerenerate an alignement matrix.            
        starting_slice: int
            The starting slice for the alignment.
            
        See also
        --------                
        align_with_stackReg
        
        Notes
        -----
        Defined by P. Thevenaz, U. Ruttimann, and M. Unser,
        IEEE Transaction on IMage Processing 7(1), pp 27-41 (1998)    
        
        The version of MulitStackReg has been modified. Translation and save 
        save the alignement is used.
                
        """
        
        
            

    
    
