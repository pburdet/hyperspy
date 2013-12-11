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
import matplotlib.pyplot as plt
import copy

from hyperspy._signals.spectrum import Spectrum
from hyperspy.signal import Signal
from hyperspy._signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.utils import isiterable
import hyperspy.components as components
from hyperspy import utils

class EDSSpectrum(Spectrum):
    _signal_type = "EDS"
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        if self.mapped_parameters.signal_type == 'EDS':
            print('The microscope type is not set. Use '
            'set_signal_type(\'EDS_TEM\') or set_signal_type(\'EDS_SEM\')')
            
    def sum(self, axis):
        """Sum the data over the given axis.

        Parameters
        ----------
        axis : {int, string}
           The axis can be specified using the index of the axis in 
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1).data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.sum(-1, True).plot()
        
        """
        #modify time spend per spectrum
        if hasattr(self.mapped_parameters, 'SEM'):
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            mp.EDS.live_time = mp.EDS.live_time * self.axes_manager.shape[axis]
        return super(EDSSpectrum, self).sum(axis)
        
    def rebin(self, new_shape):
        """Rebins the data to the new shape

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape must be a divisor of the original shape
            
        """
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) / 
                           np.array(new_shape_in_array))
        s = super(EDSSpectrum, self).rebin(new_shape)
        #modify time per spectrum
        if "SEM.EDS.live_time" in s.mapped_parameters:
            for factor in factors:
                s.mapped_parameters.SEM.EDS.live_time *= factor
        if "TEM.EDS.live_time" in s.mapped_parameters:
            for factor in factors:
                s.mapped_parameters.TEM.EDS.live_time *= factor
        return s
    
    def set_elements(self, elements):
        """Erase all elements and set them.
        
        Parameters
        ----------
        elements : list of strings
            A list of chemical element symbols.  
               
        See also
        --------
        add_elements, set_line, add_lines.
            
        Examples
        --------
        
        >>> s = signals.EDSSEMSpectrum(np.arange(1024))
        >>> s.set_elements(['Ni', 'O'],['Ka','Ka'])   
        Adding Ni_Ka Line
        Adding O_Ka Line
        >>> s.mapped_paramters.SEM.beam_energy = 10
        >>> s.set_elements(['Ni', 'O'])
        Adding Ni_La Line
        Adding O_Ka Line
        
        """          
        #Erase previous elements and X-ray lines
        if "Sample.elements" in self.mapped_parameters:
            del self.mapped_parameters.Sample.elements
        self.add_elements(elements)
        
    def add_elements(self, elements):
        """Add elements and the corresponding X-ray lines.
        
        The list of elements is stored in `mapped_parameters.Sample.elements`     
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
            
        See also
        --------
        set_elements, add_lines, set_lines.
        
        """
        if not isiterable(elements) or isinstance(elements, basestring):
            raise ValueError(
            "Input must be in the form of a list. For example, "
            "if `s` is the variable containing this EELS spectrum:\n "
            ">>> s.add_elements(('C',))\n"
            "See the docstring for more information.")
        if "Sample.elements" in self.mapped_parameters:
            elements_ = set(self.mapped_parameters.Sample.elements)
        else:
            elements_ = set()
        for element in elements:            
            if element in elements_db:               
                elements_.add(element)
            else:
                raise ValueError(
                    "%s is not a valid chemical element symbol." % element)
                   
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
            
        self.mapped_parameters.Sample.elements = sorted(list(elements_))
                                                        
                                            
    def set_lines(self,
                  lines,
                  only_one=True,
                  only_lines=("Ka", "La", "Ma")):
        """Erase all Xrays lines and set them.
        
        See add_lines for details.
        
        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb. 
            Additionally, if `mapped_parameters.Sample.elements` is 
            defined, add the lines of those elements that where not
            given in this list.
        only_one: bool
            If False, add all the lines of each element in 
            `mapped_parameters.Sample.elements` that has not line 
            defined in lines. If True (default), 
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.
               
        See also
        --------
        add_lines, add_elements, set_elements..
        
        """          
        if "Sample.Xray_lines" in self.mapped_parameters:
            del self.mapped_parameters.Sample.Xray_lines
        self.add_lines(lines=lines,
                       only_one=only_one,
                       only_lines=only_lines)
    
    def add_lines(self,
                  lines=(),
                  only_one=True,
                  only_lines=("Ka", "La", "Ma")):
        """Add X-rays lines to the internal list.
        
        Although most functions do not require an internal list of 
        X-ray lines because they can be calculated from the internal 
        list of elements, ocassionally it might be useful to customize the 
        X-ray lines to be use by all functions by default using this method.
        The list of X-ray lines is stored in 
        `mapped_parameters.Sample.Xray_lines`
        
        Parameters
        ----------
        lines : list of strings
            A list of valid element X-ray lines to add e.g. Fe_Kb. 
            Additionally, if `mapped_parameters.Sample.elements` is 
            defined, add the lines of those elements that where not
            given in this list. If the list is empty (default), and 
            `mapped_parameters.Sample.elements` is 
            defined, add the lines of all those elements.
        only_one: bool
            If False, add all the lines of each element in 
            `mapped_parameters.Sample.elements` that has not line 
            defined in lines. If True (default), 
            only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be added.
               
        See also
        --------
        set_lines, add_elements, set_elements.
        
        """
        if "Sample.Xray_lines" in self.mapped_parameters:
            Xray_lines = set(self.mapped_parameters.Sample.Xray_lines)
        else:
            Xray_lines = set()
        # Define the elements which Xray lines has been customized
        # So that we don't attempt to add new lines automatically
        elements = set()
        for line in Xray_lines:
            elements.add(line.split("_")[0])
        end_energy = self.axes_manager.signal_axes[0].high_value            
        for line in lines:
            try:
                element, subshell = line.split("_")
            except ValueError:
                raise ValueError(
                    "Invalid line symbol. "
                    "Please provide a valid line symbol e.g. Fe_Ka")
            if element in elements_db:
                elements.add(element)
                if subshell in elements_db[element]['Xray_energy']:
                    lines_len = len(Xray_lines)
                    Xray_lines.add(line)
                    #if lines_len != len(Xray_lines):
                    #    print("%s line added," % line)
                    #else:
                    #    print("%s line already in." % line)
                    if (elements_db[element]['Xray_energy'][subshell] > 
                            end_energy):
                      print("Warning: %s %s is above the data energy range." 
                             % (element, subshell))  
                else:
                    raise ValueError(
                        "%s is not a valid line of %s." % (line, element))
            else:
                raise ValueError(
                    "%s is not a valid symbol of an element." % element)
        if "Sample.elements" in self.mapped_parameters:
            extra_elements = (set(self.mapped_parameters.Sample.elements) - 
                              elements)
            if extra_elements:
                new_lines = self._get_lines_from_elements(
                                            extra_elements,
                                            only_one=only_one,
                                            only_lines=only_lines)
                if new_lines:
                    self.add_lines(new_lines)
        self.add_elements(elements)
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        if "Sample.Xray_lines" in self.mapped_parameters:
            Xray_lines = Xray_lines.union(
                    self.mapped_parameters.Sample.Xray_lines)
        self.mapped_parameters.Sample.Xray_lines = sorted(list(Xray_lines))
        
        
            
    def _get_lines_from_elements(self,
                                 elements,
                                 only_one=False,
                                 only_lines=("Ka", "La", "Ma")):
        """Returns the X-ray lines of the given elements in spectral range
        of the data. 
        
        Parameters
        ----------
        elements : list of strings
            A list containing the symbol of the chemical elements. 
        only_one : bool
            If False, add all the lines of each element in the data spectral
            range. If True only add the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, only the given lines will be returned.
            
            
        Returns
        -------
            
        """
        if hasattr(self.mapped_parameters, 'SEM') and \
            hasattr(self.mapped_parameters.SEM,'beam_energy') :
            beam_energy = self.mapped_parameters.SEM.beam_energy
        elif hasattr(self.mapped_parameters, 'TEM') and \
            hasattr(self.mapped_parameters.TEM,'beam_energy') :
            beam_energy = self.mapped_parameters.TEM.beam_energy
        else:
            raise AttributeError(
                "To use this method the beam energy `TEM.beam_energy` "
                "or `SEM.beam_energy` must be defined in "
                "`mapped_parameters`.")
        
        end_energy = self.axes_manager.signal_axes[0].high_value
        if beam_energy < end_energy:
           end_energy = beam_energy
        lines = []         
        for element in elements:
            #Possible line (existing and excited by electron)
            element_lines = []
            for subshell in elements_db[element]['Xray_energy'].keys():
                if only_lines and subshell not in only_lines:
                    continue
                if (elements_db[element]['Xray_energy'][subshell] < 
                        end_energy):
                    
                    element_lines.append(element + "_" + subshell)
            if only_one and element_lines:           
            #Choose the best line
                select_this = -1            
                for i, line in enumerate(element_lines):
                    if (elements_db[element]['Xray_energy']
                        [line.split("_")[1]] < beam_energy / 2):
                        select_this = i
                        break
                element_lines = [element_lines[select_this],]
                     
            if not element_lines:
                print(("There is not X-ray line for element %s " % element) + 
                       "in the data spectral range")
            else:
                lines.extend(element_lines)
        return lines
                             
    def get_lines_intensity(self,
                            Xray_lines=None,
                            plot_result=False,
                            integration_window_factor=2.,
                            only_one=True,
                            only_lines=("Ka", "La", "Ma"),
                            lines_deconvolution=None,
                            bck=0,
                            plot_fit=False):
        """Return the intensity map of selected Xray lines.
        
        The intensity maps are computed by integrating the spectrum over the 
        different X-ray lines. The integration window width
        is calculated from the energy resolution of the detector
        defined as defined in 
        `self.mapped_parameters.SEM.EDS.energy_resolution_MnKa` or 
        `self.mapped_parameters.SEM.EDS.energy_resolution_MnKa`.
        
        
        Parameters
        ----------
        
        Xray_lines: {None, "best", list of string}
            If None,
            if `mapped.parameters.Sample.elements.Xray_lines` contains a 
            list of lines use those.
            If `mapped.parameters.Sample.elements.Xray_lines` is undefined
            or empty but `mapped.parameters.Sample.elements` is defined, 
            use the same syntax as `add_line` to select a subset of lines
            for the operation.
            Alternatively, provide an iterable containing 
            a list of valid X-ray lines symbols.
        plot_result : bool
            If True, plot the calculated line intensities. If the current 
            object is a single spectrum it prints the result instead.
        integration_window_factor: Float
            The integration window is centered at the center of the X-ray
            line and its width is defined by this factor (2 by default) 
            times the calculated FWHM of the line.
        only_one : bool
            If False, use all the lines of each element in the data spectral
            range. If True use only the line at the highest energy
            above an overvoltage of 2 (< beam energy / 2).
        only_lines : {None, list of strings}
            If not None, use only the given lines.
        lines_deconvolution : None | 'model' | 'standard'
            Deconvolution of the line with a gaussian model. Take time
        bck : float
            background to substract. Only for deconvolution
            
        Returns
        -------
        intensities : list
            A list containing the intensities as Signal subclasses.
            
        Examples
        --------
        
        >>> specImg.plot_intensity_map(["C_Ka", "Ta_Ma"])
        
        See also
        --------
        
        set_elements, add_elements.
        
        """
        from hyperspy.hspy import create_model 
        if Xray_lines is None:
            if 'Sample.Xray_lines' in self.mapped_parameters:
                Xray_lines = self.mapped_parameters.Sample.Xray_lines
            elif 'Sample.elements' in self.mapped_parameters:
                Xray_lines = self._get_lines_from_elements(
                        self.mapped_parameters.Sample.elements,
                        only_one=only_one,
                        only_lines=only_lines)
            else:
                raise ValueError(
                    "Not X-ray line, set them with `add_elements`")
                            
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        elif self.mapped_parameters.signal_type == 'EDS_TEM':
            FWHM_MnKa = self.mapped_parameters.TEM.EDS.energy_resolution_MnKa
        else:
            raise NotImplementedError(
                "This method only works for EDS_TEM or EDS_SEM signals. "
                "You can use `set_signal_type(\"EDS_TEM\")` or"
                "`set_signal_type(\"EDS_SEM\")` to convert to one of these"
                "signal types.")                 
        intensities = []
        #test 1D Spectrum (0D problem)
            #signal_to_index = self.axes_manager.navigation_dimension - 2                  
        if lines_deconvolution is None:
            for Xray_line in Xray_lines:                
                element, line = utils_eds._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = utils_eds.FWHM(FWHM_MnKa,line_energy)
                det = integration_window_factor * line_FWHM / 2.
                img = self[...,line_energy - det:line_energy + det
                        ].integrate_simpson(-1)
                img.mapped_parameters.title = (
                    'Intensity of %s at %.2f %s from %s' % 
                    (Xray_line,
                     line_energy,
                     self.axes_manager.signal_axes[0].units,
                     self.mapped_parameters.title)) 
                if img.axes_manager.navigation_dimension >= 2:
                    img = img.as_image([0,1])
                #useless never the case
                elif img.axes_manager.navigation_dimension == 1:
                    img.axes_manager.set_signal_dimension(1)
                if plot_result:
                    if img.axes_manager.signal_dimension != 0:
                        img.plot(navigator=None)
                    else:
                        print("%s at %s %s : Intensity = %.3f" 
                        % (Xray_line,
                           line_energy,
                           self.axes_manager.signal_axes[0].units,
                           img.data))
                intensities.append(img)                                
        else:
            fps = []
            if lines_deconvolution == 'standard':  
                m = create_model(self)
            else : 
                s = self - bck
                m = create_model(s)                
                
            for Xray_line in Xray_lines:
                element, line = utils_eds._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = utils_eds.FWHM(FWHM_MnKa,line_energy)
                if lines_deconvolution == 'model':   
                    fp = components.Gaussian()    
                    fp.centre.value = line_energy
                    fp.name = Xray_line
                    fp.sigma.value = line_FWHM/2.355 
                    fp.centre.free = False
                    fp.sigma.free = False
                elif lines_deconvolution == 'standard':
                    std = self.get_result(element,'standard_spec')
                    std[:line_energy-1.5*line_FWHM] = 0
                    std[line_energy+1.5*line_FWHM:] = 0
                    fp = components.ScalableFixedPattern(std)
                    fp.set_parameters_not_free(['offset','xscale','shift'])
                fps.append(fp)    
                m.append(fps[-1])
                if lines_deconvolution == 'model':
                    for li in elements_db[element]['Xray_energy']:
                        if line[0] in li and line != li: 
                            line_energy = elements_db[element]['Xray_energy'][li]
                            line_FWHM = utils_eds.FWHM(FWHM_MnKa,line_energy)
                            fp = components.Gaussian()    
                            fp.centre.value = line_energy
                            fp.name = element + '_' + li
                            fp.sigma.value = line_FWHM/2.355 
                            fp.A.twin = fps[-1].A             
                            fp.centre.free = False
                            fp.sigma.free = False
                            ratio_line = elements_db['lines']['ratio_line'][li]
                            fp.A.twin_function = lambda x: x * ratio_line
                            fp.A.twin_inverse_function = lambda x: x / ratio_line
                            m.append(fp)
            m.multifit()
            if plot_fit:
                m.plot()
                plt.title('Fit') 
            for i, fp in enumerate(fps):                
                Xray_line = Xray_lines[i] 
                element, line = utils_eds._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                
                if self.axes_manager.navigation_dimension == 0:
                    if lines_deconvolution == 'model': 
                        data_res = fp.A.value
                    elif lines_deconvolution == 'standard': 
                        data_res = fp.yscale.value 
                else: 
                    if lines_deconvolution == 'model': 
                        data_res = fp.A.as_signal().data
                    elif lines_deconvolution == 'standard': 
                        data_res = fp.yscale.as_signal().data
                img = self._set_result( Xray_line, 'Int', 
                    data_res, plot_result=False, store_in_mp=False)             
                
                #img = self[...,0]             
                #if img.axes_manager.navigation_dimension >= 2:                    
                    #img = img.as_image([0,1])
                    #if lines_deconvolution == 'model': 
                        #img.data = fp.A.as_signal().data
                    #elif lines_deconvolution == 'standard': 
                        #img.data = fp.yscale.as_signal().data
                #elif img.axes_manager.navigation_dimension == 1:
                    #img.axes_manager.set_signal_dimension(1) 
                    #if lines_deconvolution == 'model': 
                        #img.data = fp.A.as_signal().data
                    #elif lines_deconvolution == 'standard': 
                        #img.data = fp.yscale.as_signal().data  
                #elif img.axes_manager.navigation_dimension == 0:
                    #img = img.sum(0)
                    #if lines_deconvolution == 'model': 
                        #img.data = fp.A.value
                    #elif lines_deconvolution == 'standard': 
                        #img.data = fp.yscale.value 
                                 
                img.mapped_parameters.title = (
                    'Intensity of %s at %.2f %s from %s' % 
                    (Xray_line,
                     line_energy,
                     self.axes_manager.signal_axes[0].units,
                     self.mapped_parameters.title)) 
                if plot_result:
                    if img.axes_manager.signal_dimension != 0:
                        img.plot(navigator=None)
                    else:
                        print("%s at %s %s : Intensity = %.3f" 
                        % (Xray_line,
                           line_energy,
                           self.axes_manager.signal_axes[0].units,
                           img.data))
                intensities.append(img)
        return intensities

    def running_sum(self,shape_convo='square',corner=-1) :
        #cross not tested
        """
        Apply a running sum on the data.
        
        Parameters
        ----------
        
        shape_convo: 'square'|'cross'
            Define the shape to convolve with
        
        corner : -1 || 1
            For square, running sum induce a shift of the images towards 
            one of the corner: if -1, towards top left, if 1 towards 
            bottom right.
            For 'cross', if -1 vertical/horizontal cross, if 1 from corner
            to corner.
        
        """
        dim = self.data.shape
        data_s = np.zeros_like(self.data)        
        data_s = np.insert(data_s, 0, 0,axis=-3)
        data_s = np.insert(data_s, 0, 0,axis=-2)    
        if shape_convo == 'square':
            end_mirrors = [[0,0],[-1,0],[0,-1],[-1,-1]]
            for end_mirror in end_mirrors:  
                tmp_s=np.insert(self.data, end_mirror[0], self.data[...,end_mirror[0],:,:],axis=-3)
                data_s += np.insert(tmp_s, end_mirror[1], tmp_s[...,end_mirror[1],:],axis=-2)
            if corner == -1:
                data_s = data_s[...,1:,:,:][...,1:,:]
            else :
                data_s = data_s[...,:-1,:,:][...,:-1,:]
                
            
            
        elif shape_convo == 'cross':
            data_s = np.insert(data_s, 0, 0,axis=-3)
            data_s = np.insert(data_s, 0, 0,axis=-2)
            if corner == -1:
                end_mirrors = [[0,-1,0,-1],[-1,-1,0,-1],[0,0,0,-1],[0,-1,0,0],[0,-1,-1,-1]]
            elif corner == 1:
                end_mirrors = [[0,-1,0,-1],[0,0,0,0],[-1,-1,0,0],[0,0,-1,-1],[-1,-1,-1,-1]]
            else:
                end_mirrors = [[0,-1,0,-1],[-1,-1,0,-1],[0,0,0,-1],[0,-1,0,0], 
                [0,-1,-1,-1],[0,0,0,0],[-1,-1,0,0],[0,0,-1,-1],[-1,-1,-1,-1]]
                
            for end_mirror in end_mirrors:  
                tmp_s=np.insert(self.data, end_mirror[0], self.data[...,end_mirror[0],:,:],axis=-3)
                tmp_s=np.insert(tmp_s, end_mirror[1], tmp_s[...,end_mirror[0],:,:],axis=-3)
                tmp_s=np.insert(tmp_s, end_mirror[2], tmp_s[...,end_mirror[1],:],axis=-2)
                data_s += np.insert(tmp_s, end_mirror[3], tmp_s[...,end_mirror[1],:],axis=-2)
            data_s = data_s[...,1:-2,:,:][...,1:-2,:]

        
        if hasattr(self.mapped_parameters, 'SEM'):            
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            mp.EDS.live_time = mp.EDS.live_time * len(end_mirrors)
        self.data = data_s
        
    def plot_Xray_line(self,line_to_plot='selected'):
        """
        Annotate a spec.plot() with the name of the selected X-ray 
        lines
        
        Parameters
        ----------
        
        line_to_plot: string 'selected'|'a'|'ab|'all'
            Defined which lines to annotate. 'selected': the selected one,
            'a': all alpha lines of the selected elements, 'ab': all alpha and 
            beta lines, 'all': all lines of the selected elements
        
        See also
        --------
        
        set_elements, add_elements 
        
        """
        if self.axes_manager.navigation_dimension > 0:
            raise ValueError("Works only for single spectrum")
        
        
        mp = self.mapped_parameters
        if hasattr(self.mapped_parameters, 'SEM') and\
          hasattr(self.mapped_parameters.SEM, 'beam_energy'): 
            beam_energy = mp.SEM.beam_energy
        elif hasattr(self.mapped_parameters, 'TEM') and\
          hasattr(self.mapped_parameters.TEM, 'beam_energy'):  
            beam_energy = mp.TEM.beam_energy
        else:
           beam_energy = 300 
        
        elements = []
        lines = []    
        if line_to_plot=='selected':            
            Xray_lines = mp.Sample.Xray_lines
            for Xray_line in Xray_lines:
                element, line = utils_eds._get_element_and_line(Xray_line)
                elements.append(element)
                lines.append(line)

        else:
            for element in mp.Sample.elements:
                for line, en in elements_db[element]['Xray_energy'].items():
                    if en < beam_energy:
                        if line_to_plot=='a' and line[1]=='a':
                            elements.append(element)
                            lines.append(line)
                        elif line_to_plot=='ab':
                            if line[1]=='a' or line[1]=='b':  
                                elements.append(element)
                                lines.append(line)
                        elif line_to_plot=='all':
                            elements.append(element)
                            lines.append(line)                                          
                
        Xray_lines =[]
        line_energy =[]
        intensity = []       
        for i, element in enumerate(elements):                   
            line_energy.append(elements_db[element]['Xray_energy'][lines[i]])
            if lines[i]=='a':
                intensity.append(self[line_energy[-1]].data[0])
            else:                
                relative_factor=elements_db['lines']['ratio_line'][lines[i]]
                a_eng=elements_db[element]['Xray_energy'][lines[i][0]+'a']
                intensity.append(self[a_eng].data[0]*relative_factor)
            Xray_lines.append(element+'_'+lines[i])
            

        self.plot() 
        for i in range(len(line_energy)):
            plt.text(line_energy[i],intensity[i]*1.1,Xray_lines[i],
              rotation =90)
            plt.vlines(line_energy[i],0,intensity[i]*0.8,color='black')
            
    def calibrate_energy_resolution(self,Xray_line,bck='auto',
        set_Mn_Ka=True,model_plot=True):
        """
        Calibrate the energy resolution from a peak
        
        Estimate the FHWM of the peak, estimate the energy resolution and
        extrapolate to FWHM of Mn Ka
        
        Parameters:
        Xray_line : str
            the selected X-ray line. It shouldn't have peak around
            
        bck: float | 'auto'
            the linear background to substract.
            
        set_Mn_Ka : bool
            If true, set the obtain resolution. If false, return the 
            FHWM at Mn Ka.
            
        model_plot : bool
            If True, plot the fit        
        
        """
        
        from hyperspy.hspy import create_model 
        mp = self.mapped_parameters        
        element, line = utils_eds._get_element_and_line(Xray_line)
        Xray_energy = elements_db[element]['Xray_energy'][line]
        FWHM = utils_eds.FWHM(mp.SEM.EDS.energy_resolution_MnKa,
            Xray_energy)
        if bck=='auto':
            spec_bck = self[Xray_energy+2.5*FWHM:Xray_energy+2.7*FWHM]
            bck = spec_bck.sum(0).data/spec_bck.axes_manager.shape[0]                
        sb = self - bck
        m = create_model(sb)
         

        fp = components.Gaussian()    
        fp.centre.value = Xray_energy
        fp.sigma.value = FWHM/2.355
        m.append(fp)
        m.set_signal_range(Xray_energy-1.2*FWHM,Xray_energy+1.6*FWHM)
        m.multifit()
        if model_plot:
            m.plot()

        res_MnKa = utils_eds.FWHM(fp.sigma.value*2.355*1000,
            elements_db['Mn']['Xray_energy']['Ka'],Xray_line)        
        if set_Mn_Ka:            
            mp.SEM.EDS.energy_resolution_MnKa = res_MnKa*1000
            print 'Resolution at Mn Ka ', res_MnKa*1000
            print 'Shift eng eV ', (Xray_energy-fp.centre.value)*1000
        else : 
            return res_MnKa*1000
            
    def get_result(self, Xray_line, result):
        """
        get the result of one X-ray line (result stored in 
        'mapped_parameters.Sample'):
        
         Parameters
        ----------        
        result : string {'kratios'|'quant'|'intensities'}
            The result to get
            
        Xray_lines: string
            the X-ray line to get.
        
        """
        mp = self.mapped_parameters 
        for res in mp.Sample[result]:
            if Xray_line in res.mapped_parameters.title:
                return res
        raise ValueError("Didn't find it")       

        
        
    def _set_result(self, Xray_line, result, data_res, plot_result,
        store_in_mp=True):
        """
        Transform data_res (a result) into an image or a signal and
        stored it in 'mapped_parameters.Sample'
        """
        
        mp = self.mapped_parameters
        if len(Xray_line) < 3 :
            Xray_lines = mp.Sample.elements
        else:
            Xray_lines = mp.Sample.Xray_lines
                
        for j in range(len(Xray_lines)):
            if Xray_line == Xray_lines[j]:
                break  
        
        axes_res = self.axes_manager.deepcopy()
        axes_res.remove(-1)
        
        if self.axes_manager.navigation_dimension == 0:
            res_img = Signal(np.array(data_res))
        else:
            res_img = Signal(data_res)
            res_img.axes_manager = axes_res
            if self.axes_manager.navigation_dimension > 1:
                res_img = res_img.as_image([0,1])
        res_img.mapped_parameters.title = result + ' ' + Xray_line
        if plot_result:                
            if self.axes_manager.navigation_dimension == 0:
                #to be changed with new version
                print("%s of %s : %s" % (result, Xray_line, data_res))
            else:
                res_img.plot(None)
        #else:
        #    print("%s of %s calculated" % (result, Xray_line))
            
        res_img.get_dimensions_from_data()
            
        if store_in_mp:
            mp.Sample[result][j] = res_img 
        else:
            return res_img 
            
    def normalize_result(self,result,return_element='all'):
        """
        Normalize the result
        
        The sum over all elements for any pixel is equal to one.
        
        Paramters
        ---------
        
        result: str
            the result to normalize
            
        return_element: str 
            If 'all', all elements are return.
        """
        #look at dim...
        mp = self.mapped_parameters
        res = copy.deepcopy(mp.Sample[result])
        
        re = utils.stack(res)
        if re.axes_manager.signal_dimension==0:
            tot = re.sum(1)
            for r in range(re.axes_manager.shape[1]):
                res[r].data = (re[::,r]/tot).data             
        elif re.axes_manager.signal_dimension==1:
            tot = re.sum(0)
            for r in range(re.axes_manager.shape[0]):
                res[r].data = (re[r]/tot).data 
        else:
            tot = re.sum(1)
            for r in range(re.axes_manager.shape[1]):
                res[r].data = (re[::,r]/tot).data  
        
        if return_element=='all':
            return res
        else:
            for el in res:
                if return_element in el.mapped_parameters.title:
                    return el
        
    def plot_histogram_result(self,
        result,
        bins=50,  
        colors='auto',
        line_styles='auto'):
        """
        Plot an histrogram of the result
        
        Paramters
        ---------
        
        result: str
            the result to plot
            
        bins: int
            the number of bins

        
        colors: list
            If 'auto', automatically selected, eg: ('red','blue')
        
        line_styles: list
            If 'auto', continuous lines, eg: ('-','--','steps','-.',':')
        """
        mp = self.mapped_parameters
        res = copy.deepcopy(mp.Sample[result])       
        
        utils_eds.compare_histograms(res,bins=bins,legend_labels='auto',
        colors=colors,line_styles=line_styles)

        
    def plot_orthoview_result(self,
        element,
        result,
        index,
        plot_index=False,
        space=2,
        plot_result=True,
        normalize=False):

        """
        Plot an orthogonal view of a 3D images
        
        Parameters
        ----------
        
        element: str
            The element to get.
        
        result: str
            The result to get  
            
        index: list
            The position [x,y,z] of the view.
            
        plot_index: bool
            Plot the line indicating the index position.
            
        space: int
            the spacing between the images in pixel.
        """
        if element=='all':
            res_element = copy.deepcopy(self.mapped_parameters.Sample[result])
            res_element = utils.stack(res_element).sum(1) 
        elif normalize:
            self.deepcopy()
            res_element = self.normalize_result(result,return_element=element)
        else:
            res_element = self.get_result(element,result)
        fig = utils_eds.plot_orthoview(res_element,
            index,plot_index,space,plot_result)

        return fig
        
    def add_poissonian_noise(self, **kwargs):
        """Add Poissonian noise to the data"""
        original_type = self.data.dtype
        self.data = np.random.poisson(self.data, **kwargs).astype(
                                      original_type)
        
    
        
        
        
            


