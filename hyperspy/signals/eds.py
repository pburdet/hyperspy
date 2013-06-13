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
import matplotlib.mlab as mlab

from hyperspy.signals.spectrum import Spectrum
from hyperspy.signals.image import Image
from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.eds.FWHM import FWHM_eds
from hyperspy.misc import utils


class EDSSpectrum(Spectrum):
    
    def __init__(self, *args, **kwards):
        Spectrum.__init__(self, *args, **kwards)
        # Attributes defaults
        if hasattr(self,'elements')==False:
            self.elements = set()
        if hasattr(self,'Xray_lines')==False:
            self.Xray_lines = set()
            
    def sum(self,axis):
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

        Usage
        -----
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
        return super(EDSSpectrum, self).sum( axis)
        
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
        #modify time per spectrum
        if hasattr(self.mapped_parameters, 'SEM'):
            mp = self.mapped_parameters.SEM
        else:
            mp = self.mapped_parameters.TEM
        if hasattr(mp, 'EDS') and hasattr(mp.EDS, 'live_time'):
            for factor in factors:
                mp.EDS.live_time = mp.EDS.live_time * factor
        Spectrum.rebin(self, new_shape)
    
    def set_elements(self, elements, lines=None):
        """Erase all elements and set them with the corresponding
        X-ray lines.
        
        The X-ray lines can be choosed manually or automatically.
        
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
        lines : list of strings
            One X-ray line for each element ('Ka', 'La', 'Ma',...). If None 
            the set of highest ionized lines with sufficient intensity 
            is selected. The beam energy is needed.
            
        See also
        --------
        add_elements, 
            
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
        self.elements = set()
        self.Xray_lines = set()
        if hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.Sample.elements = []
            self.mapped_parameters.Sample.Xray_lines = []
                
        self.add_elements(elements, lines)
           
        
        
    def add_elements(self, elements, lines=None):
        """Add elements and the corresponding X-ray lines.
        
        The X-ray lines can be choosed manually or automatically.        
        
        Parameters
        ----------
        elements : list of strings
            The symbol of the elements.  
        
        lines : list of strings
            One X-ray line for each element ('Ka', 'La', 'Ma',...). If none 
            the set of highest ionized lines with sufficient intensity 
            is selected. The beam energy is needed. All available lines
            are return for a wrong lines.
            
        See also
        --------
        set_elements, 
        
        """
        
        for element in elements:            
            if element in elements_db:               
                self.elements.add(element)
            else:
                print(
                    "%s is not a valid symbol of an element" % element)
                    
        if not hasattr(self.mapped_parameters, 'Sample'):
            self.mapped_parameters.add_node('Sample')
        self.mapped_parameters.Sample.elements = np.sort(list(self.elements))
        
               
        #Set X-ray lines
        if lines is None:
            self._add_lines_auto(elements)
        else:
           self._add_lines(elements,lines)                
        self.mapped_parameters.Sample.Xray_lines = np.sort(list(self.Xray_lines))
        
    
        
    def _add_lines(self,elements,lines):
        
        end_energy = self.axes_manager.signal_axes[0].axis[-1]            
            
        i = 0
        for line in lines:
            element = elements[i]
            if element in elements_db: 
                if line in elements_db[element]['Xray_energy']:
                    print("Adding %s_%s line" % (element,line))
                    self.Xray_lines.add(element+'_'+line)                                                   
                    if elements_db[element]['Xray_energy'][line] >\
                    end_energy:
                      print("Warning: %s %s is higher than signal range." 
                      % (element,line))  
                else:
                    print("%s is not a valid line of %s." % (line,element))
                    print("Valid lines for %s are (importance):" % element)
                    for li in elements_db[element]['Xray_energy']:
                        print("%s (%s)" % (li,
                         elements_db['lines']['ratio_line'][li]))
            else:
                print(
                    "%s is not a valid symbol of an element." % element)
            i += 1
        
            
    def _add_lines_auto(self,elements):
        """Choose the highest set of X-ray lines for the elements 
        present in self.elements  
        
        Possible line are in the current energy range and below the beam 
        energy. The highest line above an overvoltage of 2 
        (< beam energy / 2) is prefered.
            
        """
        if not hasattr(self.mapped_parameters.SEM,'beam_energy'):
            raise ValueError("Beam energy is needed in "
            "mapped_parameters.SEM.beam_energy")
        
        end_energy = self.axes_manager.signal_axes[0].axis[-1]
        beam_energy = self.mapped_parameters.SEM.beam_energy
        if beam_energy < end_energy:
           end_energy = beam_energy
           
        true_line = []
        
        for element in elements: 
                        
            #Possible line (existing and excited by electron)         
            for line in ('Ka','La','Ma'):
                if line in elements_db[element]['Xray_energy']:
                    if elements_db[element]['Xray_energy'][line] < \
                    end_energy:
                        true_line.append(line)
                        
            #Choose the better line
            i = 0
            select_this = -1            
            for line in true_line:
                if elements_db[element]['Xray_energy'][line] < \
                beam_energy/2:
                    select_this = i
                    break
                i += 1           
                     
            if true_line == []:
                print("No possible line for %s" % element)
            else:       
                self.Xray_lines.add(element+'_'+true_line[select_this])
                print("Adding %s_%s line" % (element,true_line[select_this]))
                
         
    
    def get_intensity_map(self, Xray_lines = 'auto',plot_result=True,
        width_energy_reso=1):
        """Return the intensity map of selected Xray lines.
        
        The intensity is the sum over several energy channels. The width
        of the sum is determined using the energy resolution of the detector
        defined with the FWHM of Mn Ka in
        self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        (by default 130eV). 
        
        
        Parameters
        ----------
        
        Xray_lines: list of string | 'auto'
            If 'auto' (default option), the lines defined with set_elements are used, which 
            are in 'mapped.parameters.Sample.X_ray_lines'. 
        
        width_energy_reso: Float
            factor to change the width used for the sum. 1 is equivalent
            of a width of 2 X FWHM 
            
        Examples
        --------
        
        >>> specImg.plot_intensity_map(["C_Ka", "Ta_Ma"])
        
        See also
        --------
        
        deconvolve_intensity
        
        """
        
                
        if Xray_lines == 'auto':
            if hasattr(self.mapped_parameters, 'Sample') and \
            hasattr(self.mapped_parameters.Sample, 'Xray_lines'):
                Xray_lines = self.mapped_parameters.Sample.Xray_lines
            else:
                raise ValueError("Not X-ray line, set them with add_elements")            
        
        if self.mapped_parameters.signal_type == 'EDS_SEM':
            FWHM_MnKa = self.mapped_parameters.SEM.EDS.energy_resolution_MnKa
        else:
            FWHM_MnKa = self.mapped_parameters.TEM.EDS.energy_resolution_MnKa
                        
        intensities = []
        #test 1D Spectrum (0D problem)
        if self.axes_manager.navigation_dimension > 1:
            signal_to_index = self.axes_manager.navigation_dimension - 2                  
            for Xray_line in Xray_lines:
                element, line = utils._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                img = self.to_image(signal_to_index)
                img.mapped_parameters.title = 'Intensity of ' + Xray_line +\
                ' at ' + str(line_energy) + ' keV'
                det = width_energy_reso*line_FWHM
                if plot_result:
                    img[line_energy-det:line_energy+det].sum(0).plot(None)
                intensities.append(img[line_energy-det:line_energy+det].sum(0))
        else:
            for Xray_line in Xray_lines:
                element, line = utils._get_element_and_line(Xray_line)           
                line_energy = elements_db[element]['Xray_energy'][line]
                line_FWHM = FWHM_eds(FWHM_MnKa,line_energy)
                det = width_energy_reso*line_FWHM
                if plot_result:
                    print("%s at %s keV : Intensity = %s" 
                    % (Xray_line, line_energy,\
                     self[line_energy-det:line_energy+det].sum(0).data) )
                intensities.append(self[line_energy-det:line_energy+det].sum(0).data)
        return intensities
        
    
        
        
        


    def running_sum(self) :
        """
        Apply a running sum on the data.
        
        """
        dim = self.data.shape
        data_s = np.zeros_like(self.data)        
        data_s = np.insert(data_s, 0, 0,axis=-3)
        data_s = np.insert(data_s, 0, 0,axis=-2)
        end_mirrors = [[0,0],[-1,0],[0,-1],[-1,-1]]
        
        for end_mirror in end_mirrors:  
            tmp_s=np.insert(self.data, end_mirror[0], self.data[...,end_mirror[0],:,:],axis=-3)
            data_s += np.insert(tmp_s, end_mirror[1], tmp_s[...,end_mirror[1],:],axis=-2)
        data_s = data_s[...,1::,:,:][...,1::,:]
        
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
        
        line: string 'selected'|'a'|'ab|'all'
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
                element, line = utils._get_element_and_line(Xray_line)
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
            
def phase_inspector(self,bins=[20,20,20],plot_result=True):
    """
    Generate an binary image of different channel
    """
    bins=[20,20,20]
    minmax = []
    
    #generate the bins
    for s in self:    
        minmax.append([s.data.min(),s.data.max()])
    center = []
    for i, mm in enumerate(minmax):
        temp = list(mlab.frange(mm[0],mm[1],(mm[1]-mm[0])/bins[i]))
        temp[-1]+= 1
        center.append(temp)
        
    #calculate the Binary images
    dataBin = []
    if len(self) ==1:
        for x in range(bins[0]):
            temp = self[0].deepcopy()
            dataBin.append(temp)
            dataBin[x].data = ((temp.data >= center[0][x])*
              (temp.data < center[0][x+1])).astype('int')
    elif len(self) == 2 :    
        for x in range(bins[0]):
            dataBin.append([])
            for y in range(bins[1]):
                temp = self[0].deepcopy()
                temp.data = np.ones_like(temp.data)
                dataBin[-1].append(temp)
                a = [x,y]
                for i, s in enumerate(self):
                    dataBin[x][y].data *= ((s.data >= center[i][a[i]])*
                     (s.data < center[i][a[i]+1])).astype('int')
            dataBin[x] = utils.stack(dataBin[x])
    elif len(self) == 3 :    
        for x in range(bins[0]):
            dataBin.append([])
            for y in range(bins[1]):
                dataBin[x].append([])                    
                for z in range(bins[2]):
                    temp = self[0].deepcopy()
                    temp.data = np.ones_like(temp.data)
                    dataBin[-1][-1].append(temp)
                    a = [x,y,z]
                    for i, s in enumerate(self):
                        dataBin[x][y][z].data *= ((s.data >=
                         center[i][a[i]])*(s.data < 
                         center[i][a[i]+1])).astype('int')
                dataBin[x][y] = utils.stack(dataBin[x][y])
            dataBin[x] = utils.stack(dataBin[x])
    img = utils.stack(dataBin)

    for i in range(len(self)):
        img.axes_manager[i].name = self[i].mapped_parameters.title
        img.axes_manager[i].scale = (minmax[i][1]-minmax[i][0])/bins[i]
        img.axes_manager[i].offest = minmax[i][0]
        img.axes_manager[i].units = '-'
    img.get_dimensions_from_data()
    return img 
                 
   
    
        
    
            
    
       
       

    
