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

# To do: weight_fraction different for different pixe. (so basckground)
# Calibrate on standard and transfer dictionnary
# k-ratios

import copy
import numpy as np
#import traits.api as t
import math

#from hyperspy.model import Model
from hyperspy.models.edsmodel import EDSModel
#from hyperspy._signals.eds import EDSSpectrum
#from hyperspy.misc.elements import elements as elements_db
#from hyperspy.misc.eds import utils as utils_eds
import hyperspy.components as create_component
#from hyperspy import utils

class EDSTEMModel(EDSModel):

    """Build a fit a model

    Parameters
    ----------
    spectrum : an Spectrum (or any Spectrum subclass) instance
    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically
        a powerlaw to the model and estimate the parameters by the
        two-area method.

    """

    def __init__(self, spectrum, auto_background=True,
                 auto_add_lines=True,
                 *args, **kwargs):
        EDSModel.__init__(self, spectrum, auto_add_lines, *args, **kwargs)

        self.background_components = list()        

        if auto_background is True:
            self.add_background()

    def add_background(self,
                       generation_factor=1,
                       detector_name='osiris',
                       weight_fraction='auto',
                       thicknesses=[100.],
                       density='auto',
                       gateway='auto'):
        """
        Add a backround to the model in the form of several
        scalable fixed patterns.

        Each pattern is the muliplication of the detector efficiency,
        the absorption in the sample (constant
        X-ray pdouction) and a continuous X-ray
        generation.

        Parameters
        ----------
        generation_factor: int
            For each number n, add (E0-E)^n/E
            1 is equivalent to Kramer equation.
        det_name: int, str, None
            If None, no det_efficiency
            If {0,1,2,3,4}, INCA efficiency database
            If str, model from DTSAII
        weight_fraction: list of float
             The sample composition used for the sample absorption.
             If 'auto', takes value in metadata. If not there,
             use and equ-composition
        thicknesses : list float
            Generated several model of byc in function of Thicknesses
             of thin film. 
        density: float or 'auto'
            Set the density. in g/cm^3
            if 'auto', calculated from weight_fraction
        gateway: execnet Gateway
            If 'auto', generate automatically the connection to jython.

        See also
        --------
        database.detector_efficiency_INCA,
        utils_eds.get_detector_properties
        """

        generation = self.spectrum.compute_continuous_xray_generation(
            generation_factor)                 
        absorptions = []
        
        background_names = [float(xr.name.split('_')[1]) for xr in self.background_components]
        thicknesses = filter(lambda x: x not in background_names, thicknesses)
        
        for thickness in thicknesses:
            if thickness == 0.:
                absorptions.append(generation.deepcopy())
                absorptions[-1].data = np.ones_like(generation.data)
            else :
                absorptions.append(self.spectrum.compute_continuous_xray_absorption(
                    thickness=thickness, density=density,
                    weight_fraction=weight_fraction))
        
        if detector_name is None:
            det_efficiency = generation.deepcopy()
            det_efficiency.data = np.ones_like(generation.data)
        elif detector_name == 'osiris':
            det_efficiency = self.spectrum.compute_detector_efficiency_from_layers(microscope_name=detector_name)
        else : 
            det_efficiency = self.spectrum.get_detector_efficiency(
                detector_name, gateway=gateway)

        for absorption, thickness in zip(absorptions, thicknesses):
            bck = det_efficiency * generation * absorption
            # bck.plot()
            bck = bck[self.axes_manager[-1].scale:]
            bck.metadata.General.title = 'bck_' + str(thickness) + '_nm'
            component = create_component.ScalableFixedPattern(bck)
            component.set_parameters_not_free(['xscale', 'shift'])
            component.name = bck.metadata.General.title
            #component.yscale.ext_bounded = True
            #component.yscale.bmin = 0
            component.yscale.ext_force_positive = True
            component.isbackground = True
            self.append(component)
            self.background_components.append(component)

   

    
