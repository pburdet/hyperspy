import numpy as np
import math
import matplotlib.mlab as mlab

from hyperspy.misc.eds import epq_database
from hyperspy.misc.elements import elements as elements_db

def bck_generation_kramer(self,E,a=1):
    """Xray generation bremstrahlung
    """
    beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
    return a/np.power(E,1)*(beam_energy-E)
    
def bck_generation_lifshin(self,E,a=1,b=1):
    """Xray generation bremstrahlung
    """
    beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
    return (a/E*(beam_energy-E)+b/E*(beam_energy-E)*(beam_energy-E))
    
def bck_generation_lifshin_exp(self,E,exp_factor,a=1):
    """Xray generation bremstrahlung
    """
    beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
    return a/E*np.power((beam_energy-E),exp_factor)
    
#should accepth spectrum or list
def absorption_PHD(self, E,weight_percent='auto',gateway='auto'):
    """Absorption within sample
    """
    if E < 0:
        return 0
    else:
        units_name = self.axes_manager.signal_axes[0].units
        if gateway == 'auto':
            gateway = utils_eds.get_link_to_jython()
        beam_energy = self._get_beam_energy()
        elements = self.metadata.Sample.elements
        if weight_percent =='auto': 
            if  'weight_percent' in  self.metadata.Sample:
                weight_percent = self.metadata.Sample.weight_percent
            else :
                weight_percent = []
                for elm in elements:
                    weight_percent.append(1. / len(elements))
        TOA = self.get_take_off_angle()    
        xi = epq_database.get_mass_absorption_coefficient(
            E,elements, weight_percent,gateway=gateway
            )[0] / np.sin(np.radians(TOA))

        if units_name == 'eV':
            coeff = 4.5*1e2
        else :
            coeff = 4.5*1e5
        sig = coeff / (np.power(beam_energy,1.65)-np.power(E,1.65))
        h = 0
        for el,wt in zip(elements,weight_percent):
            A = elements_db[el]['General_properties']['atomic_weight']
            Z = elements_db[el]['General_properties']['Z']
            h += wt*A/(Z*Z)
        return 1 / ((1+xi/sig)*(1+h/(1+h)*xi/sig))
    
def absorption_Yakowitz(self,E):
    """Absorption within sample
    """
    beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
    weight_percent = self.metadata.Sample.weight_percent
    TOA = self.get_take_off_angle()
    elements = self.metadata.Sample.elements
    a1=2.4*1e-6 # gcm-2keV-1.65
    a2=1.44*1e-12 # g2cm-4keV-3.3
    xc = epq_database.get_mass_absorption_coefficient(
        E,elements,weight_percent,gateway=gateway)[0] / np.sin(np.radians(TOA))
    return 1/(1+a1*(np.power(beam_energy,1.65)-np.power(E,1.65))*xc+
          a2*np.power(np.power(beam_energy,1.65)-np.power(E,1.65),2)*xc*xc)
          
def function_to_spectrum(self,function,**kwargs):
    spec = self.deepcopy()
    eng = mlab.frange(spec.axes_manager[0].low_value,
       spec.axes_manager[0].high_value,
       spec.axes_manager[0].scale)
    spec.data = np.array([function(self,en,**kwargs) for en in eng])
    return spec
