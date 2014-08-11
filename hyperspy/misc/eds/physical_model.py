import numpy as np
import math
import matplotlib.mlab as mlab


def xray_generation(energy,
                               generation_factor,
                               beam_energy):
    """X-ray generation.

    Kramer or Lisfshin equation

    Parameters
    ----------
    energy: float or list of float
        The energy of the generated X-ray.
    generation_factor: int
        The power law to use.
        1 si equivalent to Kramer equation.
        2 is equivalent to Lisfhisn modification of Kramer equation.
    beam_energy:  float
        The energy of the electron beam
    """
    if isinstance(energy, list):
        energy = np.array(energy)

    return 1 / energy * np.power((
        beam_energy - energy), generation_factor)


def xray_absorption_bulk(energy,
                               weight_fraction,
                               elements,
                               beam_energy,
                               TOA):
    """X-ray Absorption within bulk sample

    PDH equation (Philibert-Duncumb-Heinrich)

    Parameters
    ----------
    energy: float or list of float
        The energy of the generated X-ray in keV.
    weight_fraction: list of float
        The sample composition
    elements: list of str
        The elements of the sample
    TOA:
        the take off angle  in degree
    beam_energy:  float
        The energy of the electron beam in keV.
    """
    from hyperspy import utils
    h = 0
    for el, wt in zip(elements, weight_fraction):
        A = utils.material.elements[el]['General_properties']['atomic_weight']
        Z = utils.material.elements[el]['General_properties']['Z']
        h += wt * 1.2 * A / np.power(Z, 2)

    coeff = 4.5e5 # keV^1.65

    xi = np.array(utils.material.compound_mass_absorption_coefficient(
        energies=energy, elements=elements,
        weight_fraction=weight_fraction)) / np.sin(np.radians(TOA))
    sig = coeff / (np.power(beam_energy, 1.65
                            ) - np.power(energy, 1.65))
    return 1. / ((1. + xi / sig) * (1. + h / (1. + h) * ( xi / sig) ))

# def absorption_Yakowitz(self, E):
    #"""Absorption within sample
    #"""
    #beam_energy = self.metadata.Acquisition_instrument.SEM.beam_energy
    #weight_percent = self.metadata.Sample.weight_percent
    #TOA = self.get_take_off_angle()
    #elements = self.metadata.Sample.elements
    # a1 = 2.4 * 1e-6  # gcm-2keV-1.65
    # a2 = 1.44 * 1e-12  # g2cm-4keV-3.3
    # xc = epq_database.get_mass_absorption_coefficient(
        # E, elements, weight_percent, gateway=gateway)[0] / np.sin(np.radians(TOA))
    # return 1 / (1 + a1 * (np.power(beam_energy, 1.65) - np.power(E, 1.65)) * xc +
                # a2 * np.power(np.power(beam_energy, 1.65) - np.power(E,
                # 1.65), 2) * xc * xc)
    
def xray_absorption_thin_film(energy,
                                   weight_fraction,
                                   elements,
                                    density,
                                    thickness,
                                    TOA):
    """X-ray absorption in thin film sample
    
    Depth distribution of X-ray production is assumed constant
    
    Parameters
    ---------- 
    energy: float or list of float
        The energy of the generated X-ray in keV.
    weight_fraction: list of float
        The sample composition
    elements: list of str
        The elements of the sample
    density: float
        Set the density. in g/cm^3
    thickness: float
        Set the thickness in nm
    TOA: float
        the take of angle in degree
    """
    from hyperspy import utils
    mac_sample = np.array(utils.material.compound_mass_absorption_coefficient(
        energies=energy, elements=elements,
        weight_fraction=weight_fraction))
    rt =  density * thickness * 1e-7 / np.sin(np.radians(TOA))
    fact = mac_sample*rt
    abs_corr= np.nan_to_num((1-np.exp(-(fact)))/fact)
    return abs_corr
