import numpy as np
import math
import matplotlib.mlab as mlab

from hyperspy import utils


def continuous_xray_generation(energy,
                               generation_factor,
                               beam_energy):
    """Continous X-ray generation.

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


def continuous_xray_absorption(energy,
                               weight_fraction,
                               elements,
                               beam_energy,
                               TOA):
    """Contninous X-ray Absorption within sample

    PDH equation (Philibert-Duncumb-Heinrich)

    Parameters
    ----------
    energy: float or list of float
        The energy of the generated X-ray in keV. 
    weight_percent: list of float
        The sample composition
    elements: list of str
        The elements of the sample
    TOA:
        the take off angle
    beam_energy:  float
        The energy of the electron beam in keV.
    """
    h = 0
    for el, wt in zip(elements, weight_fraction):
        A = utils.material.elements[el]['General_properties']['atomic_weight']
        Z = utils.material.elements[el]['General_properties']['Z']
        h += wt * 1.2 * A / (Z * Z)

    coeff = 4.5e5 # keV^1.65

    xi = np.array(utils.material.compound_mass_absorption_coefficient(
        energies=energy, elements=elements,
        weight_fraction=weight_fraction)) / np.sin(np.radians(TOA))
    sig = coeff / (np.power(beam_energy, 1.65
                            ) - np.power(energy, 1.65))
    return 1 / ((1 + xi / sig) * (1 + h / (1 + h) * xi / sig))

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
