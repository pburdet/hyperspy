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
        h += wt * A / (Z * Z)

    coeff = 4.5e5 # keV^1.65


    xi = np.array(utils.material.compound_mass_absorption_coefficient(
        energies=energy, elements=elements,
        weight_fraction=weight_fraction)) / np.sin(np.radians(TOA))
    sig = coeff / (np.power(beam_energy, 1.65
                            ) - np.power(energy, 1.65))
    return 1 / ((1 + xi / sig) * (1 + h / (1 + h) * xi / sig))
