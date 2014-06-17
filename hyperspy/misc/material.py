import numpy as np

from hyperspy.misc.elements import elements as elements_db


def weight_to_atomic(elements, weight_percent):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    weight_percent: list of float
        The weight fractions (composition) of the sample.

    Returns
    -------
    atomic_percent : list
        Composition in atomic percent.

    """
    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight'] for element in elements])
    atomic_percent = weight_percent / atomic_weights / (
        weight_percent / atomic_weights).sum() * 100
    return atomic_percent.tolist()


def atomic_to_weight(elements, atomic_percent):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    atomic_percent: list of float
        The atomic fractions (composition) of the sample.

    Returns
    -------
    weight_percent : composition in weight percent.

    """

    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight'] for element in elements])

    weight_percent = atomic_percent * atomic_weights / (
        atomic_percent * atomic_weights).sum() * 100
    return weight_percent.tolist()


def density_of_mixture_of_pure_elements(elements, weight_percent):
    """Calculate the density a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    elements: list of str
        A list of element symbols, e.g. ['Al', 'Zn']
    weight_percent: list of float
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------

    Calculate the density of modern bronze given its weight percent:
    >>> utils.material.density_of_mixture_of_pure_elements(("Cu", "Sn"), (88, 12))
    8.6903187973131466

    """
    densities = np.array(
        [elements_db[element]['Physical_properties']['density (g/cm^3)'] for element in elements])
    density = (weight_percent / densities / sum(weight_percent)).sum() ** -1
    return density
    
def _mac_interpolation(mac, mac1, energy,
                       energy_db, energy_db1):
    """
    Interpolate between the tabulated mass absorption coefficients
    for an energy

    Parameters
    ----------
    mac, mac1: float
        The mass absorption coefficients in cm^2/g
    energy,energy_db,energy_db1:
        The energy. The given energy and the tabulated energy,
        respectively

    Return
    ------
    mass absorption coefficient in cm^2/g
    """
    return np.exp(np.log(mac1) + np.log(mac / mac1)
                  * (np.log(energy / energy_db1) / np.log(
                      energy_db / energy_db1)))
                                  
def get_mass_absorption_coefficient(element, energies):
    """
    Get the mass absorption coefficient of a given energy (energies)

    Parameters
    ----------
    element: str
        The element symbol of the absorber, e.g. 'Al'.
    energies: float or list of float
        The energy or energies of the Xray in keV

    Return
    ------
    mass absorption coefficient in cm^2/g
    """
    from hyperspy.misc.eds.ffast_mac import ffast_mac_db as ffast_mac
    energies_db = ffast_mac[element].energies_keV
    if hasattr(energies,'__iter__'):
        is_iter = True
    else:
        is_iter = False
        energies = [energies]
    mac_res=[]
    for energy in energies:
        for index, energy_db in enumerate(energies_db):
            if energy <= energy_db:
                break
        mac = ffast_mac[element].mass_absorption_coefficient_cm2g[index]
        mac1 = ffast_mac[element].mass_absorption_coefficient_cm2g[index - 1]
        energy_db = ffast_mac[element].energies_keV[index]
        energy_db1 = ffast_mac[element].energies_keV[index - 1] 
        if energy == energy_db or energy_db1 == 0:
            mac_res.append(mac)
        else:
            mac_res.append(_mac_interpolation(mac, mac1, energy,
                                      energy_db, energy_db1))
    if is_iter:
        return mac_res
    else:
        return mac_res[0]
