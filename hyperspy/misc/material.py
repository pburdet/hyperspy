import numpy as np
import numbers
import warnings

from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds.ffast_mac import ffast_mac_db as ffast_mac
from hyperspy.misc.utils import stack


def _weight_to_atomic(weight_percent, elements):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    weight_percent: array of float
        The weight fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    Returns
    -------
    atomic_percent : array of float
        Composition in atomic percent.

    Calculate the atomic percent of modern bronze given its weight percent:
    >>> utils.material.weight_to_atomic((88, 12), ("Cu", "Sn"))
    array([ 93.19698614,   6.80301386])

    """
    if len(elements) != len(weight_percent):
        raise ValueError(
            'The number of elements must match the size of the first axis'
            'of weight_percent.')
    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight']
            for element in elements])
    atomic_percent = np.array(map(np.divide, weight_percent, atomic_weights))
    sum_weight = atomic_percent.sum(axis=0)/100.
    for i, el in enumerate(elements):
        warnings.simplefilter("ignore")
        atomic_percent[i] /= sum_weight
        warnings.simplefilter('default')
        atomic_percent[i] = np.where(sum_weight == 0.0, 0.0, atomic_percent[i])
    return atomic_percent


def weight_to_atomic(weight_percent, elements='auto'):
    """Convert weight percent (wt%) to atomic percent (at.%).

    Parameters
    ----------
    weight_percent: list of float or list of signals
        The weight fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']. If elements is
        'auto', take the elements in en each signal metadata of th
        weight_percent list.

    Returns
    -------
    atomic_percent : as weight_percent
        Composition in atomic percent.

    Examples
    --------
    Calculate the atomic percent of modern bronze given its weight percent:
    >>> utils.material.weight_to_atomic((88, 12), ("Cu", "Sn"))
    array([ 93.19698614,   6.80301386])

    """
    elements = _elements_auto(weight_percent, elements)
    if isinstance(weight_percent[0], numbers.Number):
        return _weight_to_atomic(weight_percent, elements)
    else:
        atomic_percent = stack(weight_percent)
        atomic_percent.data = _weight_to_atomic(
            atomic_percent.data, elements)
        atomic_percent.data = np.nan_to_num(atomic_percent.data)
        atomic_percent = atomic_percent.split()
        return atomic_percent


def _atomic_to_weight(atomic_percent, elements):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    atomic_percent: array
        The atomic fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    Returns
    -------
    weight_percent : array of float
        composition in weight percent.

    Examples
    --------
    Calculate the weight percent of modern bronze given its atomic percent:
    >>> utils.material.atomic_to_weight([93.2, 6.8], ("Cu", "Sn"))
    array([ 88.00501989,  11.99498011])

    """
    if len(elements) != len(atomic_percent):
        raise ValueError(
            'The number of elements must match the size of the first axis'
            'of atomic_percent.')
    atomic_weights = np.array(
        [elements_db[element]['General_properties']['atomic_weight']
            for element in elements])
    weight_percent = np.array(map(np.multiply, atomic_percent, atomic_weights))
    sum_atomic = weight_percent.sum(axis=0)/100.
    for i, el in enumerate(elements):
        warnings.simplefilter("ignore")
        weight_percent[i] /= sum_atomic
        warnings.simplefilter('default')
        weight_percent[i] = np.where(sum_atomic == 0.0, 0.0, weight_percent[i])
    return weight_percent


def atomic_to_weight(atomic_percent, elements='auto'):
    """Convert atomic percent to weight percent.

    Parameters
    ----------
    atomic_percent: list of float or list of signals
        The atomic fractions (composition) of the sample.
    elements: list of str
        A list of element abbreviations, e.g. ['Al','Zn']. If elements is
        'auto', take the elements in en each signal metadata of the
        atomic_percent list.

    Returns
    -------
    weight_percent : as atomic_percent
        composition in weight percent.

    Examples
    --------
    Calculate the weight percent of modern bronze given its atomic percent:
    >>> utils.material.atomic_to_weight([93.2, 6.8], ("Cu", "Sn"))
    array([ 88.00501989,  11.99498011])

    """
    elements = _elements_auto(atomic_percent, elements)
    if isinstance(atomic_percent[0], numbers.Number):
        return _atomic_to_weight(atomic_percent, elements)
    else:
        weight_percent = stack(atomic_percent)
        weight_percent.data = _atomic_to_weight(
            weight_percent.data, elements)
        weight_percent = weight_percent.split()
        return weight_percent


def _density_of_mixture_of_pure_elements(weight_percent, elements):
    """Calculate the density a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    weight_percent: array
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).
    elements: list of str
        A list of element symbols, e.g. ['Al', 'Zn']

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------
    Calculate the density of modern bronze given its weight percent:
    >>> utils.material.density_of_mixture_of_pure_elements(
            (88, 12),("Cu", "Sn"))
    8.6903187973131466

    """
    if len(elements) != len(weight_percent):
        raise ValueError(
            'The number of elements must match the size of the first axis'
            'of weight_percent.')
    densities = np.array(
        [elements_db[element]['Physical_properties']['density (g/cm^3)']
            for element in elements])
    sum_densities = np.zeros_like(weight_percent, dtype='float')
    for i, weight in enumerate(weight_percent):
        sum_densities[i] = weight / densities[i]
    sum_densities = sum_densities.sum(axis=0)
    warnings.simplefilter("ignore")
    density = np.sum(weight_percent, axis=0) / sum_densities
    warnings.simplefilter('default')
    return np.where(sum_densities == 0.0, 0.0, density)


def density_of_mixture_of_pure_elements(weight_percent, elements='auto'):
    """Calculate the density of a mixture of elements.

    The density of the elements is retrieved from an internal database. The
    calculation is only valid if there is no interaction between the
    components.

    Parameters
    ----------
    weight_percent: list of float or list of signals
        A list of weight percent for the different elements. If the total
        is not equal to 100, each weight percent is divided by the sum
        of the list (normalization).
    elements: list of str
        A list of element symbols, e.g. ['Al', 'Zn']. If elements is 'auto',
        take the elements in en each signal metadata of the weight_percent
        list.

    Returns
    -------
    density: The density in g/cm3.

    Examples
    --------
    Calculate the density of modern bronze given its weight percent:
    >>> utils.material.density_of_mixture_of_pure_elements(
            (88, 12),("Cu", "Sn"))
    8.6903187973131466

    """
    elements = _elements_auto(weight_percent, elements)
    if isinstance(weight_percent[0], numbers.Number):
        return _density_of_mixture_of_pure_elements(weight_percent, elements)
    else:
        density = weight_percent[0].deepcopy()
        density.data = _density_of_mixture_of_pure_elements(
            stack(weight_percent).data, elements)
        return density

# working for signals as well
# weight_fraction=utils.stack(weight_fraction)
# if isinstance(weight_fraction,list):
    # weight_fraction=np.array(weight_fraction)
#densities = []
# for element, weight in zip(elements,weight_fraction):
    #density = utils.material.elements\
        #[element]['Physical_properties']['density (g/cm^3)']
    # densities.append(weight/density)
# if isinstance(densities[0],list):
    # densities=np.array(densities)
# elif isinstance(densities[0],signals.Signal):
    # densities=utils.stack(densities)
# weight_fraction.sum(0)/densities.sum(0)
# old version
# densities = np.array(
    #[elements_db[element]['Physical_properties']['density (g/cm^3)'] for element in elements])
#density = (weight_percent / densities / sum(weight_percent)).sum() ** -1
# return density


# def _mac_interpolation(mac, mac1, energy,
        # energy_db, energy_db1):
    #"""
    # Interpolate between the tabulated mass absorption coefficients
    # for an energy

    # Parameters
    #----------
    # mac, mac1: float
        # The mass absorption coefficients in cm^2/g
    # energy,energy_db,energy_db1:
        # The energy. The given energy and the tabulated energy,
        # respectively

    # Return
    #------
    # mass absorption coefficient in cm^2/g
    #"""
    # return np.exp(np.log(mac1) + np.log(mac / mac1)
        #* (np.log(energy / energy_db1) / np.log(
        # energy_db / energy_db1)))


def mass_absorption_coefficient(element, energies):
    """
    Get the mass absorption coefficient of a X-ray(s)

    In a pure material for a Xray(s) of given energy(ies) or given name(s)

    Parameters
    ----------
    element: str
        The element symbol of the absorber, e.g. 'Al'.
    energies: {float or list of float or str or list of str}
        The energy or energies of the Xray in keV, or the name eg 'Al_Ka'

    Return
    ------
    mass absorption coefficient(s) in cm^2/g
    """

    energies_db = np.array(ffast_mac[element].energies_keV)
    macs = np.array(ffast_mac[element].mass_absorption_coefficient_cm2g)
    if isinstance(energies, str):
        energies = utils_eds._get_energy_xray_line(energies)
    elif hasattr(energies, '__iter__'):
        if isinstance(energies[0], str):
            for i, energy in enumerate(energies):
                energies[i] = utils_eds._get_energy_xray_line(energy)

    index = np.searchsorted(energies_db, energies)

    mac_res = np.exp(np.log(macs[index - 1])
                     + np.log(macs[index] / macs[index - 1])
                     * (np.log(energies / energies_db[index - 1])
                     / np.log(energies_db[index] / energies_db[index - 1])))
    return np.nan_to_num(mac_res)


def mass_absorption_coefficient_of_mixture_of_pure_elements(elements,
                                                            weight_fraction,
                                                            energies):
    """Calculate the mass absorption coefficients a mixture of elements.

    A compund is a mixture of pure elements

    Parameters
    ----------
    elements: list of str
        The list of element symbol of the absorber, e.g. ['Al','Zn'].
    weight_fraction: np.array
        dim = {el,z,y,x} The fraction of elements in the sample by weight
    energies: {float or list of float or str or list of str}
        The energy or energies of the Xray in keV, or the name eg 'Al_Ka'

    Examples
    --------
    >>> utils.material.mass_absorption_coefficient_of_mixture_of_pure_elements(
    >>>     ['Al','Zn'],[0.5,0.5],'Al_Ka')

    Return
    ------
    mass absorption coefficient(s) in cm^2/g

    See also
    --------
    utils.material.mass_absorption_coefficient
    """
    if len(elements) != len(weight_fraction):
        raise ValueError(
            "Elements and weight_fraction should have the same lenght")

    if hasattr(weight_fraction[0], '__iter__'):

        weight_fraction = np.array(weight_fraction)
        # mac_res = 0
        mac_res = np.zeros(weight_fraction.shape[1:])
        #mac_res = np.zeros_like(energies,dtype=float)
        # mac_re = np.array([mass_absorption_coefficient(
        #    el, energies) for el in elements])
        for element, weight in zip(elements, weight_fraction):
            mac_re = mass_absorption_coefficient(
                element, energies)
        # for weight in weight_fraction:
            mac_res += mac_re * weight
            #mac_res.append(np.dot(weight, mac_re))
        return mac_res
    else:
        mac_res = np.array([mass_absorption_coefficient(
            el, energies) for el in elements])
        mac_res = np.dot(weight_fraction, mac_res)
        return mac_res

    # if hasattr(energies, '__iter__'):
        #is_iter = True
    # else:
        #is_iter = False
        #energies = [energies]

    #mac_res = []
    # for i, energy in enumerate(energies):
        # if isinstance(weight_fraction[0], float):
        # mac_res.append(0)
        # else:
        # mac_res.append(np.zeros_like(weight_fraction))
        # for el, weight in zip(elements, weight_fraction):
        # mac_res[i] += weight * np.array(mass_absorption_coefficient(
        # el, energy))
    # if is_iter:
        # return mac_res
    # else:
        # return mac_res[0]

def _elements_auto(composition, elements):
    if isinstance(composition[0], numbers.Number):
        if isinstance(elements,str):
            raise ValueError("The elements needs to be provided.")
    else:
        if isinstance(elements,str):
            elements = []
            for compo in composition:
                if len(compo.metadata.Sample.elements) > 1:
                    raise ValueError(
                        "The signal %s contains more than one "
                        "element but this function requires only one element "
                        "per signal." % compo.metadata.General.title)
                else:
                    elements.append(compo.metadata.Sample.elements[0])
    return elements
