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
    if isinstance(weight_percent[0],float):
        densities = np.array(
            [elements_db[element]['Physical_properties']['density (g/cm^3)'] for element in elements])
        density = (weight_percent / densities / sum(weight_percent)).sum() ** -1
        return density
    else:
        weight_percent=np.array(weight_percent)
        densities = []
        for element, weight in zip(elements,weight_percent):
            density = elements_db[element]['Physical_properties']\
                ['density (g/cm^3)']
            densities.append(weight/density)
        densities=np.array(densities)
        return weight_percent.sum(0)/densities.sum(0)
        
#working for signals as well
#weight_fraction=utils.stack(weight_fraction)
#if isinstance(weight_fraction,list):
    #weight_fraction=np.array(weight_fraction)
#densities = []
#for element, weight in zip(elements,weight_fraction):
    #density = utils.material.elements\
         #[element]['Physical_properties']['density (g/cm^3)']
    #densities.append(weight/density)
#if isinstance(densities[0],list):
    #densities=np.array(densities)
#elif isinstance(densities[0],signals.Signal):
    #densities=utils.stack(densities)
#weight_fraction.sum(0)/densities.sum(0)
