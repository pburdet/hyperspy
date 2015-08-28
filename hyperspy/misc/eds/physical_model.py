import numpy as np
from scipy import ndimage


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

    coeff = 4.5e5  # keV^1.65

    xi = utils.material.mass_absorption_coefficient_of_mixture_of_pure_elements(
        energies=energy, elements=elements,
        weight_percent=weight_fraction) / np.sin(np.radians(TOA))
    sig = coeff / (np.power(beam_energy, 1.65
                            ) - np.power(energy, 1.65))
    return 1. / ((1. + xi / sig) * (1. + h / (1. + h) * (xi / sig)))

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
    mac_sample = utils.material.\
        mass_absorption_coefficient_of_mixture_of_pure_elements(
            energies=energy, elements=elements,
            weight_percent=weight_fraction)
    rt = density * thickness * 1e-7 / np.sin(np.radians(TOA))
    fact = mac_sample * rt
    abs_corr = np.nan_to_num((1 - np.exp(-(fact))) / fact)
    return abs_corr


def detetector_efficiency_from_layers(energies,
                                      elements,
                                      thicknesses_layer,
                                      thickness_detector):
    """Detector efficiency from layers

    Parameters
    ----------
    energy: float or list of float
        The energy of the generated X-ray in keV.
    elements: list of str
        The elements of the layer
    thicknesses_layer: list of float
        Thicknesses of layer in nm
    thickness_detector: float
        The thickness of the detector in mm

    Notes
    -----
    Equation adapted from  Alvisi et al 2006
    """
    from hyperspy import utils
    absorption = np.ones_like(energies)

    for element, thickness in zip(elements,
                                  thicknesses_layer):
        macs = np.array(utils.material.mass_absorption_coefficient(
            energies=energies,
            element=element))
        density = utils.material.elements[element]\
            .Physical_properties.density_gcm3
        absorption *= np.nan_to_num(np.exp(-(
            macs * density * thickness * 1e-7)))
    macs = np.array(utils.material.mass_absorption_coefficient(
        energies=energies,
        element='Si'))
    density = utils.material.elements.Si\
        .Physical_properties.density_gcm3
    absorption *= (1 - np.nan_to_num(np.exp(-(macs * density *
                                              thickness_detector * 1e-1))))

    return absorption


def absorption_correction_matrix(weight_fraction,
                                 xray_lines,
                                 elements,
                                 thickness,
                                 density,
                                 azimuth_angle,
                                 elevation_angle,
                                 mask_el,
                                 tilt=0.):
    """
    Matrix of absorption for an isotropic 3D data cube of composition

    Parameters
    ----------
    weight_fraction: np.array
        dim = {el,z,y,x} The sample composition
    xray_lines: list of str
         The X-ray lines eg ['Al_Ka']
    elements: list of str
        The elements of the sample
    thickness: float
            Set the thickness in cm
    density: array
        dim = {z,y,x} The density to correct of the sample.
        If 'auto' use the weight_fraction
        to calculate it. in gm/cm^3
    azimuth_angle: float
        the azimuth_angle in degree
    elevation_angle: float
        the elevation_angle in degree
    mask: bool array
        A mask to be applied to the correction absorption
    tilt: float
        The tilt of the sample.

    Return
    ------
    The absorption matrix: np.array
        {xray_lines,z,y,x}
    """
    from hyperspy import utils
    from hyperspy.misc import material

    x_ax, y_ax, z_ax = 3, 2, 1
    order = 3
    # reflect is not really good to deal with border in z direction
    weight_fraction_r = ndimage.rotate(weight_fraction,
                                       angle=-azimuth_angle,
                                       axes=(x_ax, y_ax),
                                       order=order, mode='reflect')
    weight_fraction_r = ndimage.rotate(weight_fraction_r,
                                       angle=-elevation_angle-tilt,
                                       axes=(x_ax, z_ax), order=order,
                                       mode='reflect')

    elements = np.array(elements)
    if density == 'auto':
        density_r = material._density_of_mixture_of_pure_elements(
            weight_fraction_r * 100., elements)
    else:
        density_r = ndimage.rotate(density,
                                   angle=-azimuth_angle,
                                   axes=(x_ax - 1, y_ax - 1),
                                   order=order, mode='nearest')
        density_r = ndimage.rotate(density_r,
                                   angle=-elevation_angle-tilt,
                                   axes=(x_ax - 1, z_ax - 1),
                                   order=order, mode='nearest')
    if mask_el is None:
        mask_el_r = [1.] * len(xray_lines)
    else:
        mask_el_r = ndimage.rotate(mask_el,
                                   angle=-azimuth_angle,
                                   axes=(x_ax, y_ax),
                                   order=0, mode='reflect')
        mask_el_r = ndimage.rotate(mask_el_r,
                                   angle=-elevation_angle-tilt,
                                   axes=(x_ax, z_ax),
                                   order=0, mode='reflect')

    # abs_corr = np.zeros_like(weight_fraction_r)
    abs_corr = np.zeros([len(xray_lines)]+list(weight_fraction_r.shape[1:]))
    for i, xray_line in enumerate(xray_lines):
        mac = utils.material.\
            mass_absorption_coefficient_of_mixture_of_pure_elements(
                elements=elements, weight_percent=weight_fraction_r,
                energies=[xray_line])[0]
        fact = np.nan_to_num(density_r * mac * thickness * mask_el_r[i])

        fact_sum = np.zeros_like(fact)
        fact_sum[:, :, -1] = fact[:, :, -1] / 2.  # approx
        for j in range(len(fact[0, 0]) - 2, -1, -1):
            fact_sum[:, :, j] = fact_sum[:, :, j+1] + fact[:, :, j]
        abs_co = np.exp(-(fact_sum))
        abs_corr[i] = abs_co
#    interv = (abs_co.max() - abs_co.min())
#    nb_same_pix, max_corr, atol = (6, 0.9, 0.01)
#    index_same = [None] + range(1, nb_same_pix)
#    mask = abs_co[:, :, : -nb_same_pix + 1] < interv*max_corr + abs_co.min()
#    for i in range(nb_same_pix - 1):
#        mask = np.bitwise_and(mask,
#                              abs(abs_co[:, :, nb_same_pix-1:] -
#                                  abs_co[:, :, index_same[i]:
#                                         -index_same[nb_same_pix-i-1]])
#                              < interv*atol)
#    for i in range(nb_same_pix - 1):
#        mask = np.insert(mask, -1, False, axis=2)
#    for i, xray_line in enumerate(xray_lines):
#        # abs_corr[i] *= mask
#        np.place(abs_corr[i], mask, 1.0)
    abs_corr = ndimage.rotate(abs_corr, angle=elevation_angle,
                              axes=(x_ax, z_ax), reshape=False, order=0)
    abs_corr = ndimage.rotate(abs_corr, angle=azimuth_angle,
                              axes=(x_ax, y_ax), reshape=False, order=0)
    dim = np.array(weight_fraction.shape[1:])
    dim2 = np.array(abs_corr.shape[1:])
    diff = (dim2 - dim) / 2
    abs_corr = abs_corr[:, diff[0]:diff[0] + dim[0],
                        diff[1]:diff[1] + dim[1], diff[2]:diff[2] + dim[2]]
    np.place(abs_corr, (abs_corr == 0.), 1.)
    # abs_corr*=(abs_corr == 0.)
    abs_corr[:, 0] = np.ones_like(abs_corr[:, 0])
    abs_corr[abs_corr > 1.] = 1.
    return abs_corr


def absorption_correction_matrix2(weight_fraction,
                                  xray_lines,
                                  elements,
                                  thickness,
                                  density,
                                  azimuth_angle,
                                  elevation_angle,
                                  mask_el,
                                  tilt=0.):
    """
    Matrix of absorption for an isotropic 3D data cube of composition

    Parameters
    ----------
    weight_fraction: np.array
        dim = {el,z,y,x} The sample composition
    xray_lines: list of str
         The X-ray lines eg ['Al_Ka']
    elements: list of str
        The elements of the sample
    thickness: float
            Set the thickness in cm
    density: array
        dim = {z,y,x} The density to correct of the sample.
        If 'auto' use the weight_fraction
        to calculate it. in gm/cm^3
    azimuth_angle: float
        the azimuth_angle in degree
    elevation_angle: float
        the elevation_angle in degree
    mask: bool array
        A mask to be applied to the correction absorption
    tilt: float
        The tilt of the sample.

    Return
    ------
    The absorption matrix: np.array
        {xray_lines,z,y,x}
    """
    from hyperspy import utils
    from hyperspy.misc import material

    x_ax, y_ax, z_ax = 3, 2, 1
    order = 3
    weight_fraction_r = ndimage.rotate(weight_fraction,
                                       angle=-azimuth_angle[0],
                                       axes=(x_ax, y_ax),
                                       order=order, mode='reflect')
    weight_fraction_r = ndimage.rotate(weight_fraction_r,
                                       angle=-elevation_angle-tilt,
                                       axes=(x_ax, z_ax), order=order,
                                       mode='reflect')
    weight_fraction_r2 = ndimage.rotate(weight_fraction,
                                        angle=-azimuth_angle[1],
                                        axes=(x_ax, y_ax),
                                        order=order, mode='reflect')
    weight_fraction_r2 = ndimage.rotate(weight_fraction_r2,
                                        angle=-elevation_angle-tilt,
                                        axes=(x_ax, z_ax), order=order,
                                        mode='reflect')
    elements = np.array(elements)
    if density == 'auto':
        density_r = material._density_of_mixture_of_pure_elements(
            weight_fraction_r * 100., elements)
        density_r2 = density_r
    else:
        density_r = ndimage.rotate(density,
                                   angle=-azimuth_angle[0],
                                   axes=(x_ax - 1, y_ax - 1),
                                   order=order, mode='nearest')
        density_r = ndimage.rotate(density_r,
                                   angle=-elevation_angle-tilt,
                                   axes=(x_ax - 1, z_ax - 1),
                                   order=order, mode='nearest')
        density_r2 = ndimage.rotate(density,
                                    angle=-azimuth_angle[1],
                                    axes=(x_ax - 1, y_ax - 1),
                                    order=order, mode='nearest')
        density_r2 = ndimage.rotate(density_r2,
                                    angle=-elevation_angle-tilt,
                                    axes=(x_ax - 1, z_ax - 1),
                                    order=order, mode='nearest')
    if mask_el is None:
        mask_el_r = [1.] * len(xray_lines)
        mask_el_r2 = mask_el_r
    else:
        mask_el_r = ndimage.rotate(mask_el,
                                   angle=-azimuth_angle[0],
                                   axes=(x_ax, y_ax),
                                   order=0, mode='reflect')
        mask_el_r = ndimage.rotate(mask_el_r,
                                   angle=-elevation_angle-tilt,
                                   axes=(x_ax, z_ax),
                                   order=0, mode='reflect')
        mask_el_r2 = ndimage.rotate(mask_el,
                                    angle=-azimuth_angle[1],
                                    axes=(x_ax, y_ax),
                                    order=0, mode='reflect')
        mask_el_r2 = ndimage.rotate(mask_el_r2,
                                    angle=-elevation_angle-tilt,
                                    axes=(x_ax, z_ax),
                                    order=0, mode='reflect')
    # abs_corr = np.zeros_like(weight_fraction_r)
    abs_corr = np.zeros([len(xray_lines)]+list(weight_fraction_r.shape[1:]))
    abs_corr2 = np.zeros([len(xray_lines)]+list(weight_fraction_r2.shape[1:]))
    for i, xray_line in enumerate(xray_lines):
        mac = utils.material.\
            mass_absorption_coefficient_of_mixture_of_pure_elements(
                elements, weight_fraction_r, [xray_line])[0]
        fact = np.nan_to_num(
            density_r * mac * thickness / 2 * mask_el_r[i])
        fact_sum = np.zeros_like(fact)
        fact_sum[:, :, -1] = fact[:, :, -1] / 2.  # approx
        for j in range(len(fact[0, 0]) - 2, -1, -1):
            fact_sum[:, :, j] = fact_sum[:, :, j+1] + fact[:, :, j]
        abs_corr[i] = fact_sum
        #
        mac2 = utils.material.\
            mass_absorption_coefficient_of_mixture_of_pure_elements(
                elements, weight_fraction_r2, [xray_line])[0]
        fact2 = np.nan_to_num(
            density_r2 * mac2 * thickness / 2 * mask_el_r2[i])
        fact_sum2 = np.zeros_like(fact2)
        fact_sum2[:, :, -1] = fact2[:, :, -1] / 2.  # approx
        for j in range(len(fact2[0, 0]) - 2, -1, -1):
            fact_sum2[:, :, j] = fact_sum2[:, :, j+1] + fact2[:, :, j]
        abs_corr2[i] = fact_sum2
        # abs_co = np.exp(-(fact_sum + fact_sum2))
        # abs_corr[i] = abs_co
    abs_corr = ndimage.rotate(abs_corr, angle=elevation_angle,
                              axes=(x_ax, z_ax), reshape=False, order=0)
    abs_corr = ndimage.rotate(abs_corr, angle=azimuth_angle[0],
                              axes=(x_ax, y_ax), reshape=False, order=0)
    abs_corr2 = ndimage.rotate(abs_corr2, angle=elevation_angle,
                               axes=(x_ax, z_ax), reshape=False, order=0)
    abs_corr2 = ndimage.rotate(abs_corr2, angle=azimuth_angle[1],
                               axes=(x_ax, y_ax), reshape=False, order=0)
    # abs_corr = np.exp(-(abs_corr + abs_corr2))
    abs_corr = np.exp(-abs_corr)
    abs_corr2 = np.exp(-abs_corr2)
    # abs_corr = (abs_corr + abs_corr2) / 2.
#    abs_co = abs_corr[-1]
#    # Masking
#    interv = (abs_co.max() - abs_co.min())
#    nb_same_pix, max_corr, atol = (6, 0.9, 0.01)
#    index_same = [None] + range(1, nb_same_pix)
#    mask = abs_co[:, :, : -nb_same_pix + 1] < interv*max_corr + abs_co.min()
#    for i in range(nb_same_pix - 1):
#        mask = np.bitwise_and(mask,
#                              abs(abs_co[:, :, nb_same_pix-1:] -
#                                  abs_co[:, :, index_same[i]:
#                                         -index_same[nb_same_pix-i-1]])
#                              < interv*atol)
#    for i in range(nb_same_pix - 1):
#        mask = np.insert(mask, -1, False, axis=2)
#    for i, xray_line in enumerate(xray_lines):
#        # abs_corr[i] *= mask
#        np.place(abs_corr[i], mask, 1.0)
#    abs_corr = ndimage.rotate(abs_corr, angle=elevation_angle,
#                              axes=(x_ax, z_ax), reshape=False, order=0)
#    abs_corr = ndimage.rotate(abs_corr, angle=azimuth_angle,
#                              axes=(x_ax, y_ax), reshape=False, order=0)
    dim = np.array(weight_fraction.shape[1:])
    dim2 = np.array(abs_corr.shape[1:])
    diff = (dim2 - dim) / 2
    abs_corr = abs_corr[:, diff[0]:diff[0] + dim[0],
                        diff[1]:diff[1] + dim[1], diff[2]:diff[2] + dim[2]]
    np.place(abs_corr, (abs_corr == 0.), 1.)
    # abs_corr*=(abs_corr == 0.)
    abs_corr[:, 0] = np.ones_like(abs_corr[:, 0])
    abs_corr[abs_corr > 1.] = 1.
    return abs_corr
