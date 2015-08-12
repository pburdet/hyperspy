from __future__ import division

import numpy as np
import math
import execnet
# import os
import copy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from functools import reduce


# import hyperspy.utils
# from hyperspy.misc.config_dir import config_path
# import hyperspy.components as components
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import database
# from hyperspy.misc.eds.MAC import MAC_db as MAC
from hyperspy.misc.eds import physical_model
from functools import reduce



def _get_element_and_line(xray_line):
    lim = xray_line.find('_')
    return xray_line[:lim], xray_line[lim + 1:]


def _get_energy_xray_line(xray_line):
    energy, line = _get_element_and_line(xray_line)
    return elements_db[energy]['Atomic_properties']['Xray_lines'][
        line]['energy (keV)']


def _get_xray_lines_family(xray_line):
    return xray_line[:xray_line.find('_') + 2]


def get_FWHM_at_Energy(energy_resolution_MnKa, E):
    """Calculates the FWHM of a peak at energy E.

    Parameters
    ----------
    energy_resolution_MnKa : float
        Energy resolution of Mn Ka in eV
    E : float
        Energy of the peak in keV

    Returns
    -------
    float : FWHM of the peak in keV

    Notes
    -----
    From the textbook of Goldstein et al., Plenum publisher,
    third edition p 315

    """
    FWHM_ref = energy_resolution_MnKa
    E_ref = _get_energy_xray_line('Mn_Ka')

    FWHM_e = 2.5 * (E - E_ref) * 1000 + FWHM_ref * FWHM_ref

    return math.sqrt(FWHM_e) / 1000  # In mrad


def xray_range(xray_line, beam_energy, density='auto'):
    """Return the Anderson-Hasler X-ray range.

    Return the maximum range of X-ray generation in a pure bulk material.

    Parameters
    ----------
    xray_line: str
        The X-ray line, e.g. 'Al_Ka'
    beam_energy: float
        The energy of the beam in kV.
    density: {float, 'auto'}
        The density of the material in g/cm3. If 'auto', the density
        of the pure element is used.

    Returns
    -------
    X-ray range in micrometer.

    Examples
    --------
    >>> # X-ray range of Cu Ka in pure Copper at 30 kV in micron
    >>> hs.eds.xray_range('Cu_Ka', 30.)
    1.9361716759499248

    >>> # X-ray range of Cu Ka in pure Carbon at 30kV in micron
    >>> hs.eds.xray_range('Cu_Ka', 30., hs.material.elements.C.
    >>>                      Physical_properties.density_gcm3)
    7.6418811280855454

    Notes
    -----
    From Anderson, C.A. and M.F. Hasler (1966). In proceedings of the
    4th international conference on X-ray optics and microanalysis.

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 286

    """

    element, line = _get_element_and_line(xray_line)
    if density == 'auto':
        density = elements_db[
            element][
            'Physical_properties'][
            'density (g/cm^3)']
    Xray_energy = _get_energy_xray_line(xray_line)

    return 0.064 / density * (np.power(beam_energy, 1.68) -
                              np.power(Xray_energy, 1.68))


def electron_range(element, beam_energy, density='auto', tilt=0):
    """Return the Kanaya-Okayama electron range.

    Return the maximum electron range in a pure bulk material.

    Parameters
    ----------
    element: str
        The element symbol, e.g. 'Al'.
    beam_energy: float
        The energy of the beam in keV.
    density: {float, 'auto'}
        The density of the material in g/cm3. If 'auto', the density of
        the pure element is used.
    tilt: float.
        The tilt of the sample in degrees.

    Returns
    -------
    Electron range in micrometers.

    Examples
    --------
    >>> # Electron range in pure Copper at 30 kV in micron
    >>> hs.eds.electron_range('Cu', 30.)
    2.8766744984001607

    Notes
    -----
    From Kanaya, K. and S. Okayama (1972). J. Phys. D. Appl. Phys. 5, p43

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 72.

    """

    if density == 'auto':
        density = elements_db[
            element]['Physical_properties']['density (g/cm^3)']
    Z = elements_db[element]['General_properties']['Z']
    A = elements_db[element]['General_properties']['atomic_weight']

    return (0.0276 * A / np.power(Z, 0.89) / density *
            np.power(beam_energy, 1.67) * math.cos(math.radians(tilt)))


def take_off_angle(tilt_stage,
                   azimuth_angle,
                   elevation_angle):
    """Calculate the take-off-angle (TOA).

    TOA is the angle with which the X-rays leave the surface towards
    the detector.

    Parameters
    ----------
    tilt_stage: float
        The tilt of the stage in degrees. The sample is facing the detector
        when positively tilted.
    azimuth_angle: float
        The azimuth of the detector in degrees. 0 is perpendicular to the tilt
        axis.
    elevation_angle: float
        The elevation of the detector in degrees.

    Returns
    -------
    take_off_angle: float.
        In degrees.

    Examples
    --------
    >>> hs.eds.take_off_angle(tilt_stage=10.,
    >>>                          azimuth_angle=45., elevation_angle=22.)
    28.865971201155283

    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)

    """

    a = math.radians(90 + tilt_stage)
    b = math.radians(azimuth_angle)
    c = math.radians(elevation_angle)

    return math.degrees(np.arcsin(-math.cos(a) * math.cos(b) * math.cos(c)
                                  + math.sin(a) * math.sin(c)))


def xray_lines_model(elements=['Al', 'Zn'],
                     beam_energy=200,
                     weight_percents=[50, 50],
                     energy_resolution_MnKa=130,
                     counts_rate=1,
                     live_time=1.,
                     energy_axis={'name': 'E', 'scale': 0.01, 'units': 'keV',
                                  'offset': -0.1, 'size': 1024}
                     ):
    """
    Generate a model of X-ray lines using a Gaussian epr x-ray lines.

    The area under a main peak (alpha) is equal to 1 and weighted by the
    composition.

    Parameters
    ----------
    elements : list of strings
        A list of chemical element symbols.
    beam_energy: float
        The energy of the beam in keV.
    weight_percents: list of float
        The composition in weight percent.
    energy_resolution_MnKa: float
        The energy resolution of the detector in eV
    counts_rate: int
        Number of detected X-ray per second
    live_time:
        Time of active X-ray detections in second
    energy_axis: dic
        The dictionary for the energy axis. It must contains 'size' and the
        units must be 'eV' of 'keV'.

    Example
    -------
    >>> s = utils_eds.simulate_model(['Cu', 'Fe'], beam_energy=30)
    >>> s.plot()
    """
    from hyperspy._signals.eds_tem_spectrum_simulation \
        import EDSTEMSpectrumSimulation
    from hyperspy.model import Model
    from hyperspy import components
    s = EDSTEMSpectrumSimulation(np.zeros(energy_axis['size']),
                                 axes=[energy_axis])
    s.set_microscope_parameters(
        beam_energy=beam_energy,
        energy_resolution_MnKa=energy_resolution_MnKa,
        live_time=live_time)
    s.add_elements(elements)
    if weight_percents is None:
        weight_percents = [100] * len(elements)
    weight_percents = np.array(weight_percents, dtype=float)
    s.metadata.Sample.weight_percents = weight_percents
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.counts_rate =\
        counts_rate
    weight_fractions = weight_percents / weight_percents.sum()
    m = Model(s)
    for i, (element, weight_fraction) in enumerate(zip(
            elements, weight_fractions)):
        for line in elements_db[
                element]['Atomic_properties']['Xray_lines'].keys():
            line_energy = elements_db[element]['Atomic_properties'][
                'Xray_lines'][line]['energy (keV)']
            ratio_line = elements_db[element]['Atomic_properties'][
                'Xray_lines'][line]['weight']
            if s._get_xray_lines_in_spectral_range(
                    [element+'_'+line])[1] == []:
                g = components.Gaussian()
                g.centre.value = line_energy
                g.sigma.value = get_FWHM_at_Energy(
                    energy_resolution_MnKa, line_energy) / 2.355
                g.A.value = live_time * counts_rate * \
                    weight_fraction * ratio_line
                m.append(g)
    s.data = m.as_signal().data
    return s


def get_index_from_names(self, axis_names, index_name, axis_name_in_mp=True):
    """Get the index of an axis that is link to a list of names.


    Parameters
    ----------
    axis_names: list of str | str
        the list name corresponding to the axis
    index_name: str
        The name of the index to find
    axis_name_in_mp: bool
        if axis_name is in metadata.Sample.
    """
    if axis_name_in_mp:
        axis_names = self.metadata.Sample[axis_names]

    for i, name in enumerate(axis_names):
        if name == index_name:
            return i


def get_sample_mass_absorption_coefficients(energies,
                                            weight_fraction,
                                            elements):
    """Return the mass absorption coefficients of a sample

    Parameters
    ----------
    energies: {float or list of float or str or list of str}
        The energy or energies of the Xray in keV, or the name eg 'Al_Ka'
    weight_fraction: list of float
        the composition of the sample
    elements: {list of str,'auto'}
        The list of element symbol of the absorber, e.g. ['Al','Zn'].
        if 'auto', use the elements of the X-ray lines

    Return
    ------
    mass absorption coefficient in cm^2/g
    """
    from hyperspy.misc import material
    if hasattr(elements, '__iter__') is False and elements == 'auto':
        if isinstance(energies[0], str) is False:
            raise ValueError("need X-ray lines name for elements='auto'")
        elements = []
        for xray_line in energies:
            element, line = _get_element_and_line(xray_line)
            elements.append(element)
        elements = set(elements)
    if len(elements) != len(weight_fraction):
        raise ValueError("Add elements first, see 'set_elements'")
    if isinstance(weight_fraction[0], float):
        mac = 0
    else:
        mac = weight_fraction[0].deepcopy()
        mac.data = np.zeros_like(mac.data)
    for el, weight in zip(elements, weight_fraction):
        mac += weight * np.array(material.mass_absorption_coefficient(
            el, energies))
    return mac


def simulate_one_spectrum(nTraj, dose=100, mp='gui',
                          elements='auto',
                          compo_at='auto',
                          density='auto',
                          detector='Si(Li)',
                          gateway='auto'):
    # must create a class, EDS simulation
    # to be retested, det still here
    """"
    Simulate a spectrum using DTSA-II (NIST-Monte)
    Parameters
    ----------

    nTraj: int
        number of electron trajectories

    dose: float
        Electron current time the live time in nA*sec

    mp: dict
        Microscope parameters. If 'gui' raise a general interface.

    elements: list of str
        Set the elements. If auto, look in mp.Sample if elements are defined.
        auto cannot be used with 'gui' option.

    compo_at: list of string
        Give the composition (atomic). If auto, equally parted

    density: list of float
        Set the density. If 'auto', obtain from the compo_at.

    detector: str
        Give the detector name defined in DTSA-II

    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Note
    ----

    For further details on DTSA-II please refer to
    http://www.cstl.nist.gov/div837/837.02/epq/dtsa2/index.html

    """
    from hyperspy import signals
    from hyperspy.misc import material

    if mp == 'gui':
        if elements == 'auto':
            raise ValueError('Elements need to be set (set_elements) ' +
                             'with gui option')
            return 0
        else:
            spec.set_microscope_parameters()
            spec.set_elements(elements)
            spec.add_lines()
        mp = spec.metadata
    else:
        mp = mp.deepcopy()

    if elements == 'auto':
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError('Elements need to be set (set_elements)')
            return 0
    else:
        mp.Sample.elements = elements

    if compo_at == 'auto':
        compo_at = []
        # Not tested. atm vs w
        # if hasattr(mp.Sample, 'quant'):
        #    for elm in elements:
        #        compo_at.append(float(spec.get_result(elm,'quant').data))
        # else:
        for elm in elements:
            compo_at.append(1. / len(elements))
    mp.Sample.compo_at = compo_at

    compo_wt = np.array(
        material.atomic_to_weight(
            compo_at, elements)) / 100
    compo_wt = list(compo_wt)
    if density == 'auto':
        density = material.density_of_mixture_of_pure_elements(
            compo_wt, elements)
    mp.Sample.density = density

    e0 = mp.Acquisition_instrument.SEM.beam_energy
    tilt = np.radians(mp.Acquisition_instrument.SEM.tilt_stage)
    ltime = mp.Acquisition_instrument.SEM.Detector.EDS.live_time

    if gateway == 'auto':
        gateway = get_link_to_jython()

    spec_dir = get_detector_properties(detector,
                                       gateway=gateway)
    if spec_dir.metadata.General.title != detector:
        print('The default detector '
              + spec_dir.metadata.General.title + ' was used')
    channelMax = len(spec_dir.data)
    WD = spec_dir.metadata.Acquisition_instrument.SEM.\
        Detector.EDS.optimal_working_distance
    channel = gateway.remote_exec("""
        import dtsa2
        import math
        epq = dtsa2.epq
        nm = dtsa2.nm
        elements = """ + str(elements) + """
        elms = []
        for element in elements:
            elms.append(getattr(dtsa2.epq.Element,element))
        density = """ + str(density) + """
        compo_wt = """ + str(compo_wt) + """
        e0 =  """ + str(e0) + """
        dose =  """ + str(dose) + """
        tilt = """ + str(tilt) + """
        live_time = """ + str(ltime) + """
        channelMax = """ + str(channelMax) + """
        WD = """ + str(WD * 1.0e-3) + """
        nTraj = """ + str(nTraj) + """

        #Position of detector and sample (WD in km, d-to-crystal in m)
        origin = [0.0,0.0,WD]
        z0 = origin[2]
        det = dtsa2.findDetector('""" + detector + """')
        el = 0
        if len(elms) == 1:
            mat=epq.MaterialFactory.createPureElement(elms[el])
        else:
            mat = epq.Material(epq.Composition(elms,compo_wt ),
                                    epq.ToSI.gPerCC(density))

        # Create a simulator and initialize it
        monteb = nm.MonteCarloSS()
        monteb.setBeamEnergy(epq.ToSI.keV(e0))

        # top substrat
        monteb.addSubRegion(monteb.getChamber(), mat,
            nm.MultiPlaneShape.createSubstrate([0.0,
            math.sin(tilt),-math.cos(tilt)], origin) )
        # Add event listeners to model characteristic radiation
        xrel=nm.XRayEventListener2(monteb,det)
        monteb.addActionListener(xrel)

        # Add event listeners to model bBremsstrahlung
        brem=nm.BremsstrahlungEventListener(monteb,det)
        monteb.addActionListener(brem)
        # Reset the detector and run the electrons
        det.reset()
        monteb.runMultipleTrajectories(nTraj)
        # Get the spectrum and assign properties
        specb=det.getSpectrum(dose*1.0e-9 / (
                    nTraj * epq.PhysicalConstants.ElectronCharge) )
        dtsa2.display(specb)
        for i in range(channelMax):
            channel.send(specb.getCounts(i))

    """)

    datas = []
    for i, item in enumerate(channel):
        datas.append(item)
    spec = _create_spectrum_from_DTSA_detector(datas=np.array(datas),
                                               nTraj=nTraj, mp=mp,
                                               spec_detector=spec_dir,
                                               gateway=gateway)
    return spec


def _create_spectrum_from_DTSA_detector(datas,
                                        nTraj,
                                        mp,
                                        spec_detector,
                                        gateway,
                                        title='Simulated spectrum'):
    """
    Import the properties of a DTSAII and create a spectrum

    Parameters
    ----------
    datas: np.array
        The simulated data
    nTraj: int
        number of electron trajectories
    mp: dict
        The spectrum metadata containing the simulation properties.
        eg s.metadata
    spec_detector: signals.EDSSEMSpectrum
        Give the detector name defined in DTSA-II
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.
    """
    from hyperspy import signals

    if 'TEM' in mp.Acquisition_instrument:
        spec = signals.EDSTEMSpectrum(datas)
        spec.metadata = copy.deepcopy(mp)
        microscope = spec.metadata.Acquisition_instrument.TEM
        mp_mic = mp.Acquisition_instrument.TEM
    else:
        spec = signals.EDSSEMSpectrum(datas)
        spec.metadata = copy.deepcopy(mp)
        microscope = spec.metadata.Acquisition_instrument.SEM
        mp_mic = mp.Acquisition_instrument.SEM

    # Parameters from DTSAII det
    microscope.Detector = \
        spec_detector.metadata.Acquisition_instrument.SEM.Detector
    spec.axes_manager._axes[-1] = spec_detector.axes_manager._axes[-1]
    spec.original_metadata.spectrum_properties = \
        spec_detector.original_metadata.spectrum_properties

    # Get back the live time
    microscope.Detector.EDS.live_time = mp_mic.Detector.EDS.live_time

    spec.metadata.General.title = title
    spec.metadata.add_node('simulation')
    spec.metadata.simulation.nTraj = nTraj
    spec.metadata.simulation.software = 'NistMonte'
    spec.metadata.simulation.detector = spec_detector.metadata.General.title

    return spec


def simulate_Xray_depth_distribution(nTraj, bins=120, mp='gui',
                                     elements='auto',
                                     xray_lines='auto',
                                     compo_at='auto',
                                     density='auto',
                                     detector='Si(Li)',
                                     gateway='auto'):
    # must create a class, EDS simulation
    # check if all param well stored
    # dim*cos(tilt)
    """"
    Simulate the X-ray depth distribution using DTSA-II (NIST-Monte)

    Parameters
    ----------

    nTraj: int
        number of electron trajectories

    bins: int
        number of bins in the z direction

    mp: dict
        Microscope parameters. If 'gui' raise a general interface.

    elements: list of str | 'auto'
        Set the elements. If auto, look in mp.Sample if elements are defined.
        auto cannot be used with 'gui' option.

    xray_lines: list of str | 'auto'
        Set the elements. If auto, look in mp.Sample if elements are defined.

    compo_at: list of flaot | 'auto'
        Set the atomic fraction (composition). If auto, get the values in quant (if a
        spectrum). Or equal repartition between elements.

    density: list of float
        Set the density. If 'auto', obtain from the compo_at.

    detector: str
        Give the detector name defined in DTSA-II

    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Return
    ------

    A signals.Spectrum. Depth (nm) as signal axis. Generated/emitted and
    Xray-lines as navigation axis.

    Note
    ----

    For further details on DTSA-II please refer to
    http://www.cstl.nist.gov/div837/837.02/epq/dtsa2/index.html

    """
    from hyperspy import signals
    from hyperspy.misc import material

    spec = signals.EDSSEMSpectrum(np.zeros(1024))
    if mp == 'gui':
        spec.set_microscope_parameters()
        if elements == 'auto':
            raise ValueError('Elements need to be set (set_elements) ' +
                             'with gui option')
            return 0
        else:
            spec.set_elements(elements)
            spec.set_lines()
        mp = spec.metadata
    else:
        spec.metadata = mp.deepcopy()
        mp = spec.metadata

    if elements == 'auto':
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError('Elements need to be set (set_elements)')

    if xray_lines == 'auto':
        if hasattr(mp.Sample, 'xray_lines'):
            xray_lines = list(mp.Sample.xray_lines)
        else:
            raise ValueError('xray_lines need to be set (set_lines)')

    if compo_at == 'auto':
        compo_at = []
        # if hasattr(mp.Sample, 'quant'):
        #    for elm in elements:
        #        compo_at.append(float(spec.get_result(elm,'quant').data))
        # else:
        for elm in elements:
            compo_at.append(1. / len(elements))

    compo_wt = np.array(material.atomic_to_weight(compo_at, elements)) / 100
    compo_wt = list(compo_wt)
    if density == 'auto':
        density = material.density_of_mixture_of_pure_elements(
            compo_wt, elements)

    e0 = mp.Acquisition_instrument.SEM.beam_energy
    tilt = np.radians(mp.Acquisition_instrument.SEM.tilt_stage)
    ltime = mp.Acquisition_instrument.SEM.Detector.EDS.live_time
    elevation = np.radians(
        mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle)
    azim = np.radians(
        90 -
        mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle)

    if gateway == 'auto':
        gateway = get_link_to_jython()
    channel = gateway.remote_exec("""
        import dtsa2
        import math
        epq = dtsa2.epq
        epu = dtsa2.epu
        nm = dtsa2.nm
        elements = """ + str(elements) + """
        xray_lines = """ + str(xray_lines) + """
        elms = []
        for element in elements:
            elms.append(getattr(dtsa2.epq.Element,element))
        density = """ + str(density) + """
        compo_wt = """ + str(compo_wt) + """
        e0 =  """ + str(e0) + """
        tilt = """ + str(tilt) + """
        elevation = """ + str(elevation) + """
        azim = """ + str(azim) + """
        live_time = """ + str(ltime) + """
        nTraj = """ + str(nTraj) + """

        #Position of detector and sample (WD in km, d-to-crystal in m)
        prop = epq.SpectrumProperties()
        prop.setDetectorPosition(elevation, azim, 0.005, 2e-5)
        posi = prop.getArrayProperty(epq.SpectrumProperties.DetectorPosition)
        posi = [posi[0]/1000,posi[1]/1000,posi[2]/1000]
        origin = [0.0,0.0,5e-6]
        z0 = origin[2]

        el = 0
        if len(elms) == 1:
            mat=epq.MaterialFactory.createPureElement(elms[el])
        else:
            mat = epq.Material(epq.Composition(elms,compo_wt ),
                                    epq.ToSI.gPerCC(density))

        # Create a simulator and initialize it
        monteb = nm.MonteCarloSS()
        monteb.setBeamEnergy(epq.ToSI.keV(e0))

        # top substrat
        monteb.addSubRegion(monteb.getChamber(), mat,
            nm.MultiPlaneShape.createSubstrate([0.0,0.0,-1.0], origin) )

        monteb.rotate([0.0,0.0,z0], -tilt,0.0,0.0)

        # Add event listeners to model characteristic radiation
        xrel=nm.XRayEventListener2(monteb,posi)
        monteb.addActionListener(xrel)

        dim=epq.ElectronRange.KanayaAndOkayama1972.compute(mat,
            epq.ToSI.keV(e0)) / mat.getDensity()
        prz = nm.PhiRhoZ(xrel, z0 - 0 * dim, z0 + 1 * dim, """ + str(bins) + """)
        xrel.addActionListener(prz)

        # Reset the detector and run the electrons
        #det.reset()
        monteb.runMultipleTrajectories(nTraj)

        for xray_line in xray_lines:
            lim = xray_line.find('_')
            el = getattr(dtsa2.epq.Element,xray_line[:lim])
            li = xray_line[lim+1:]
            if li == 'Ka':
                transSet = epq.XRayTransition(el,0)
            elif li == 'La':
                transSet = epq.XRayTransition(el,12)
            elif li == 'Ma':
                transSet = epq.XRayTransition(el,72)

            res = prz.getGeneratedIntensity(transSet)
            for re in res:
                channel.send(re)
            res = prz.getEmittedIntensity(transSet)
            for re in res:
                channel.send(re)

        channel.send(dim)

    """)

    datas = []
    for i, item in enumerate(channel):
        datas.append(item)

    dim = datas[-1]
    datas = np.reshape(datas[:-1], (len(xray_lines), 2, bins))
    datas = np.rollaxis(datas, 1, 0)

    frz = signals.EDSSEMSpectrum(np.array(datas))
    frz.metadata.Acquisition_instrument.SEM = mp.Acquisition_instrument.SEM
    mp = frz.metadata
    mp.add_node('Sample')
    mp.Sample.elements = elements
    mp.Sample.compo_at = compo_at
    mp.Sample.xray_lines = xray_lines
    mp.Sample.density = density

    frz.axes_manager[0].name = 'Generated|Emitted'
    frz.axes_manager[1].name = 'xray_lines'
    #frz.axes_manager[1].units = 'keV'
    frz.axes_manager[2].name = 'Depth'
    frz.axes_manager[2].units = 'nm'
    frz.axes_manager[2].scale = dim / bins * 1000000000
    mp.General.title = 'Simulated Depth distribution'

    mp.add_node('simulation')
    mp.simulation.nTraj = nTraj
    #mp.signal_origin = "simulation"
    mp.simulation.software = 'NistMonte'

    return frz


def get_link_to_jython():
    # must go in IO
    """Return the execnet gateway to jython.
    """
    return execnet.makegateway(
        "popen//python=jython")


def load_EDSSEMSpectrum(filenames=None,
                        record_by=None,
                        signal_type=None,
                        signal_origin=None,
                        stack=False,
                        stack_axis=None,
                        new_axis_name="stack_element",
                        mmap=False,
                        mmap_dir=None,
                        **kwds):
    # must desappear, result == spec
    """Load the EDSSEMSpectrum and the result.

    See also
    --------

    load
    """
    from hyperspy.io import load

    s = load(filenames, record_by, signal_type, signal_origin, stack,
             stack_axis, new_axis_name, mmap, mmap_dir, **kwds)

    mp = s.metadata
    if hasattr(mp, 'Sample'):
        for result in ['standard_spec', 'kratios', 'quant', 'quant_enh', 'intensities']:
            if hasattr(mp.Sample, result):
                #mp.Sample[result] = mp.Sample[result].split()
                _set_result_signal_list(mp, result)

    return s

# might be simplified with auto split...
# must desappear


def _set_result_signal_list(mp, result):
    """
    signal to list of signal use to load()
    """
    std = mp.Sample[result]
    # if '_' in std.metadata.General.title:
    #    number_of_parts=len(mp.Sample.xray_lines)
    #    is_xray = True
    # else:
    #    number_of_parts=len(mp.Sample.elements)
    #    is_xray = False
    number_of_parts = std.data.shape[0]

    if result == 'standard_spec':
        # Need to change
        # number_of_parts=len(mp.Sample.elements)
        if "Acquisition_instrument.SEM" in std.metadata:
            l_time = std.metadata.Acquisition_instrument\
                .SEM.Detector.EDS.live_time
        elif "Acquisition_instrument.TEM" in std.metadata:
            l_time = std.metadata.Acquisition_instrument\
                .TEM.Detector.EDS.live_time
        # number_of_parts=len(mp.Sample.xray_lines)
        title_back = std.metadata.General.title
        title_back = title_back[title_back.find('_'):]
        temp = std.split(axis=0, number_of_parts=number_of_parts)
    elif len(std.data.shape) == 1:
        temp = std.split(axis=0, number_of_parts=number_of_parts)
    else:
        #temp = std.split(axis=1,number_of_parts=number_of_parts)
        temp = std.split(axis=-3, number_of_parts=number_of_parts)
    std = []
    for i, tp in enumerate(temp):
        tp = tp.squeeze()
        if result == 'standard_spec':
            # to change
            if 'xray_lines' in mp.Sample:
                if number_of_parts == len(mp.Sample.xray_lines):
                    # if number_of_parts == len(mp.Sample.elements):
                    el, li = _get_element_and_line(mp.Sample.xray_lines[i])
                else:
                    el, li = _get_element_and_line(mp.Sample.elements[i])
            else:
                el = mp.Sample.elements[i]
            tp.metadata.General.title = el + title_back
            if "Acquisition_instrument.SEM" in tp.metadata:
                tp.metadata.Acquisition_instrument.\
                    SEM.Detector.EDS.live_time = l_time[i]
            elif "Acquisition_instrument.TEM" in tp.metadata:
                tp.metadata.Acquisition_instrument.\
                    TEM.Detector.EDS.live_time = l_time[i]
        elif number_of_parts == len(mp.Sample.xray_lines):
            tp.metadata.General.title = result + ' ' + mp.Sample.xray_lines[i]
        elif number_of_parts == len(mp.Sample.elements):
            tp.metadata.General.title = result + ' ' + mp.Sample.elements[i]
        std.append(tp)
    mp.Sample[result] = std


# Us the _create_spectrum function
def simulate_linescan(nTraj,
                      compos_at,
                      min_max,
                      lscan_scale,
                      lscan_axis='x',
                      elements='auto',
                      density='auto',
                      mp='gui',
                      detector='Si(Li)',
                      gateway='auto'):
    """Simulate a linescan accross a boundary between two materials

    Implemented for linescan along z. Spectra simulated using DTSA-II
    (NIST-Monte)

    Parameters
    ----------

    nTraj: int
        number of electron trajectories

    compos_at: list of list of float
        Give the atomic fraction of each material (right/left or top/bottom).
         (eg. [[0.33,0.33,0.33],[1,0,0])

    min_max: list of float
        The start and the end of the linesscan, zero being the interface.
        Given in [mum].

    lscan_scale: float
        the distance between two spectrum. Given in [mum].

    lscan_axis: 'x'|'y'|'z'
        the orientation of the linescan. The interface is perpendiculait to
        the axis

    elements: list of str
        All elements present. If auto, look in mp.Sample if elements are defined.
        auto cannot be used with 'gui' option.

    density: list of float
        The density of each material. If 'auto', obtain from the compositions.

    mp: dict
        Microscope parameters. If 'gui' raise a general interface.

    detector: str
        Give the detector name defined in DTSA-II. 'Si(Li)' is the default one

    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Note
    ----

    For further details on DTSA-II please refer to
    http://www.cstl.nist.gov/div837/837.02/epq/dtsa2/index.html
    """
    from hyperspy import signals
    from hyperspy.misc import material
    spec = signals.EDSSEMSpectrum(np.zeros(1024))
    if mp == 'gui':
        spec.set_microscope_parameters()
        if elements == 'auto':
            raise ValueError('Elements need to be set (set_elements) ' +
                             'with gui option')
            return 0
        else:
            spec.set_elements(elements)
            spec.set_lines()
        mp = spec.metadata
    else:
        spec.metadata = mp.deepcopy()
        mp = spec.metadata

    if elements == 'auto':
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError('Elements need to be set (set_elements)')
            return 0
    else:
        mp.Sample.elements = elements

    compos_wt = []
    for compo_at in compos_at:
        compos_wt.append(list(np.array(
            material.atomic_to_weight(
                compo_at, elements)) / 100))

    if density == 'auto':
        density = []
        for compo_wt in compos_wt:
            density.append(
                material.density_of_mixture_of_pure_elements(
                    compo_wt, elements))

    mp.Sample.compo_at = compo_at
    mp.Sample.density = density

    e0 = mp.Acquisition_instrument.SEM.beam_energy
    tilt = np.radians(mp.Acquisition_instrument.SEM.tilt_stage)
    ltime = mp.Acquisition_instrument.SEM.Detector.EDS.live_time
    elevation = np.radians(
        mp.Acquisition_instrument.SEM.Detector.EDS.elevation_angle)
    azim = np.radians(
        90 -
        mp.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle)

    if gateway == 'auto':
        gateway = get_link_to_jython()

    def simu_film(interface_xyz):
        channel = gateway.remote_exec("""
            import dtsa2
            import math
            epq = dtsa2.epq
            epu = dtsa2.epu
            nm = dtsa2.nm
            elements = """ + str(elements) + """
            elms = []
            for element in elements:
                elms.append(getattr(dtsa2.epq.Element,element))
            density = """ + str(density) + """
            compos_wt = """ + str(compos_wt) + """
            e0 =  """ + str(e0) + """
            tilt = """ + str(tilt) + """
            live_time = """ + str(ltime) + """
            elevation = """ + str(elevation) + """
            azim = """ + str(azim) + """
            print 1
            nTraj = """ + str(nTraj) + """
            dose = 100

            #Position of detector and sample (WD in km, d-to-crystal in m)
            prop = epq.SpectrumProperties()
            prop.setDetectorPosition(elevation, azim, 0.005, 2e-5)
            posi = prop.getArrayProperty(epq.SpectrumProperties.DetectorPosition)
            posi = [posi[0]/1000.,posi[1]/1000.,posi[2]/1000.]
            origin = [0.0,0.0,2e-5]
            z0 = origin[2]

            det = dtsa2.findDetector('""" + detector + """')
            prop = det.getDetectorProperties()
            prop.setPosition(posi)

            el = []
            for i in range(2):
                el.append([j for j, x in enumerate(compos_wt[i]) if x > 0])

            if len(el[0])==1:
                filmMat=epq.MaterialFactory.createPureElement(elms[el[0][0]])
            else:
                filmMat = epq.Material(epq.Composition(elms,compos_wt[0] ),
                                        epq.ToSI.gPerCC(density[0]))

            if len(el[1])==1:
                subMat=epq.MaterialFactory.createPureElement(elms[el[1][0]])
            else:
                subMat =  epq.Material(epq.Composition(elms,compos_wt[1] ),
                                        epq.ToSI.gPerCC(density[1]))

            # Create a simulator and initialize it
            monteb = nm.MonteCarloSS()
            monteb.setBeamEnergy(epq.ToSI.keV(e0))

            # Create a first layer of film
            interface_xyz =""" + str(interface_xyz * 1e-6) + """
            lscan_axis = '""" + lscan_axis + """'

            big_d = 1e-3

            if lscan_axis != 'z':
                if lscan_axis == 'x':
                    center0= [big_d/2-interface_xyz,0.0,z0]
                    center1 = [-big_d/2-interface_xyz,0.0,z0]
                elif lscan_axis == 'y':
                    center0= [0.0,big_d/2-interface_xyz,z0]
                    center1 = [0.0,-big_d/2-interface_xyz,z0]
                sub0 = nm.MultiPlaneShape.createBlock([big_d]*3,
                    center1,0.0,0.0,0.0)
                block = nm.MultiPlaneShape.createBlock([big_d]*3,
                    center0,0.0,0.0,0.0)
                monteb.addSubRegion(monteb.getChamber(), subMat,block)
                monteb.addSubRegion(monteb.getChamber(), filmMat,sub0)
                monteb.rotate([0,0,z0-big_d/2], -tilt,0.0,0.0)
            elif lscan_axis == 'z':
                center0=epu.Math2.plus(origin,[0.0,0.0,-interface_xyz/2])
                #sub0 = nm.MultiPlaneShape.createSubstrate([0.0,0.0,-1.0], origin)
                #block = nm.MultiPlaneShape.createFilm([0.0,0.0,-1.0],
                #    center0, interface_xyz)
                sub0 = nm.MultiPlaneShape.createSubstrate([0.0,
                    math.sin(tilt),-math.cos(tilt)],origin)
                block = nm.MultiPlaneShape.createFilm([0.0,math.sin(tilt),-math.cos(tilt)],
                    center0, interface_xyz)
                if interface_xyz!=0:
                    film = monteb.addSubRegion(monteb.getChamber(),filmMat,block)
                    sub = monteb.addSubRegion(monteb.getChamber(),
                        subMat,nm.ShapeDifference(sub0,block))
                        #subMat,sub0)
                    #film = monteb.addSubRegion(sub,filmMat,block)
                else:
                    sub = monteb.addSubRegion(monteb.getChamber(), subMat,sub0)


            # Add event listeners to model characteristic radiation
            xrel=nm.XRayEventListener2(monteb,det)
            monteb.addActionListener(xrel)

            # Add event listeners to model bBremsstrahlung
            brem=nm.BremsstrahlungEventListener(monteb,det)
            monteb.addActionListener(brem)
            # Reset the detector and run the electrons
            det.reset()
            monteb.runMultipleTrajectories(nTraj)
            # Get the spectrum and assign properties
            specb=det.getSpectrum(dose*1.0e-9 / (nTraj * epq.PhysicalConstants.ElectronCharge) )
            propsb=specb.getProperties()
            propsb.setNumericProperty(epq.SpectrumProperties.LiveTime, dose)
            propsb.setNumericProperty(epq.SpectrumProperties.FaradayBegin,1.0)
            propsb.setNumericProperty(epq.SpectrumProperties.BeamEnergy,e0)
            noisyb=epq.SpectrumUtils.addNoiseToSpectrum(specb,live_time)
            dtsa2.display(noisyb)

            a = det.calibration.getProperties()

            channelWidth = det.calibration.getChannelWidth()
            offset = det.calibration.getZeroOffset()
            resolution = a.getPropertyByName('Resolution')
            if e0 < 15.0 :
                channelMax = 1024
            else:
                channelMax = 2048
            channel.send(channelWidth)
            channel.send(offset)
            channel.send(resolution)
            for i in range(channelMax):
                channel.send(specb.getCounts(i))
        """)

        datas = []
        for i, item in enumerate(channel):
            if i == 0:
                scale = item
            elif i == 1:
                offset = item
            elif i == 2:
                reso = item
            else:
                datas.append(item)
        return datas, scale, offset, reso

    spec_datas = []
    for thck in mlab.frange(min_max[0], min_max[1], lscan_scale):
        if thck < 0 and lscan_axis == 'z':
            tmp, scale, offset, reso = simu_film(0)
        else:
            tmp, scale, offset, reso = simu_film(thck)
        spec_datas.append(tmp)
    spec = signals.EDSSEMSpectrum(np.array(spec_datas))
    spec.metadata = mp

    mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa = reso
    spec.axes_manager[-1].scale = scale / 1000
    spec.axes_manager[-1].offset = offset
    spec.axes_manager[-1].name = 'Energy'
    spec.axes_manager[-1].units = 'keV'
    spec.axes_manager[0].scale = lscan_scale
    spec.axes_manager[0].offset = min_max[0]
    spec.axes_manager[0].name = 'Scan'
    spec.axes_manager[0].units = '${\mu}m$'
    spec.metadata.General.title = 'Simulated linescan along ' + lscan_axis
    mp.add_node('simulation')
    mp.simulation.nTraj = nTraj
    mp.simulation.software = 'NistMonte'

    #mp.signal_origin = "simulation"

    return spec


# Control of detector geometry not good. Do it trhough interface
# Similar to simulate_one_spectrum, but with 4 detectors

def simulate_one_spectrum_TEM(nTraj, dose=100, mp='gui',
                              elements='auto',
                              weight_fraction='auto',
                              density='auto',
                              thickness='auto',
                              detector='Si(Li)',
                              gateway='auto'):
    """"
    Simulate a spectrum using DTSA-II (NIST-Monte)

    Parameters
    ----------
    nTraj: int
        number of electron trajectories
    dose: float
        Electron current time the live time in nA*sec
    mp: dict
        Microscope parameters. If 'gui' raise a general interface.
    elements: list of str
        Set the elements. If auto, look in mp.Sample if elements are defined.
        auto cannot be used with 'gui' option.
    weight_fraction: list of string
        Give the composition (weight). If auto, equally parted
    density: list of float
        Set the density. If 'auto', obtain from the compo_at.
    thickness: float
        Set the thickness. If 'auto', look in mp.Sample or set to 100nm
    detector: str
        Give the detector name defined in DTSA-II
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Note
    ----

    For further details on DTSA-II please refer to
    http://www.cstl.nist.gov/div837/837.02/epq/dtsa2/index.html

    """
    from hyperspy import signals
    from hyperspy.misc import material

    if mp == 'gui':
        spec = signals.EDSTEMSpectrum(np.zeros(2048))
        spec.axes_manager[-1].units = 'keV'
        if elements == 'auto':
            raise ValueError('Elements need to be set (set_elements) ' +
                             'with gui option')
            return 0
        else:
            spec.set_microscope_parameters()
            spec.set_elements(elements)
            spec.add_lines()
        mp = spec.metadata
        azimDeg = [mp.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle]
    else:
        if isinstance(mp.Acquisition_instrument.
                      TEM.Detector.EDS.azimuth_angle, list):
            azimDeg = mp.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
        else:
            azimDeg = [
                mp.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle]
        # if len(azimDeg) > 1:
         #   spec = signals.EDSTEMSpectrum(np.zeros([len(azimDeg), 2048]))
        # else:
         #   spec = signals.EDSTEMSpectrum(np.zeros(2048))
        mp = mp.deepcopy()
        #mp = spec.metadata
    # Sample
    if elements == 'auto':
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError('Elements need to be set (set_elements)')
            return 0
    else:
        mp.Sample.elements = elements
    if weight_fraction == 'auto':
        if 'Sample.weight_fraction' in mp:
            weight_fraction = mp.Sample.weight_fraction
        else:
            weight_fraction = []
            for elm in elements:
                weight_fraction.append(1. / len(elements))
            print 'Weight fraction is automatically set to ' + str(weight_fraction)
    #mp.Sample.compo_at = compo_at
    # compo_wt = np.array(
        # material.atomic_to_weight(
            # elements,
            # compo_at)) / 100
    compo_wt = list(weight_fraction)
    if density == 'auto':
        density = material.density_of_mixture_of_pure_elements(
            compo_wt, elements)
    mp.Sample.density = density

    if thickness == 'auto':
        if 'thickness' in mp.Sample:
            thickness = mp.Sample.thickness
        else:
            thickness = 100

    # microscope right units
    e0 = mp.Acquisition_instrument.TEM.beam_energy
    tilt = mp.Acquisition_instrument.TEM.tilt_stage
    ltime = mp.Acquisition_instrument.TEM.Detector.EDS.live_time
    elevation = mp.Acquisition_instrument.TEM.Detector.EDS.elevation_angle
    #TOangle = np.radians(spec.get_take_off_angle())
    TOangle = [take_off_angle(tilt, az,
                              elevation) for az in azimDeg]
    # print TOangle
    #TOangle = [np.radians(TO) for TO in TOangle]
    azim = [np.radians(90 - az) for az in azimDeg]
    tilt = np.radians(tilt)
    elevation = np.radians(elevation - 90)

    if gateway == 'auto':
        gateway = get_link_to_jython()
    spec_dir = get_detector_properties(detector + '0',
                                       gateway=gateway)
    channelMax = len(spec_dir.data)
    WD = spec_dir.metadata.Acquisition_instrument.SEM.\
        Detector.EDS.optimal_working_distance
    channel = gateway.remote_exec("""
        import dtsa2
        import math
        epq = dtsa2.epq
        epu = dtsa2.epu
        nm = dtsa2.nm
        elements = """ + str(elements) + """
        elms = []
        for element in elements:
            elms.append(getattr(dtsa2.epq.Element,element))
        density = """ + str(density) + """
        compo_wt = """ + str(compo_wt) + """
        thickness = """ + str(thickness * 1e-9) + """
        e0 =  """ + str(e0) + """
        dose =  """ + str(dose) + """
        tilt = """ + str(tilt) + """
        channelMax = """ + str(channelMax) + """
        WD = """ + str(WD * 1.0e-3) + """
        live_time = """ + str(ltime) + """
        elevation = """ + str(elevation) + """
        azim = """ + str(azim) + """
        #TOA = """ + str(TOangle) + """
        nTraj = """ + str(nTraj) + """
        print WD
        #Position of detector and sample (WD in km, d-to-crystal in m)
        origin = [0.0,0.0,WD]
        z0 = origin[2]
        det = []
        
        if 1 == 0:
            for j, az in enumerate(azim):
                prop = epq.SpectrumProperties()
                #prop.setDetectorPosition(elevation, az, 0.005, 2e-5)
                WD = 3e-3
                det_sample_dist = 20e-3
                prop.setDetectorPosition(elevation, az, det_sample_dist, WD )
                posi = prop.getArrayProperty(epq.SpectrumProperties.\
                        DetectorPosition)
                #posi = [posi[0]/1000.,posi[1]/1000.,posi[2]/1000.]
                det_name = '""" + detector + """'+str(j)
                det.append(dtsa2.findDetector(det_name))
                prop = det[-1].getDetectorProperties()
                prop.setPosition(posi)
        else:
            for j, az in enumerate(azim):
                det_name = '""" + detector + """'+str(j)
                det.append(dtsa2.findDetector(det_name))
                print det[-1]
                #print det[-1].getProperties()
        el = 0
        if len(elms) == 1:
            mat=epq.MaterialFactory.createPureElement(elms[el])
        else:
            mat = epq.Material(epq.Composition(elms,compo_wt ),
                                    epq.ToSI.gPerCC(density))


        # Create a simulator and initialize it
        monteb = nm.MonteCarloSS()
        monteb.setBeamEnergy(epq.ToSI.keV(e0))

        # film
        center0=epu.Math2.plus(origin,[0.0,0.0,-thickness/2])
        block = nm.MultiPlaneShape.createFilm([0.0,-math.sin(tilt),
                    math.cos(tilt)],
                    origin, thickness)
                    #center0, thickness)
        monteb.addSubRegion(monteb.getChamber(), mat, block)
        # Add event listeners to model characteristic radiation
        
        xrel=[]
        brem = []
        for de in det:
            
            xrel.append(nm.XRayEventListener2(monteb,de))
            monteb.addActionListener(xrel[-1])
            # Add event listeners to model bBremsstrahlung
            brem.append(nm.BremsstrahlungEventListener(monteb,de))
            monteb.addActionListener(brem[-1])
            # Reset the detector
            de.reset()
        #if tilt ==0:
            #ei=nm.EmissionImage.watchDefaultTransitions(
               #xrel[0], 512, 4.0*400.0e-9, origin)
        # run the electrons
        monteb.runMultipleTrajectories(nTraj)
        # Get the spectrum and assign properties
        specb=[]
        propsb=[]
        for j, de in enumerate(det):
            specb.append(de.getSpectrum(dose*1.0e-9 /
                (nTraj * epq.PhysicalConstants.ElectronCharge) ))
            
            #propsb.append(specb[-1].getProperties())
            #propsb[-1].setTextProperty(
                #epq.SpectrumProperties.SpectrumDisplayName,
                                  #"%s std." % (azim[j]))
            #propsb[-1].setNumericProperty(
                #epq.SpectrumProperties.LiveTime, dose)
            #propsb[-1].setNumericProperty(
                #epq.SpectrumProperties.FaradayBegin,1.0)
            #propsb[-1].setNumericProperty(
                #epq.SpectrumProperties.BeamEnergy,e0)
            dtsa2.display(specb[-1])
            #noisyb=epq.SpectrumUtils.addNoiseToSpectrum(
                #specb[-1],live_time)
            #dtsa2.display(noisyb)
            for i in range(channelMax):
                channel.send(specb[-1].getCounts(i))
        #if tilt ==0:
            #nm.EmissionImage.dumpToFiles(ei,
             #"C:\Users\pb565\.hyperspy\DTSA-II\Images_%s" % (str(tilt)))


    """)

    datas = []
    for i, item in enumerate(channel):
        datas.append(item)

    spec = _create_spectrum_from_DTSA_detector(
        datas=np.array(datas).reshape(len(azim), channelMax),
        nTraj=nTraj, mp=mp,
        spec_detector=spec_dir,
        gateway=gateway)

    if len(azim) > 1:
        spec.axes_manager[0].scale = azimDeg[1] - azimDeg[0]
        spec.axes_manager[0].offset = azimDeg[0]
        spec.axes_manager[0].name = 'azimuth'
        spec.axes_manager[0].units = 'Degree'
        spec.metadata.Acquisition_instrument.TEM.Detector\
            .EDS.azimuth_angle = mp.Acquisition_instrument.\
            TEM.Detector.EDS.elevation_angle

    return spec


def get_detector_properties(name, gateway='auto'):
    """
    Get the details properties of a detector.

    Return an efficiency spectrum

    Parameters
    ----------

    name: str
        The name of the detector

    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Return
    ------
    signals.EDSSEMSpectrum containing the efficiency of the detector
    """

    from hyperspy import signals
    if gateway == 'auto':
        gateway = get_link_to_jython()
    channel = gateway.remote_exec("""
        import dtsa2
        det = dtsa2.findDetector('""" + name + """')
        efficiency = det.efficiency
        prop = det.properties
        print det
        #print prop
        channel.send(str(det.name))
        channel.send(str(prop))
        for ef in efficiency:
            channel.send(ef)
    """)
    datas = []
    for i, item in enumerate(channel):
        if i == 0:
            det_name = item
        elif i == 1:
            prop = item
        else:
            datas.append(item)
    print det_name
    spec = signals.EDSSEMSpectrum(datas)
    spec.set_microscope_parameters(
        azimuth_angle=float(prop.split(
            'Azimuthal angle=')[1].split('\xb0')[0]),
        elevation_angle=float(prop.split(
            'Elevation=')[1].split('\xb0')[0]),
        energy_resolution_MnKa=float(prop.split(
            'Resolution=')[1].split(' eV')[0]))
    spec.axes_manager[-1].offset = float(
        prop.split('Energy offset=')[1].split(' eV')[0]) / 1000.
    spec.axes_manager[-1].scale = float(
        prop.split('Energy scale=')[1].split(' eV')[0]) / 1000.
    # Why that?
    spec.axes_manager[-1].offset += spec.axes_manager[-1].scale / 2.
    spec.axes_manager[-1].name = 'Energy'
    spec.axes_manager[-1].units = 'keV'
    spec.metadata.General.title = det_name
    spec.original_metadata.spectrum_properties = prop[19:-2].split(',')

    spec.metadata.Acquisition_instrument.SEM.\
        Detector.EDS.optimal_working_distance = float(prop.split(
            'Optimal working distance=')[1].split(' mm')[0])

    # To find a properties later on...
    #prop_name = 'Optimal working distance'
    # for prop in spec_dir.original_metadata.spectrum_properties:
    #    if prop_name in prop:
    #        WD = float(prop.split(prop_name + '=')[1].split(' mm')[0])

    return spec


def fft_power_spectrum(self):
    """Compute the power spectrum
    """
    self.data = np.abs(self.data)


def fft_mirror_center(self):
    """Translate the center into the middle

    1D,2D,3D
    """
    from hyperspy import utils
    n = self.axes_manager.shape
    n = np.divide(n, 2)
    # dim=len(self.axes_manager.shape)
    tmp = self.deepcopy()
    if len(n) == 1:
        imgn = utils.stack([tmp[n[0]:], tmp[:n[0]]], axis=0)
    elif len(n) == 2:
        x1 = utils.stack([tmp[:n[0], n[1]:], tmp[:n[0], :n[1]]], axis=1)
        x2 = utils.stack([tmp[n[0]:, n[1]:], tmp[n[0]:, :n[1]]], axis=1)
        imgn = utils.stack([x2, x1], axis=0)
    elif len(n) == 3:
        x1 = utils.stack([tmp[:n[0], n[1]:, :n[2]],
                          tmp[:n[0], :n[1], :n[2]]], axis=1)
        x2 = utils.stack([tmp[n[0]:, n[1]:, :n[2]],
                          tmp[n[0]:, :n[1], :n[2]]], axis=1)
        x3 = utils.stack([tmp[:n[0], n[1]:, n[2]:],
                          tmp[:n[0], :n[1], n[2]:]], axis=1)
        x4 = utils.stack([tmp[n[0]:, n[1]:, n[2]:],
                          tmp[n[0]:, :n[1], n[2]:]], axis=1)
        y1 = utils.stack([x3, x1], axis=2)
        y2 = utils.stack([x4, x2], axis=2)
        imgn = utils.stack([y2, y1], axis=0)
    else:
        print 'dimension not supported'

    axes = range(len(self.axes_manager.shape))
    for ax in axes:
        axis = imgn.axes_manager[ax]
        axis.offset = 0
        axis.offset = -axis.high_value / 2.

    return imgn


def fft_rtransform(self, n_dim=1, n_power=1, norm=True):
    """Radial projection for square fft

    Parameters
    ----------

    n_dim: float
        ratio between number of number of bin and fft dimension

    n_power: float
        power law of the bins

    norm:
        divide the content of each bin by the number of pixel

    Example
    ------

    >>> im = signals.Image(random.random([150,100,80]))
    >>> dim = 100
    >>> scale = 0.1
    >>> im_fft = utils_eds.fft(im,shape_fft=[dim]*3,scale=[scale]*3)
    >>> im_fft2 = utils_eds.fft_mirror_center(im_fft)
    >>> utils_eds.fft_power_spectrum(im_fft2)
    >>> bins, ydat = utils_eds.fft_rtransform(im_fft2,norm=True)
    >>> plot(bins,ydat)
    >>> def scalef(x):
    >>>     return dim*scale/x
    >>> yscale('log')
    >>> xscale('log')
    >>> b=[2.0,0.75,0.35,0.15,0.075]
    >>> a = map(scalef,b)
    >>> xticks(a,b)
    >>> ax = gca()


    """
    dim = self.axes_manager.shape
    if dim[0] != dim[1]:
        raise ValueError("All dimension should equal.")
    if len(self.axes_manager.shape) == 3:
        if dim[0] != dim[2]:
            raise ValueError("All dimension should equal.")

    dim = dim[0]
    # scaling=self.axes_manager[0].scale
    if len(self.axes_manager.shape) == 2:

        part1 = np.ones((dim, dim)) * \
            np.power(plt.mlab.frange(-(dim) / 2 + 0.5, (dim) / 2 - 0.5), 2)
        dist_mat = np.power(part1 + part1.T, 0.5)
        bins = np.power(
            plt.mlab.frange(0., np.power(dist_mat[-1, -1], 1 / n_power),
                            np.power(dist_mat[-1, -1], 1 / n_power) / dim * 1.5 / n_dim), n_power)
    elif len(self.axes_manager.shape) == 3:
        part1 = np.ones((dim, dim, dim)) * \
            np.power(plt.mlab.frange(-(dim) / 2 + 0.5, (dim) / 2 - 0.5), 2)
        dist_mat = np.power(
            part1 + part1.T + np.array(map(np.transpose, part1)), 0.5)
        # dist_mat/=scaling
        bins = np.power(
            plt.mlab.frange(0., np.power(dist_mat[-1, -1, -1], 1 / n_power),
                            np.power(dist_mat[-1, -1, -1], 1 / n_power) / dim * 1.5 / n_dim), n_power)
        # bins/=scaling
    ydat = []
    for i in range(len(bins) - 1):
        mask_tmp = (dist_mat < bins[i + 1]) * (dist_mat > bins[i])
        tmp = self.data[mask_tmp]
        if norm:
            nonzero = np.count_nonzero(tmp)
            if nonzero == 0:
                ydat.append(sum(tmp))
            else:
                ydat.append(sum(tmp) / np.count_nonzero(tmp))
        else:
            ydat.append(sum(tmp))

    # def scalef(x):
    #    return dim/self.axes_manager[0].scale/x

    #bins = map(scalef,bins)
    # bins/=self.axes_manager[0].scale
    return bins[:-1], ydat


def fft_ifft(self, s=None, axes=None):
    """
    Compute the inverse discrete Fourier Transform.

    This function computes the inverse of the discrete
    Fourier Transform over any number of axes in an M-dimensional array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(a)) == a`` to within numerical accuracy.
    For a description of the definitions and conventions used, see `numpy.fft`.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e. it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------

    s : int or sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input (along the axes specified
        by `axes`) is used.  See notes for issue on `ifft` zero padding.
    axes : int or sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the inverse transform over that
        axis is performed multiple times.

    Return
    ------
    signals.Signal

    Notes
    -----
    For further information see the documentation of numpy.fft.ifft,
    numpy.fft.ifft2 or numpy.fft.ifftn

    """

    from hyperspy.signals import Signal

    dim = len(self.axes_manager.shape)
    if dim == 1:
        if axes is None:
            axis = -1
        im_ifft = Signal(np.fft.ifft(self.data, n=s, axis=axis).real)
    elif dim == 2:
        if axes is None:
            axes = (-2, -1)
        im_ifft = Signal(np.fft.ifft2(self.data, s=s, axes=axes).real)
    else:
        im_ifft = Signal(np.fft.ifftn(self.data, s=s, axes=axes).real)

    if self.axes_manager.signal_dimension == 2:
        im_ifft.axes_manager.set_signal_dimension(2)
    # scale,, to be verified
    for i in range(dim):
        im_ifft.axes_manager[i].scale = 1 / self.axes_manager[i].scale

    return im_ifft


def fft(self, shape_fft=None, axes=None):
    """Compute the discrete Fourier Transform.

    This function computes the discrete Fourier Transform over
    any number of axes in an *M*-dimensional array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    shape_fft : int or sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (`s[0]` refers to axis 0, `s[1]` to axis 1, etc.).
        This corresponds to `n` for `fft(x, n)`.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped.  If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input (along the axes specified
        by `axes`) is used.
    axes : int or sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
        Repeated indices in `axes` means that the transform over that axis is
        performed multiple times.

    Return
    ------
    signals.Signal

    Notes
    -----
    For further information see the documentation of numpy.fft.fft,
    numpy.fft.fft2 or numpy.fft.fftn
    """

    from hyperspy.signals import Signal, Spectrum, Image

    #dim = len(self.axes_manager.shape)
    # if dim == 1:
    # if axes is None:
    #axis = -1
    #im_fft = Signal(np.fft.fft(self.data, n=shape_fft, axis=axis))
    # elif dim == 2:
    # if axes is None:
    #axes = (-2, -1)
    #im_fft = Signal(np.fft.fft2(self.data, s=shape_fft, axes=axes))
    # else:
    # if axes is None:
    #axes = range(-dim,0)
    #im_fft = Signal(np.fft.fftn(self.data, s=shape_fft, axes=axes))
    if self.axes_manager.signal_dimension == 2:
        im_fft = Image(np.fft.fftn(self.data, s=shape_fft, axes=axes))
    else:
        im_fft = Spectrum(np.fft.fftn(self.data, s=shape_fft, axes=axes))
    if axes is None:
        axes = range(len(self.axes_manager.shape))
    if shape_fft is None:
        shape_fft = self.axes_manager.shape
    # if self.axes_manager.signal_dimension == 2:
        # im_fft.axes_manager.set_signal_dimension(2)
        #im_fft = im_fft.as_image([-2,-1])
    # else:
    #    im_fft = im_fft.as_spectrum(-1)
    # scale, to be verified

    for ax, dim in zip(axes, shape_fft):
        #im_fft.axes_manager[i].scale = scale[i]
        axis = im_fft.axes_manager[ax]
        axis.scale = 1. / dim / self.axes_manager[ax].scale
        #axis.name= 'Spatial frequency'
        axis.units = str(self.axes_manager[ax].units) + '$^{-1}$'
        axis.offset = 0
        axis.offset = -axis.high_value / 2.

    return im_fft


def simulate_model(elements=None,
                   shape_spectrum=None,
                   beam_energy=None,
                   live_time=None,
                   weight_percents=None,
                   energy_resolution_MnKa=None,
                   counts_rate=None,
                   elemental_map='random'):
    """Simulate a model with default param defined

    See database_1Dspec()
    """
    from hyperspy import signals

    spec = signals.EDSSEMSpectrum(np.zeros(1024))
    s = database.spec1D()

    if elements is not None:
        s.set_elements(elements)
    else:
        elements = s.metadata.Sample.elements
    if weight_percents is not None:
        s.metadata.Sample.weight_percents = weight_percents

    if counts_rate is not None:
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.counts_rate = counts_rate

    s.set_microscope_parameters(beam_energy=beam_energy,
                                live_time=live_time,
                                energy_resolution_MnKa=energy_resolution_MnKa)
    if shape_spectrum is not None:
        smap = signals.EDSSEMSpectrum(np.zeros(list(shape_spectrum)))
        smap.get_calibration_from(s)
        smap.set_elements(elements)
        smap.simulate_model(elemental_map=elemental_map)
        return smap
    else:
        model = s.simulate_model(elemental_map=elemental_map)
        return model


def get_kfactors(xray_lines,
                 beam_energy,
                 detector_efficiency=None,
                 gateway='auto'):
    """Calculate the kfactors cofficient of Cliff-Lorimer method from first
    principle

    Parameters
    ----------
    xray_lines: list of string
        The X-ray lines, e.g. ['Al_Ka', 'Zn_Ka']
    beam_energy: float
        The energy of the beam in kV.
    detector_efficiency: {signals.EDSSEMSpectrum,None}
        A spectrum containing the detector efficiency. If None, set the
        efficiency to one.
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    """
    from hyperspy.misc.eds import epq_database
    if gateway == 'auto':
        gateway = get_link_to_jython()
    kab = []
    for xray_line in xray_lines:
        xray_prop = epq_database.get_xray_transition_properties(
            xray_line, beam_energy, gateway=gateway)
        xray_prop = reduce(lambda x, y: x * y, xray_prop)
        element, line = _get_element_and_line(xray_line)
        A = elements_db[element]\
            ['General_properties']['atomic_weight']
        if detector_efficiency is None:
            kab.append(xray_prop / A)
        else:
            line_energy = detector_efficiency._get_line_energy(xray_line)
            kab.append(xray_prop / A *
                       detector_efficiency[line_energy].data[0])
    #kab = kab[0] / kab[1]
    if len(xray_lines) == 1:
        return kab[0]
    else:
        kab = kab[1] / kab[0]
        return kab

# def absorption_correction_factor_for_thin_film(mac_sample,
        # density,
        # thickness,
        # TOA):
    #"""
    # Compute the absorption corrections factor (ACF) for a thin film

    # Return the ACF for each X-ray-lines

    # Parameters
    #----------
    # mac_sample:
        # The mass absorption cofficients, in cm^2/g
    # density: float
        # Set the density. in g/cm^3
    # thickness: float
        # Set the thickness in nm
    # TOA: float
        # the take of angle
    #"""
    #rt =  density * thickness * 1e-7 / np.sin(np.radians(TOA))
    # abs_corr=[]
    # for mac in mac_sample:
        #fact = mac*rt
        # abs_corr.append(np.nan_to_num((1-np.exp(-(fact)))/fact))
    # return abs_corr


def quantification_absorption_corrections_thin_film(intensities,
                                                    elements,
                                                    xray_lines,
                                                    kfactors,
                                                    TOA,
                                                    thickness,
                                                    max_iter=50,
                                                    atol=1e-3,
                                                    all_data=False):
    """
    Quantification with absorption correction

    Based on Cliff-Lorimer

    Parameters
    ----------
    intensities: list of signal
        List of intensities
    elements: list of str
        List of elements, e.g. ['Al', 'Zn']
    xray_lines: list of string
        The X-ray lines, e.g. ['Al_Ka', 'Zn_Ka']
    kfactors: list of float
        The list of kfactor, compared to the first
        elements. eg. kfactors = [1.2, 2.5]
        for kfactors_name = ['Al_Ka/Cu_Ka', 'Al_Ka/Nb_Ka']
    thickness: float
        Set the thickness in nm
    TOA: float
        the take of angle
    thickness: float
        Set the thickness in nm
    max_iter: int
        The maximum of iteration
    atol:
        The tolerance factor for the conve

    Return
    ------
    The weight fractions for each step of the iteration
    """
    from hyperspy.misc import material

    xray_energy = [
        _get_energy_xray_line(xray_line) for xray_line in xray_lines]
    weight_fractions = [quantification_cliff_lorimer(
                        intensities=intensities, kfactors=kfactors)]
    kfactors_abs = []
    for j in range(max_iter):
        density = material.density_of_mixture_of_pure_elements(
             weight_fractions[-1], elements)
        # mac_sample = material.mass_absorption_coefficient_of_mixture_of_pure_elements(
        # energies=xray_lines,
        # weight_fraction=weight_fractions[-1],
        # elements=elements)
        abs_corr = physical_model.xray_absorption_thin_film(energy=xray_energy,
                                                            weight_fraction=weight_fractions[
                                                                -1],
                                                            elements=elements,
                                                            density=density,
                                                            thickness=thickness,
                                                            TOA=TOA)
        kfactors_abs = [abs_corr[0] / abs_corr[i + 1] * kab
                        for i, kab in enumerate(kfactors)]

        weight_fractions.append(quantification_cliff_lorimer(
            intensities=intensities,
            kfactors=kfactors_abs))
        if j > 0:
            dif = sum(np.abs(weight_fractions[-1] -
                             weight_fractions[-2]))
            if dif < atol:
                break
    # if j==max_iter-1:
        # print "No convergence in the limit of iteration"
    # else:
        # print 'Convergence after %s iterations' % j
    if all_data:
        return weight_fractions
    else:
        return weight_fractions[-1]

############################
# def animate_legend(figure='last'):
    #"""Animate the legend of a figure

    # Parameters
    #---------

    # figure: 'last' | matplolib.figure
        # If 'last' pick the last figure
    #"""
    # if figure=='last':
        #fig = plt.gcf()
        #ax= plt.gca()
    # else:
        #ax = fig.axes[0]
    #lines = ax.lines
    #lined = dict()
    # leg=ax.get_legend()
    # for legline, origline in zip(leg.get_lines(), lines):
        # legline.set_picker(5)  # 5 pts tolerance
        #lined[legline] = origline
    # def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        #legline = event.artist
        #origline = lined[legline]
        #vis = not origline.get_visible()
        # origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        # if vis:
        # legline.set_alpha(1.0)
        # else:
        # legline.set_alpha(0.2)
        # fig.canvas.draw()

    #fig.canvas.mpl_connect('pick_event', onpick)

    # plt.show()
############################
# def plot_3D_iso_surface(self,threshold,
        #color = 'auto',
        # figure='new',
        # scale='auto'):
        # must be the main function in Image, and here jsut to connect with result
        #"""
        # Generate an iso-surface in Mayavi.

        # Parameters
        #----------

        # threshold: float
        # Between 0 (min intensity) and 1 (max intensity).
        # If result == quant, 1 == 100%.

        # color: list
        # The color of the surface, (R,G,B). If 'auto', automatically
        # selected.

        # figure: str
        # If 'new', generate a new scene/figure. Else, use the old one.

        # scale: str || list
        # If 'auto', scale with axes_manager.scale. Else, scale with
        # the given list (x,y,z).

        # Return
        #------

        # figure: mayavi.core.scene.Scene

        # src: mayavi.sources.array_source.ArraySource

        # iso: mayavi.modules.iso_surface.IsoSurface

        #"""
        #from mayavi import mlab

        # if figure=='new':
        #figure = mlab.figure()

        #img_res = self.deepcopy()

        #img_data = img_res.data
        #img_data = np.rollaxis(img_data,0,3)
        #img_data = np.rollaxis(img_data,0,2)
        #src = mlab.pipeline.scalar_field(img_data)
        #src.name = img_res.metadata.General.title

        # if 'intensities' == result or isinstance(result,str) is False:

        #threshold = img_data.max()-threshold*img_data.ptp()

        # if scale=='auto':
        #scale = []
        # for i in [1,2,0]:
        # scale.append(img_res.axes_manager[i].scale)
        #src.spacing= scale
        # else:
        #src.spacing = scale
        # if color != 'auto':
        # iso = mlab.pipeline.iso_surface(src,
        # contours=[threshold, ],color =color)
        # else:
        # iso = mlab.pipeline.iso_surface(src,
        # contours=[threshold, ])

        #iso.compute_normals = False
        # if color != 'auto':
        ##   iso.actor.property.color = color
        ##iso.actor.property.opacity = 0.5
        # return figure, src, iso
############################
# def compare_histograms(imgs,
        # bins='freedman',
        # color=None,
        # line_style=None,
        # legend='auto',
        # fig=None,
        # range_bins=None,
        #**kwargs):
    #"""Compare the histogram of the list of image
    # Parameters
    #----------
    # bins : int or list or str (optional)
        # If bins is a string, then it must be one of:
        #'knuth' : use Knuth's rule to determine bins
        #'scotts' : use Scott's rule to determine bins
        #'freedman' : use the Freedman-diaconis rule to determine bins
        #'blocks' : use bayesian blocks for dynamic bin widths
    # color : valid matplotlib color or a list of them or `None`
        # Sets the color of the lines of the plots when `style` is "cascade"
        # or "mosaic". If a list, if its length is
        # less than the number of spectra to plot, the colors will be cycled. If
        # If `None`, use default matplotlib color cycle.
    # line_style: valid matplotlib line style or a list of them or `None`
        # The main line style are '-','--','steps','-.',':'.
        # If a list, if its length is less than the number of
        # spectra to plot, line_style will be cycled. If
        # If `None`, use 'steps'.
    # legend: None | list of str | 'auto'
        # If list of string, legend for "cascade" or title for "mosaic" is
        # displayed. If 'auto', the title of each spectra (metadata.General.title)
        # is used.
    # fig : {matplotlib figure, None}
        # If None, a default figure will be created.
    #"""
    #hists = []
    # for img in imgs:
        # hists.append(img.get_histogram(bins=bins,
        # range_bins=range_bins, **kwargs))
    # if line_style is None:
        #line_style = 'steps'
    # return hyperspy.utils.plot.plot_spectra(
        # hists, style='overlap', color=color,
        # line_style=line_style, legend=legend, fig=fig)
############################
# def compare_signal(specs,
        # indexes=None,
        # legend_labels='auto',
        # colors='auto',
        # line_styles='auto'):
    #"""Compare the signal from different indexes or|and from different
    # spectra.
    # Parameters
    #----------
    # specs: list | spectrum
        # A list of spectra or a spectrum
    # indexes: list | None
        # The list of indexes to compares. If None, specs is a list of
        # 1D spectra that are ploted together
    # legend_labels: 'auto' | list | None
        # If legend_labels is auto, then the indexes are used.
    # colors: list
        # If 'auto', automatically selected, eg: ('red','blue')
    # line_styles: list
        # If 'auto', continuous lines, eg: ('-','--','steps','-.',':')
    # Returns
    #-------
    # figure
    #"""
    # print "obsolete, should use utils.plot.plot_spectra"
    # if indexes is None:
        #nb_signals = len(specs)
    # elif isinstance(indexes[0], list) is False and isinstance(indexes[0], tuple) is False:
        #nb_signals = len(specs)
        #indexes = [indexes] * nb_signals
    # else:
        #nb_signals = len(indexes)
    # if colors == 'auto':
        # colors = ['red', 'blue', 'green', 'orange', 'violet', 'magenta',
        #'cyan', 'violet', 'black', 'yellow', 'pink']
        #colors += colors
        #colors += colors
    # elif isinstance(colors, list) is False:
        #colors = [colors] * nb_signals
    # if line_styles == 'auto':
        #line_styles = ['-'] * nb_signals
    # elif isinstance(line_styles, list) is False:
        #line_styles = [line_styles] * nb_signals
    #fig = plt.figure()
    # if legend_labels == 'auto':
        #legend_labels = []
        # if isinstance(specs, list) or isinstance(specs, tuple):
        # for spec in specs:
        # legend_labels.append(spec.metadata.General.title)
        # else:
        # for index in indexes:
        # legend_labels.append(str(index))
    # for i, index in enumerate(indexes):
    # for i in range(nb_signals):
        # if isinstance(specs, list) or isinstance(specs, tuple):
        #tmp = specs[i]
        # else:
        #tmp = specs
        # if indexes is not None:
        # for ind in indexes[i]:
        #tmp = tmp[ind]
        # maxx = (len(tmp.data) - 1) * \
        #tmp.axes_manager[0].scale + tmp.axes_manager[0].offset
        # xdata = mlab.frange(tmp.axes_manager[0].offset, maxx,
        # tmp.axes_manager[0].scale, npts=len(tmp.data))
        #plt.plot(xdata, tmp.data, color=colors[i], ls=line_styles[i])
    # plt.ylabel('Intensity')
    # plt.xlabel(str(tmp.axes_manager[0].name) +
        #' (' + str(tmp.axes_manager[0].units) + ')')
    # if legend_labels is not None:
        # plt.legend(legend_labels)
    # fig.show()
    # return fig


# def get_MAC_sample(xray_lines, weight_fraction, elements='auto'):
    #"""Return the mass absorption coefficients of a sample

    # Parameters
    #----------
    # xray_lines: list of str
        # The list of X-ray lines, e.g. ['Al_Ka','Zn_Ka','Zn_La']
    # weight_fraction: list of float
        # the composition of the sample
    # elements: {list of str | 'auto'}
        # The list of element symbol of the absorber, e.g. ['Al','Zn'].
        # if 'auto', use the elements of the X-ray lines

    # Return
    #------
    # mass absorption coefficient in cm^2/g
    #"""
    #from hyperspy import utils
    #macs = []
    # if elements == 'auto':
        #elements = []
        # for xray_line in xray_lines:
        #element, line = _get_element_and_line(xray_line)
        # elements.append(element)
        #elements = set(elements)
    # if len(elements) != len(weight_fraction):
        #raise ValueError("Add elements first, see 'set_elements'")

    # macs = utils.get_mass_absorption_coefficient_sample(
        # energies=xray_lines,
        # elements=elements,
        # weight_fraction=weight_fraction)
    # for xray_line in xray_lines:
        ##line_energy = _get_energy_xray_line(xray_line)
        ###el_emit, line = _get_element_and_line(xray_line)
        # macs.append(utils.get_mass_absorption_coefficient_sample(
        # energy=line_energy,
        # elements=elements,
        # weight_fraction=weight_fraction))
        # for i_el, el_abs in enumerate(elements):
        # macs[-1] += weight_percent[i_el] / 100 * \
        # utils.get_mass_absorption_coefficient_xray_line(el_abs,xray_line)
        # MAC[el_emit][line][el_abs]
    # return macs

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


# def get_mass_absorption_coefficient(element, energy):
    #"""
    # Get the mass absorption coefficient of an Xray

    # Parameters
    #----------
    # element: str
        # The element symbol of the absorber, e.g. 'Al'.
    # energy: float
        # The energy of the Xray in keV

    # Return
    #------
    # mass absorption coefficient in cm^2/g
    #"""
    #from hyperspy.misc.eds.ffast_mac import ffast_mac_db as ffast_mac
    #energies = ffast_mac[element].energies_keV

    # for index, energy_db in enumerate(energies):
        # if energy <= energy_db:
        # break
    # if index1 == len(energies):
    # print 'extrapolation'
    # print element
    # print energy
    # mac = ffast_mac[element].mass_absorption_coefficient_cm2g[index]
    # mac1 = ffast_mac[element].mass_absorption_coefficient_cm2g[index - 1]
    # energy_db = ffast_mac[element].energies_keV[index]
    # energy_db1 = ffast_mac[element].energies_keV[index - 1]
    # if energy == energy_db or energy_db1 == 0:
        # return mac
    # else:
        # return _mac_interpolation(mac, mac1, energy,
        # energy_db, energy_db1)

def quantification_cliff_lorimer(intensities,
                                 kfactors,
                                 mask=None,
                                 min_intensity=0.1):
    """
    Quantification using Cliff-Lorimer

    Parameters
    ----------
    intensities: numpy.array
        the intensities for each X-ray lines. The first axis should be the
        elements axis.
    kfactors: list of float
        The list of kfactor in same order as intensities eg. kfactors =
        [1, 1.47, 1.72] for ['Al_Ka','Cr_Ka', 'Ni_Ka']
    mask: array of bool
        The mask with the dimension of intensities[0]. If a pixel is True,
        the composition is set to zero.

    Return
    ------
    numpy.array containing the weight fraction with the same
    shape as intensities.
    """
    # Value used as an threshold to prevent using zeros as denominator
    
    dim = intensities.shape
    if len(dim) > 1:
        dim2 = reduce(lambda x, y: x * y, dim[1:])
        intens = intensities.reshape(dim[0], dim2)
        intens = intens.astype('float')
        for i in range(dim2):
            index = np.where(intens[:, i] > min_intensity)[0]
            if len(index) > 1:
                ref_index, ref_index2 = index[:2]
                intens[:, i] = _quantification_cliff_lorimer(
                    intens[:, i], kfactors, ref_index, ref_index2)
            else:
                intens[:, i] = np.zeros_like(intens[:, i])
                if len(index) == 1:
                    intens[index[0], i] = 1.
        intens = intens.reshape(dim)
        if mask is not None:
            for i in range(dim[0]):
                intens[i][mask] = 0
        return intens
    else:
        # intens = intensities.copy()
        # intens = intens.astype('float')
        index = np.where(intensities > min_intensity)[0]
        if len(index) > 1:
            ref_index, ref_index2 = index[:2]
            intens = _quantification_cliff_lorimer(
                intensities, kfactors, ref_index, ref_index2)
        else:
            intens = np.zeros_like(intensities)
            if len(index) == 1:
                intens[index[0]] = 1.
        return intens


def _quantification_cliff_lorimer(intensities,
                                  kfactors,
                                  ref_index=0,
                                  ref_index2=1):
    """
    Quantification using Cliff-Lorimer

    Parameters
    ----------
    intensities: numpy.array
        the intensities for each X-ray lines. The first axis should be the
        elements axis.
    kfactors: list of float
        The list of kfactor in same order as  intensities eg. kfactors =
        [1, 1.47, 1.72] for ['Al_Ka','Cr_Ka', 'Ni_Ka']
    ref_index, ref_index2: int
        index of the elements that will be in the denominator. Should be non
        zeros if possible.

    Return
    ------
    numpy.array containing the weight fraction with the same
    shape as intensities.
    """
    if len(intensities) != len(kfactors):
        raise ValueError('The number of kfactors must match the size of the '
                         'first axis of intensities.')
    ab = np.zeros_like(intensities, dtype='float')
    composition = np.ones_like(intensities, dtype='float')
    # ab = Ia/Ib / kab

    other_index = range(len(kfactors))
    other_index.pop(ref_index)
    for i in other_index:
        ab[i] = intensities[ref_index] * kfactors[ref_index]  \
            / intensities[i] / kfactors[i]
    # Ca = ab /(1 + ab + ab/ac + ab/ad + ...)
    for i in other_index:
        if i == ref_index2:
            composition[ref_index] += ab[ref_index2]
        else:
            composition[ref_index] += (ab[ref_index2] / ab[i])
    composition[ref_index] = ab[ref_index2] / composition[ref_index]
    # Cb = Ca / ab
    for i in other_index:
        composition[i] = composition[ref_index] / ab[i]
    return composition


def quantification_zeta_factor(intensities,
                               zfactors,
                               dose):
    """
    Quantification using zeta-factor method

    Parameters
    ----------
    intensities: numpy.array
        the intensities for each X-ray lines. The first axis should be the
        elements axis.
    zfactors: list of float
        The list of kfactor in same order as  intensities eg. zfactors =
        [1, 1.47, 1.72] for ['Al_Ka','Cr_Ka', 'Ni_Ka']
    dose: float
        the dose given by i*t*N, i the current, t the acquisition time, and N
        the number of electron by unit electric charge.

    Return
    ------
    numpy.array containing the weight fraction with the same
    shape as intensities and mass thickness in kg/m^2.
    """

    sumzi = np.zeros_like(intensities[0], dtype='float')
    composition = np.zeros_like(intensities, dtype='float')
    for intensity, zfactor in zip(intensities, zfactors):
        sumzi = sumzi + intensity * zfactor
    for i, (intensity, zfactor) in enumerate(zip(intensities, zfactors)):
        composition[i] = intensity * zfactor / sumzi
    mass_thickness = sumzi / dose
    return composition, mass_thickness


def detetector_efficiency_from_layers(energies,
                                      elements,
                                      thicknesses_layer,
                                      thickness_detector,
                                      cutoff_energy=0.05):
    """Compute the detector efficiency from the layers.

    The efficiency is calculated by estimating the absorption of the
    different the layers in front of the detector.

    Parameters
    ----------
    energy: float or list of float
        The energy of the  X-ray reaching the detector in keV.
    elements: list of str
        The composition of each layer. One element per layer.
    thicknesses_layer: list of float
        The thickness of each layer in nm
    thickness_detector: float
        The thickness of the detector in mm
    cutoff_energy: float
        The lower energy limit in keV below which the detector has no
        efficiency.

    Return
    ------
    An EDSspectrum instance. 1. is a totaly efficient detector

    Notes
    -----
    Equation adapted from  Alvisi et al 2006
    """
    from hyperspy import utils
    efficiency = np.ones_like(energies)

    for element, thickness in zip(elements,
                                  thicknesses_layer):
        macs = np.array(utils.material.mass_absorption_coefficient(
            energies=energies,
            element=element))
        density = utils.material.elements[element]\
            .Physical_properties.density_gcm3
        efficiency *= np.nan_to_num(np.exp(-(
            macs * density * thickness * 1e-7)))
    macs = np.array(utils.material.mass_absorption_coefficient(
        energies=energies,
        element='Si'))
    density = utils.material.elements.Si\
        .Physical_properties.density_gcm3
    efficiency *= (1 - np.nan_to_num(np.exp(-(macs * density *
                                              thickness_detector * 1e-1))))
    efficiency[energies < cutoff_energy] = 0.0
    return efficiency
