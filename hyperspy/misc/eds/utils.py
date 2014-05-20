import numpy as np
import math
import numpy as np
import execnet
import os
import copy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


import hyperspy.utils
from hyperspy.misc.config_dir import config_path
import hyperspy.components as components
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds.MAC import MAC_db as MAC
from functools import reduce


def _get_element_and_line(xray_line):
    lim = xray_line.find('_')
    return xray_line[:lim], xray_line[lim + 1:]


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
    E_ref = elements_db['Mn']['Atomic_properties']['Xray_lines'][
        'Ka']['energy (keV)']

    FWHM_e = 2.5 * (E - E_ref) * 1000 + FWHM_ref * FWHM_ref

    return math.sqrt(FWHM_e) / 1000  # In mrad


def xray_range(xray_line, beam_energy, density='auto'):
    '''Return the Anderson-Hasler X-ray range.

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

    Notes
    -----
    From Anderson, C.A. and M.F. Hasler (1966). In proceedings of the
    4th international conference on X-ray optics and microanalysis.

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 286

    '''

    element, line = _get_element_and_line(xray_line)
    if density == 'auto':
        density = elements_db[
            element][
            'Physical_properties'][
            'density (g/cm^3)']
    Xray_energy = elements_db[
        element]['Atomic_properties']['Xray_lines'][line]['energy (keV)']

    return 0.064 / density * (np.power(beam_energy, 1.68) -
                              np.power(Xray_energy, 1.68))


def electron_range(element, beam_energy, density='auto', tilt=0):
    '''Return the Kanaya-Okayama electron range.

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

    Notes
    -----
    From Kanaya, K. and S. Okayama (1972). J. Phys. D. Appl. Phys. 5, p43

    See also the textbook of Goldstein et al., Plenum publisher,
    third edition p 72.

    '''

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
        The tilt of the stage in degrees. The sample is facing the detector when
        positively tilted.
    azimuth_angle: float
        The azimuth of the detector in degrees. 0 is perpendicular to the tilt
        axis.
    elevation_angle: float
        The elevation of the detector in degrees.

    Returns
    -------
    take_off_angle: float.
        In degrees.

    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)

    """

    a = math.radians(90 + tilt_stage)
    b = math.radians(azimuth_angle)
    c = math.radians(elevation_angle)

    return math.degrees(np.arcsin(-math.cos(a) * math.cos(b) * math.cos(c)
                                  + math.sin(a) * math.sin(c)))


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
    if axis_name_in_mp == True:
        axis_names = self.metadata.Sample[axis_names]

    for i, name in enumerate(axis_names):
        if name == index_name:
            return i


def get_MAC_sample(xray_lines, weight_percent, elements='auto'):
    """Return the mass absorption coefficients of a sample

    Parameters
    ----------
    xray_lines: list of str
        The list of X-ray lines, e.g. ['Al_Ka','Zn_Ka','Zn_La']
    weight_percent: list of float
        the composition of the sample
    elements: {list of str | 'auto'}
        The list of element symbol of the absorber, e.g. ['Al','Zn'].
        if 'auto', use the elements of the X-ray lines
    """
    macs = []
    if elements == 'auto':
        elements = []
        for xray_line in xray_lines:
            element, line = _get_element_and_line(xray_line)
            elements.append(element)
        elements = set(elements)
    if len(elements) != len(weight_percent):
        raise ValueError("Add elements first, see 'set_elements'")
    for xray_line in xray_lines:
        el_emit, line = _get_element_and_line(xray_line)
        macs.append(0)
        for i_el, el_abs in enumerate(elements):
            macs[-1] += weight_percent[i_el] / 100 * MAC[el_emit][line][el_abs]
    return macs


def phase_inspector(self, bins=[20, 20, 20], plot_result=True):
    # to be further improved.
    """
    Generate an binary image of different channel

    Parameters
    ----------

    self: list of 3 images
    """
    from hyperspy import utils
    bins = [20, 20, 20]
    minmax = []

    # generate the bins
    for s in self:
        minmax.append([s.data.min(), s.data.max()])
    center = []
    for i, mm in enumerate(minmax):
        temp = list(mlab.frange(mm[0], mm[1], (mm[1] - mm[0]) / bins[i]))
        temp[-1] += 1
        center.append(temp)

    # calculate the Binary images
    dataBin = []
    if len(self) == 1:
        for x in range(bins[0]):
            temp = self[0].deepcopy()
            dataBin.append(temp)
            dataBin[x].data = ((temp.data >= center[0][x]) *
                               (temp.data < center[0][x + 1])).astype('int')
    elif len(self) == 2:
        for x in range(bins[0]):
            dataBin.append([])
            for y in range(bins[1]):
                temp = self[0].deepcopy()
                temp.data = np.ones_like(temp.data)
                dataBin[-1].append(temp)
                a = [x, y]
                for i, s in enumerate(self):
                    dataBin[x][y].data *= ((s.data >= center[i][a[i]]) *
                                           (s.data < center[i][a[i] + 1])).astype('int')
            dataBin[x] = utils.stack(dataBin[x])
    elif len(self) == 3:
        for x in range(bins[0]):
            dataBin.append([])
            for y in range(bins[1]):
                dataBin[x].append([])
                for z in range(bins[2]):
                    temp = self[0].deepcopy()
                    temp.data = np.ones_like(temp.data)
                    dataBin[-1][-1].append(temp)
                    a = [x, y, z]
                    for i, s in enumerate(self):
                        dataBin[x][y][z].data *= ((s.data >=
                                                   center[i][a[i]]) * (s.data <
                                                                       center[i][a[i] + 1])).astype('int')
                dataBin[x][y] = utils.stack(dataBin[x][y])
            dataBin[x] = utils.stack(dataBin[x])
    img = utils.stack(dataBin)

    for i in range(len(self)):
        img.axes_manager[i].name = self[i].metadata.General.title
        img.axes_manager[i].scale = (minmax[i][1] - minmax[i][0]) / bins[i]
        img.axes_manager[i].offest = minmax[i][0]
        img.axes_manager[i].units = '-'
    img.get_dimensions_from_data()
    return img


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
    from hyperspy import utils

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
        utils.material.atomic_to_weight(
            elements,
            compo_at)) / 100
    compo_wt = list(compo_wt)
    if density == 'auto':
        density = utils.material.density_of_mixture_of_pure_elements(
            elements,
            compo_wt)
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
    from hyperspy import utils
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

    compo_wt = np.array(
        utils.material.atomic_to_weight(
            elements,
            compo_at)) / 100
    compo_wt = list(compo_wt)
    if density == 'auto':
        density = utils.material.density_of_mixture_of_pure_elements(
            elements,
            compo_wt)

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
        "popen//python=C:\Users\pb565\Documents\Java\Jython2.7b\jython.bat")


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


def align_with_stackReg(img,
                        starting_slice=0,
                        align_img=False,
                        return_align_img=False,
                        gateway='auto'):
    # must be in Image
    """Align a stack of images with stackReg from Imagej.

    store the shifts in metadata.align.shifts

    Parameters
    ----------
    img: signal.Image
        The image to align.
    starting_slice: int
        The starting slice for the alignment.
    align_img:
        If True, align stack of images (align2D).
    return_align_img:
        If True, return the align stack as done by imageI.
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    See also
    --------
    align2D

    Notes
    -----
    Defined by P. Thevenaz, U. Ruttimann, and M. Unser,
    IEEE Transaction on IMage Processing 7(1), pp 27-41 (1998)

    The version of MulitStackReg has been modified. Translation and save
    save the alignement is used.

    """
    import time
    from hyperspy.io import load
    path_align_file = os.path.join(config_path, "imageJ\\TransfoMatrix.txt")
    path_img = os.path.join(config_path, "imageJ\\tmp.tiff")
    path_img_alnd = os.path.join(config_path, "imageJ\\tmp_alnd.tiff")

    if os.path.exists(path_img_alnd):
        os.remove(path_img_alnd)

    if os.path.exists(path_align_file):
        os.remove(path_align_file)

    if img.data.dtype == 'float64':
        imgtemp = img.deepcopy()
        imgtemp.change_dtype('float32')
        imgtemp.save(path_img, overwrite=True)
    else:
        img.save(path_img, overwrite=True)

    for i in range(100):
        if os.path.exists(path_img):
            break
        else:
            time.sleep(0.5)

    if gateway == 'auto':
        gateway = get_link_to_jython()
    channel = gateway.remote_exec("""
        import ij.IJ as IJ
        import ij.gui
        path_img = """ + str([path_img]) + """
        path_img_alnd =  """ + str([path_img_alnd]) + """
        imp = IJ.openImage(path_img[0])

        imp.show()
        imp.setSlice(""" + str(starting_slice) + """+1)
        IJ.runPlugIn(imp, "MultiStackReg_", "")

        return_align_img=""" + str(return_align_img) + """
        if return_align_img:
            IJ.saveAs(imp,"Tiff",path_img_alnd[0])
        imp.close()
        channel.send(1)

    """)
    for i, item in enumerate(channel):
        item = item

    shifts = _read_alignement_file()
    mp = img.metadata
    if mp.has_item('align') is False:
        mp.add_node('align')
    mp.align.crop = False
    mp.align.method = 'StackReg'
    mp.align.shifts = shifts

    if align_img:
        img.align2D(shifts=shifts)
        mp.align.is_aligned = True
    else:
        mp.align.is_aligned = False

    if return_align_img:
        for i in range(100):
            if os.path.exists(path_img_alnd):
                imgTemp = load(path_img_alnd)
                break
            else:
                time.sleep(0.5)

        data_align = imgTemp.data
        imgTemp = img.deepcopy()
        imgTemp.data = data_align
        return imgTemp


def _read_alignement_file(path_align_file='auto'):
    """ Read the Alignement file (TransformationMatrix.txt) generated by
    align_with_stackReg (MultiStackReg in imageJ)

    Parameters
    ----------
    path_align_file: str
        if 'auto', take the file in
        'C:\\Users\\pb565\\.hyperspy\\imageJ\\TransfoMatrix.txt'. The
        default file for align_with_stackReg

    Return
    ------
    shifts: an array that can be use with align2D
    """

    if path_align_file == 'auto':
        path_align_file = os.path.join(
            config_path,
            "imageJ\\TransfoMatrix.txt")
    f = open(path_align_file, 'r')
    for i in range(10):
        line = f.readline()
    middle = [float(line.split('\t')[0]), float(line.split('\t')[1][:-1])]
    # readshift
    f = open(path_align_file, 'r')
    shiftI = list()
    i = -1
    for line in f:
        if 'Source' in line:
            if i == -1:
                shiftI.append([int(line.split(' ')[-1]), middle])
            shiftI.append([int(line.split(' ')[2])])
            i = 1
        elif i == 1:
            shiftI[
                -1].append([float(line.split('\t')[0]), float(line.split('\t')[1][:-1])])
            i = 0
    f.close()
    starting_slice = shiftI[0][0]
    shiftI.sort()
    a = []
    for i, shift in enumerate(shiftI):
        a.append(shift[1])
    shiftI = (np.array(a) - middle)
    shiftIcumu = []
    for i, sh in enumerate(shiftI):
        if i < starting_slice:
            shiftIcumu.append(np.sum(shiftI[i:starting_slice], axis=0))
        else:
            shiftIcumu.append(np.sum(shiftI[starting_slice:i + 1], axis=0))
    shiftIcumu = np.array(shiftIcumu)
    shiftIcumu = np.array([shiftIcumu[::, 1], shiftIcumu[::, 0]]).T

    return shiftIcumu

# must be more general, compare image


def compare_results(specs, results, sum_elements=False,
                    normalize=False, plot_result=True, expand=False):
    """
    Plot different results side by side

    The results are found in 'mapped.metadata.Sample['results_name']'.
    They need to have the same dimension

    Parameters
    ----------
    specs: list || list of list || spec
        The list (list of list) of spectra containing the results.

    results: list || list of list || str
        The list (list of list) of name of the results (or a list of specs).

    normalize: bool
        If True, each result are normalized.

    plot_result : bool
        If True (default option), plot the result. If False, return
        the result.

    expand : bool
        if results and specs have different shape, expand in a matrix/lines.

    """
    from hyperspy import utils
    if expand == True:
        specs = copy.deepcopy(specs)
        if isinstance(specs, list):
            results = [results] * len(specs)
            for i, spec in enumerate(specs):
                specs[i] = [specs[i]] * len(results[0])
        else:
            if isinstance(results[0], list):
                specs = [[specs] * len(results[0])] * len(results)
            else:
                specs = [specs] * len(results)

    if isinstance(specs[0], list):
        if isinstance(results, list) is False:
            results = [[results] * len(specs[0])] * len(specs)
        check = []
        for j, spec in enumerate(specs):
            check_temp = []
            for i, s in enumerate(spec):
                if isinstance(results[j][i], str) is False:
                    temp = results[j][i].deepcopy()
                elif normalize:
                    temp = s.normalize_result(results[j][i])
                else:
                    temp = copy.deepcopy(
                        s.metadata.Sample[results[j][i]])
                temp = utils.stack(temp)
                if sum_elements:
                    temp = temp.sum(1)
                check_temp.append(temp)
            check.append(utils.stack(check_temp,
                                     axis=temp.axes_manager.signal_axes[0].name))

        check = utils.stack(check, axis=temp.axes_manager.signal_axes[1].name)
        check.axes_manager[-2].name += ' + results'
        check.axes_manager[-1].name += ' + specs'

    elif isinstance(specs, list):
        if isinstance(results, list) is False:
            results = [results] * len(specs)
        check = []
        for i, s in enumerate(specs):
            if isinstance(results[i], str) is False:
                temp = results[i].deepcopy()
            elif normalize:
                temp = s.normalize_result(results[i])
            else:
                temp = copy.deepcopy(s.metadata.Sample[results[i]])
            temp = utils.stack(temp)
            if sum_elements:
                temp = temp.sum(1)
            check.append(temp)

        check = utils.stack(check, axis=temp.axes_manager.signal_axes[0].name)
    else:
        raise ValueError("specs is not a list")

    check.metadata.General.title = 'Compared Results'
    if plot_result:
        check.plot(navigator='slider')
    else:
        return check

# Should use utils.plot.plot_histograms


def compare_histograms_results(specs,
                               element,
                               results,
                               normalizeI=False,
                               normalizex=False,
                               bins='freedman',
                               color=None,
                               line_style=None,
                               legend='auto',
                               fig=None):
    """
    Plot the histrogram for different results for one element.

    The results are found in 'mapped.metadata.Sample['results_name']'.

    Paramters
    ---------

    specs: list
        The list of spectra containing the results.

    element: str
        The element to consider. 'all' return the sum over all elements.

    results: list || str
        The list of name of the results (or a list of images).

    normalizeI: bool
        nomralize the intensity

    normalizex: bool
        nomralize over all the results

    bins : int or list or str (optional)
        If bins is a string, then it must be one of:
        'knuth' : use Knuth's rule to determine bins
        'scotts' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins
        'blocks' : use bayesian blocks for dynamic bin widths

    color : valid matplotlib color or a list of them or `None`
        Sets the color of the lines of the plots when `style` is "cascade"
        or "mosaic". If a list, if its length is
        less than the number of spectra to plot, the colors will be cycled. If
        If `None`, use default matplotlib color cycle.

    line_style: valid matplotlib line style or a list of them or `None`
        Sets the line style of the plots for "cascade"
        or "mosaic". The main line style are '-','--','steps','-.',':'.
        If a list, if its length is less than the number of
        spectra to plot, line_style will be cycled. If
        If `None`, use continuous lines, eg: ('-','--','steps','-.',':')

    legend: None | list of str | 'auto'
       If list of string, legend for "cascade" or title for "mosaic" is
       displayed. If 'auto', the title of each spectra (metadata.General.title)
       is used.

    fig : {matplotlib figure, None}
        If None, a default figure will be created.

    """
    from hyperspy import utils
    specs = copy.deepcopy(specs)
    if isinstance(results, list) is False:
        results = [results] * len(specs)
    elif isinstance(specs, list) is False:
        specs = [specs] * len(results)
    # else:
    #    dim_results = len(results)
    #    results = np.repeat(results,len(specs))
    #    specs = specs*dim_results
    hists = []
    for i, spec in enumerate(specs):
        if element == 'all':
            re = copy.deepcopy(spec.metadata.Sample[results[i]])
            re = utils.stack(re)
            re = re.sum(1)
            re.metadata.General.title = 'Sum ' + \
                results[i] + ' ' + spec.metadata.General.title
        elif isinstance(results[i], str):
            if normalizex:
                re = spec.normalize_result(
                    results[i])[list(spec.metadata.Sample.elements).index(element)]
            else:
                re = spec.get_result(element, results[i])
            re.metadata.General.title = element + ' ' + \
                results[i] + ' ' + spec.metadata.General.title
        else:
            re = results[i].deepcopy()
            # print 'Normalise x not available yet'
            re.metadata.General.title = (element + ' ' +
                                         re.metadata.General.title + ' ' + spec.metadata.General.title)
        #data = re.data.flatten()
        #center, hist1 = _histo_data_plot(data,bins)
        hist_tmp = re.get_histogram(bins)
        if normalizeI:
            hist_tmp = hist_tmp / float(hist_tmp.sum(0).data)
        hists.append(hist_tmp)

    return utils.plot.plot_spectra(hists, style='overlap', color=color,
                                   line_style=line_style, legend=legend, fig=fig)


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
    from hyperspy import utils
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
        compos_wt.append(list(
            utils.material.atomic_to_weight(
                elements,
                compo_at) /
            100))

    if density == 'auto':
        density = []
        for compo_wt in compos_wt:
            density.append(
                utils.material.density_of_mixture_of_pure_elements(
                    elements,
                    compo_wt))

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
            #noisyb=epq.SpectrumUtils.addNoiseToSpectrum(specb,live_time)
            #dtsa2.display(noisyb)

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


def crop_indexes_from_shift(shifts):
    """Get the crops index from shift

    Return
    ------
    top, bottom,left, right

    See also
    -------
    align2D
    """

    shifts = -shifts
    bottom, top = (int(np.floor(shifts[:, 0].min())) if
                   shifts[:, 0].min() < 0 else None,
                   int(np.ceil(shifts[:, 0].max())) if
                   shifts[:, 0].max() > 0 else 0)
    right, left = (int(np.floor(shifts[:, 1].min())) if
                   shifts[:, 1].min() < 0 else None,
                   int(np.ceil(shifts[:, 1].max())) if
                   shifts[:, 1].max() > 0 else 0)
    shifts = -shifts
    return top, bottom, left, right

# Control of detector geometry not good. Do it trhough interface
# Similar to simulate_one_spectrum, but with 4 detectors


def simulate_one_spectrum_TEM(nTraj, dose=100, mp='gui',
                              elements='auto',
                              compo_at='auto',
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

    compo_at: list of string
        Give the composition (atomic). If auto, equally parted

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
    from hyperspy import utils

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
    if compo_at == 'auto':
        compo_at = []
        for elm in elements:
            compo_at.append(1. / len(elements))
    mp.Sample.compo_at = compo_at
    compo_wt = np.array(
        utils.material.atomic_to_weight(
            elements,
            compo_at)) / 100
    compo_wt = list(compo_wt)
    if density == 'auto':
        density = utils.material.density_of_mixture_of_pure_elements(
            elements,
            compo_wt)
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
    TOangle = [utils.eds.take_off_angle(tilt, az,
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
    spec = signals.EDSSEMSpectrum(datas)
    spec.set_microscope_parameters(
        azimuth_angle=float(prop.split(
            'Azimuthal angle=')[1].split('\xb0')[0]),
        elevation_angle=float(prop.split(
            'Elevation=')[1].split('\xb0')[0]),
        energy_resolution_MnKa=float(prop.split(
            'Resolution=')[1].split(' eV')[0]))
    spec.axes_manager[-1].offset = float(
        prop.split('Energy offset=')[1].split(' eV')[0]) / 1000
    spec.axes_manager[-1].scale = float(
        prop.split('Energy scale=')[1].split(' eV')[0]) / 1000
    spec.axes_manager[-1].offset += spec.axes_manager[-1].scale / 2
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


def plot_orthoview_animated(image, isotropic_voxel=True):
    """
    Plot an orthogonal view of a 3D images

    Parameters
    ---------

    image: signals.Image
        An image in 3D.

    isotropic_voxel:
        If True, generate a new image, scaling z in order to obtain isotropic
        voxel.
    """
    if isotropic_voxel:
        im_xy, scale = get_isotropic_3D_image(image)
    else:
        im_xy = image.deepcopy()
    im_xy.metadata.General.title = 'xy'
    im_xy.axes_manager.set_signal_dimension(0)

    im_xz = im_xy.deepcopy()
    im_xz = im_xz.rollaxis(2, 1)
    im_xz.metadata.General.title = 'xz'
    im_xz.axes_manager.set_signal_dimension(0)

    im_xz.axes_manager._axes[2] = im_xy.axes_manager._axes[2]
    im_xz.axes_manager._axes[1] = im_xy.axes_manager._axes[0]
    im_xz.axes_manager._axes[0] = im_xy.axes_manager._axes[1]

    im_yz = im_xy.deepcopy()
    im_yz = im_yz.rollaxis(0, 2)
    im_yz = im_yz.rollaxis(1, 0)
    im_yz.metadata.General.title = 'yz'
    im_yz.axes_manager.set_signal_dimension(0)

    im_yz.axes_manager._axes = im_xy.axes_manager._axes[::-1]

    im_xz.axes_manager._update_attributes()
    im_yz.axes_manager._update_attributes()
    im_xy.plot()
    im_xz.plot()
    im_yz.plot()


def plot_orthoview(image,
                   index,
                   plot_index=False,
                   space=2,
                   plot_result=True,
                   isotropic_voxel=True):
    """
    Plot an orthogonal view of a 3D images

    Parameters
    ----------
    image: signals.Image
        An image in 3D.

    index: list
        The position [x,y,z] of the view.

    line_index: bool
        Plot the line indicating the index position.

    space: int
        the spacing between the images in pixel.

    plot_result: bool
        if False, return the image.

    isotropic_voxel:
        If True, generate a new image, scaling z in order to obtain isotropic
        voxel.
    """
    from hyperspy import signals
    from hyperspy import utils

    if isotropic_voxel:
        image, scale_fact = get_isotropic_3D_image(image)
    else:
        image = image.deepcopy()
        scale_fact = 1

    dim = image.axes_manager.shape

    map_color = plt.get_cmap()
    if map_color.name == 'RdYlBu_r':
        mean_img = image.mean(0).mean(0).mean(0).data
    else:
        mean_img = image.max(0).max(0).max(0).data * 0.88

    if isinstance(index[2], int):
        a = image[index[2] * scale_fact].deepcopy()
    else:
        a = image[index[2]].deepcopy()
    b = image[::, index[0]].as_image([0, 1]).deepcopy()
    c = image[::, ::, index[1]].as_image([1, 0]).deepcopy()
    if plot_index:
        a.data[::, index[0]] = np.ones(dim[2]) * mean_img
        a.data[index[1]] = np.ones(dim[1]) * mean_img
        b.data[index[1]] = np.ones(dim[0]) * mean_img
        b.data[::, index[2] * scale_fact] = np.ones(dim[2]) * mean_img
        c.data[index[2] * scale_fact] = np.ones(dim[1]) * mean_img
        c.data[::, index[0]] = np.ones(dim[0]) * mean_img

    im = utils.stack([a,
                      signals.Image(np.ones([dim[2], space]) * mean_img), b], axis=0)
    im2 = utils.stack([c,
                       signals.Image(np.ones([dim[0], dim[0] + space]) * mean_img)], axis=0)
    im = utils.stack([im,
                      signals.Image(np.ones([space, dim[1] + dim[0] + space]) * mean_img), im2], axis=1)
    # Why I need to do that
    im.axes_manager[0].offset = 0
    im.axes_manager[0].offset = 0

    if plot_result:
        fig = im.plot()
        return fig
    else:
        return im


def get_isotropic_3D_image(image):
    """Rescale the z axes to generate a  new image with isotropic voxel.

    Returns
    -------

    signals.Image, int: The resaled image and the scaling factor applied
        to z.
    """
    image = image.deepcopy()
    dim = image.axes_manager.shape
    if len(dim) != 3:
        raise ValueError('Needs a 3D image')
    scalez = image.axes_manager[0].scale
    scalex = image.axes_manager[1].scale
    if scalez > scalex:
        scale_fact = int(scalez / scalex)
        image.data = np.repeat(image.data, int(scalez / scalex), axis=0)
        image.get_dimensions_from_data()
        image.axes_manager[0].scale /= scale_fact
    else:
        scale_fact = 1
    return image, scale_fact


def get_contrast_brightness_from(img, reference, return_factors=False):
    """Set the contrast/brightness of an image to be the same as a reference.

    Fit the histogram of the image on the histogram of the reference to
    get the change in contrast bightness

    Parameters
    ---------

    img: Signal
        The signal fo which the contrast need to be adjsuted

    reference: Signal
        The contrast/brightness reference

    return_factors:bool
        If False, return the adjusted image
        If True, return the adjusted image, contrast adjustement
        and the brightness adjustment.
    """
    from hyperspy.hspy import create_model

    img = img.deepcopy()

    hist_img = img.get_histogram(bins=50)
    hist_ref = reference.get_histogram(bins=50)

    posmax_ref = list(hist_ref.data).index(max(hist_ref.data))
    posmax_img = list(hist_img.data).index(max(hist_img.data))

    m = create_model(hist_img)
    fp = components.ScalableFixedPattern(hist_ref)

    fp.xscale.value = (hist_ref.axes_manager[0].scale /
                       hist_img.axes_manager[0].scale)
    fp.shift.value = (hist_ref.axes_manager[0].scale * posmax_ref /
                      hist_img.axes_manager[0].scale / posmax_img)

    fp.set_parameters_free(['xscale', 'shift'])
    fp.set_parameters_not_free(['yscale'])
    m.append(fp)
    m.multifit()

    img *= fp.xscale.value
    img -= fp.shift.value

    if return_factors:
        return img, fp.xscale.value, fp.shift.value
    else:
        return img


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


def fft(self, shape_fft=None, axes=None, scale=None):
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

    dim = len(self.axes_manager.shape)
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

    # if self.axes_manager.signal_dimension == 2:
        # im_fft.axes_manager.set_signal_dimension(2)
        #im_fft = im_fft.as_image([-2,-1])
    # else:
    #    im_fft = im_fft.as_spectrum(-1)
    # scale, to be verified
    if scale is not None:
        for i in range(dim):
            im_fft.axes_manager[i].scale = scale[i]
            #im_fft.axes_manager[i].scale = 1 / self.axes_manager[i].scale
    elif shape_fft is None:
        for i in range(dim):
            im_fft.axes_manager[i].scale = 1 / self.axes_manager[i].scale
    else:
        print ('scale not given')

    return im_fft


def database_detector_efficiency_INCA(index=4):
    """
    Import the detector efficiency detector used by INCA

    Paramaters
    ----------

    index: {0,1,2,3,4}
        Choose between the different detector
        0: 'X-Max 4'
        1: 'x-act 3'
        2: 'OINAXmax80J1 50SD41K'
        3: 'OINAXmax80-FS 50SD41K'
        4: 'OINAXmax80ap4-FS 50SD41K'
    """
    from hyperspy import signals

    det_name = [
        'X-Max 4',
        'x-act 3',
        'OINAXmax80J1 50SD41K',
        'OINAXmax80-FS 50SD41K',
        'OINAXmax80ap4-FS 50SD41K']

    foldername = os.path.join(config_path,
                              'database//det_efficiency_INCA\\' + det_name[index] + '.efy')
    data = np.memmap(foldername, dtype="float32")

    if index < 2:
        det = signals.EDSSEMSpectrum(data[745:5646])
        det.axes_manager[-1].scale = 0.001
        det.axes_manager[-1].offset = 0.1
    else:
        det = signals.Spectrum(data[649:3000 + 646])
        det.axes_manager[-1].scale = 0.01
        det.axes_manager[-1].offset = 0.05
    det.metadata.General.title = det_name[index]
    det.axes_manager[-1].units = "keV"
    det.axes_manager[-1].name = "Energy"
    return det


def _load_in_database(name, result=False):
    from hyperspy.io import load
    foldername = os.path.join(config_path, 'database//' + name)
    if result:
        return load_EDSSEMSpectrum(foldername)
    else:
        return load(foldername)


def database_1Dspec(which_spec='BAM'):
    """
    load 1D spec

    Parameters
    ----------

    which_spec: {'BAM','msa','noisy'}
        if BAM: bam sample
        if msa: GnMeba test, coin of euro
        if noisy: 1 pixel in AlZn

    """

    if which_spec == 'BAM':
        return _load_in_database('bam.hdf5')
    elif which_spec == 'msa':
        return _load_in_database('GNmeba_test.msa')
    elif which_spec == 'noisy':
        return _load_in_database('1pix_AlZn.msa')


def database_3Dspec(which_spec='PCA_SEM'):
    """
    load 3D spec

    Parameters
    ----------

    which_spec: {'PCA_SEM','SEM','Ti_SEM','rpl','noisy'}
        if 'PCA_SEM', load RR 46 PCA rec
        if 'SEM', load RR 46 no PCA
        ifs 'Ti_SEM', load TiFeNi no PCA jonas1h croped (::,:12)
        if 'rpl', jonas1h raw
        if 'noisy', AlZn 40 .rpl, see noisy 1D
    """

    if which_spec == 'PCA_SEM':
        return _load_in_database('specImg3DBinPCAre46.hdf5')
    elif which_spec == 'SEM':
        return _load_in_database('specImg3D46.hdf5')
    elif which_spec == 'Ti_SEM':
        return _load_in_database('TiFeNi1h.hdf5')
    elif which_spec == 'rpl':
        return _load_in_database('jonas1h.rpl')
    elif which_spec == 'noisy':
        return _load_in_database('AlZn__040.rpl')


def database_4Dspec(which_spec='PCA_SEM'):
    """
    load RR PCA rec (10:15) or Cat (TEM) no PCA

    Parameters
    ----------

    which_spec: {'PCA_SEM','TEM'}
        if 'PCA_SEM', load RR (slices 10:15) PCA rec
        if 'TEM', load Cat (TEM) no PCA
    """
    if which_spec == 'PCA_SEM':
        return _load_in_database('specImg3DBinPCArec.hdf5')
    elif which_spec == 'TEM':
        return _load_in_database('cate_3D_bin_reduced.hdf5')


def database_2Dimage(which_spec='SEM'):
    """
    load 2D image

    which_spec: {'SEM','Ti_SEM','lena'}
        if SEM, RR SE 46 (TLD SE)
        if Ti_SEM, jonas1h SE image (inLens, bck corrected, croped)
        if lena, scipy.misc.lena
    """
    if which_spec == 'SEM':
        return _load_in_database('img46.hdf5')
    elif which_spec == 'Ti_SEM':
        return _load_in_database('SE_imTiFeNi1h.hdf5')
    elif which_spec == 'lena':
        import scipy.ndimage
        from hyperspy.signals import Image
        return Image(scipy.misc.lena())


def database_3Dimage():
    """
    load RR SE (10:20)
    """
    return _load_in_database('2img3DA.hdf5')


def database_3Dresult():
    """
    load RR 2 3D
    """
    return _load_in_database('2res3DrsAH.hdf5', result=True)

# doesn't work
# bug of map


def tv_denoise(self,
               weight=50,
               n_iter_max=200,
               eps=0.0002,
               method='bregman'):
    """
    Perform total-variation denoising on 2D and 3D images.

    Parameters
    ---------

    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:

        (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    method: 'chambolle' | 'bregman'

    Example
    -------

    >>> im = utils_eds.database_2Dimage()
    >>> utils_eds.tv_denoise(im,method='chambolle',
    >>>      weight=0.5,n_iter_max=4).plot()

    See also:
    -----

    skimage.filter.denoise_tv_chambolle
    skimage.filter.denoise_tv_bregman

    """

    import skimage.filter
    img = self.deepcopy()

    if method == 'bregman':
         # img.map(
         #   skimage.filter.denoise_tv_bregman, weight=weight,
         #   eps=eps, max_iter=n_iter_max)
        img.data = skimage.filter.denoise_tv_bregman(img.data, weight=weight,
                                                     eps=eps, max_iter=n_iter_max)
    elif method == 'chambolle':
        # img.map(
        #    skimage.filter.denoise_tv_chambolle,
        #    weight=weight, eps=eps, n_iter_max=n_iter_max)
        img.data = skimage.filter.denoise_tv_chambolle(img.data,
                                                       weight=weight, eps=eps, n_iter_max=n_iter_max)
    return img


def mean_filter(self, size):
    """ Apply a mean filter.

    Parameters
    ----------

    size : int | list or tuple
        `size` gives the shape that is taken from the input array,
        at every element position, to define the input to the filter
        function.

    """
    import scipy.ndimage
    dim = self.axes_manager.shape
    if isinstance(size, int):
        kernel = np.ones([size] * len(dim))
    else:
        kernel = np.ones(size)
    kernel = kernel / kernel.sum()
    img = self.map(scipy.ndimage.convolve, weights=kernel)
    return img


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
    s = database_1Dspec()

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


def get_xray_transition_properties(xray_line, beam_energy, gateway='auto'):
    """ Return the properties of a given Xray transition:

    Compute the ionization cross section, fluorescence_yield and the relative
    transition probability for a beam energy, an elements
    and the ionized shell corresponding to the given Xray-lines.

    Parameters
    ----------
    xray_line: str
        The X-ray line, e.g. 'Al_Ka'
    beam_energy: float
        The energy of the beam in kV.
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Return
    ------

    [ionization_cross_section, fluorescence_yield

    Notes
    -----
    ionization_cross_section from the BoteSalvat2008 database
    """
    if gateway == 'auto':
        gateway = get_link_to_jython()
    channel = gateway.remote_exec("""
        import dtsa2
        epq = dtsa2.epq
        
        xray_line = '""" + str(xray_line) + """'
        beam_energy = """ + str(beam_energy) + """
        beam_energy = epq.ToSI.keV(beam_energy)
        lim = xray_line.find('_')
        el = getattr(dtsa2.epq.Element,xray_line[:lim])
        li = xray_line[lim+1:]

        if 'K' in li:
            shell = 0
        elif li == 'Lb3' or li == 'Lb4' or li == 'Lg3':
            shell = 1
        elif li == 'Lb1' or li == 'Lbn' or li == 'Lg1':
            shell= 2
        elif li == 'La' or li == 'Lb2' or li == 'Ll':
            shell = 3
        elif li == 'Mz':
            shell = 5
        elif li == 'Mg':
            shell = 6
        elif li == 'Mb':
            shell = 7
        elif li == 'Ma':
            shell = 8
        atomic_shell = epq.AtomicShell(el, shell)
        print atomic_shell
        ICS= epq.AbsoluteIonizationCrossSection.\
            BoteSalvat2008.computeShell(atomic_shell,
            beam_energy)
        #ICS= epq.AbsoluteIonizationCrossSection.\
        #    Casnati82.computeShell(atomic_shell,
        #    beam_energy)

        FY= epq.FluorescenceYield.DefaultShell.compute(
            atomic_shell)
        #FY= epq.FluorescenceYield.Sogut2002.compute(
        #    atomic_shell)
        channel.send(ICS)
        channel.send(FY)
    """)
    datas = []
    for i, item in enumerate(channel):
        datas.append(item)
    element, line = _get_element_and_line(xray_line)
    fact = elements_db[element]['Atomic_properties']\
        ['Xray_lines'][line]['factor']
    return datas + [fact]


def get_mass_absorption_coefficient(energy,
                                    elements,
                                    weight_percent='auto',
                                    gateway='auto'):
    """
    Return the mass absorption coefficient for an energy in
    a sample of a given composition

    Use Chantler2005 database

    Parameters
    ----------
    energy: float
        The energy of the beam in kV.
    elements: list of strings
        The symbol of the elements.
    weight_percent: list of strings
        The corresponding composition eg. [0.2,0.8]. If 'auto' use
        the eigen composition
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    Return
    ------
    Return the mass absorption coefficient in cm^2/g

    Notes
    -----

    See

    """
    if gateway == 'auto':
        gateway = get_link_to_jython()
    elements = list(elements)
    if weight_percent == 'auto':
        weight_percent = []
        for elm in elements:
            weight_percent.append(1. / len(elements))
    else:
        weight_percent = list(weight_percent)
    if hasattr(energy, '__iter__'):
        energy = list(energy)
    channel = gateway.remote_exec("""
        import dtsa2
        epq = dtsa2.epq
        energy = """ + str(energy) + """
        elements = """ + str(elements) + """
        weight_percent = """ + str(weight_percent) + """
        elms = []
        for element in elements:
            elms.append(getattr(epq.Element,element))
        composition = epq.Composition(elms ,weight_percent)
        if isinstance(energy, list):
            for en in energy:
                en = epq.ToSI.keV(en)
                MAC = epq.MassAbsorptionCoefficient.Chantler2005.compute(
                    composition, en)
                channel.send(MAC*10)
        else:
            energy = epq.ToSI.keV(energy)
            MAC = epq.MassAbsorptionCoefficient.Chantler2005.\
                compute(composition, energy)
            #u_MAC = epq.MassAbsorptionCoefficient.Chantler2005.\
            #    computeWithUncertaintyEstimate(
            #    composition, energy)
            channel.send(MAC*10)
            #channel.send(u_MAC*10)
    """)
    datas = []
    for i, item in enumerate(channel):
        datas.append(item)
    # if isinstance(energy, list) is False:
    #    print 'with uncertainty'
    return datas


def get_kfactors(xray_lines,
                 beam_energy,
                 detector_efficiency=None,
                 gateway='auto'):
    """Calculate the kfactors cofficient of Cliff-Lorimer method from first
    principle

    Parameters
    ----------
    xray_lines: list of strin
        The X-ray lines, e.g. ['Al_Ka', 'Zn_Ka']
    beam_energy: float
        The energy of the beam in kV.
    detector_efficiency: {signals.EDSSEMSpectrum,None}
        A spectrum containing the detector efficiency. If None, set the
        efficiency to one.
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.

    """
    if gateway == 'auto':
        gateway = get_link_to_jython()
    kab = []
    for xray_line in xray_lines:
        xray_prop = get_xray_transition_properties(
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
    kab = kab[1] / kab[0]
    return kab

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
