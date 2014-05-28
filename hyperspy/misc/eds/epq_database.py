import numpy as np

from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.elements import elements as elements_db


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
    energy: float or list of float
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
        gateway = utils_eds.get_link_to_jython()
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
        gateway = utils_eds.get_link_to_jython()
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
    element, line = utils_eds._get_element_and_line(xray_line)
    fact = elements_db[element]['Atomic_properties']\
        ['Xray_lines'][line]['factor']
    return datas + [fact]


def get_energy_and_weight(line, gateway='auto'):
    """ Get the transition energy and the wieght

    Compute the transition energy (Chantler2005) and the weight of the
    line (epq library)

    Parameters
    ----------
    line: str
        The X-ray line, e.g. 'Al_Ka'
    gateway: execnet Gateway
        If 'auto', generate automatically the connection to jython.
    """
    if gateway == 'auto':
        gateway = utils_eds.get_link_to_jython()
    channel = gateway.remote_exec("""
        import dtsa2
        epq = dtsa2.epq
        xray_line = '""" + str(line) + """'
        lim = xray_line.find('_')
        el = getattr(dtsa2.epq.Element,xray_line[:lim])
        li = xray_line[lim+1:]
        print li
        lines = ['Ka','Kb','La','Lb1','Lb2',
                 'Lb3','Lb4', 'Lg1', 'Lg3', 'Ll',
                'Ln', 'M2N4','Ma','Mb','Mg', 'Mz']
        transset = [0,2,12,31,15,
                 45,46,33,50,19,
                 37,57,72,69,66,74]
        for i, line in enumerate(lines):
            if li==line:
                trans = transset[i]
        print trans
        xray_transition = epq.XRayTransition(el, trans)
        print xray_transition
        #a = epq.IonizationCrossSection
        e = epq.TransitionEnergy.Chantler2005.compute(
            xray_transition)
        #ICS= epq.FluorescenceYield.DefaultShell.compute(
        #    atomic_shell)
        channel.send(e)
        w = epq.XRayTransition.getWeight(el, trans,0)
        channel.send(w)
    """)
    datas = []
    for i, item in enumerate(channel):
        datas.append(item)
    return datas[0] / 1.60217653E-16, datas[1]
