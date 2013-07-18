import math
import numpy as np

from hyperspy.misc.eds.elements import elements as elements_db
    
def _get_element_and_line(Xray_line):
    lim = Xray_line.find('_')
    return Xray_line[:lim], Xray_line[lim+1:]    
    
def xray_range(Xray_line,beam_energy,rho=None):
    '''Return the Anderson-Hasler X-ray range.
    
    Parameters
    ----------    
    Xray_line: str
        The X-ray line, e.g. Al_Ka
        
    beam_energy: float (kV)
        The energy of the beam in kV. 
        
    rho: float (g/cm3)
        The density of the material. If None, the density of the pure 
        element is used.
        
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
    element, line = _get_element_and_line(Xray_line)
    if rho is None:
        rho = elements_db[element]['density']
    Xray_energy = elements_db[element]['Xray_energy'][line]
    
    return 0.064/rho*(np.power(beam_energy,1.68)-
        np.power(Xray_energy,1.68))
        
def FWHM(FWHM_ref,E,line_ref='Mn_Ka'):
    """Calculates the FWHM of a peak at energy E from the FWHM of a 
    reference peak.
    
    Parameters
    ----------
    energy_resolution_MnKa : float
        Energy resolution of Mn Ka in eV
        
    E : float
        Energy of the peak in keV
        
    line_ref : str
        The references X-ray line. Set by default at 'Mn_Ka'
    
            
    Returns
    -------
    float : FWHM of the peak in keV
    
    Notes
    -----
    From the textbook of Goldstein et al., Plenum publisher, 
    third edition p 315
    
    as defined by Fiori and Newbury (1978). In SEM/1978/I, AMF O'Hare,
    p 401
    
    
    """
    
    element, line = _get_element_and_line(line_ref)
    E_ref = elements_db[element]['Xray_energy'][line]
    
    
    
    FWHM_e = 2.5*(E-E_ref)*1000 + FWHM_ref*FWHM_ref
   
    return math.sqrt(FWHM_e)/1000
    
    
def TOA(self,tilt_stage=None,azimuth_angle=None,elevation_angle=None):
    """Calculate the take-off-angle (TOA).
    
    TOA is the angle with which the X-rays leave the surface towards 
    the detector. If any parameter is None, it is read in 'SEM.tilt_stage',
    'SEM.EDS.azimuth_angle' and 'SEM.EDS.elevation_angle'
     in 'mapped_parameters'.

    Parameters
    ----------
    tilt_stage: float (Degree)
        The tilt of the stage. The sample is facing the detector when
        positively tilted. 

    azimuth_angle: float (Degree)
        The azimuth of the detector. 0 is perpendicular to the tilt 
        axis. 

    elevation_angle: float (Degree)
        The elevation of the detector.
                
    Returns
    -------
    TOA: float (Degree)
    
    Notes
    -----
    Defined by M. Schaffer et al., Ultramicroscopy 107(8), pp 587-597 (2007)
    
    """
        
    if tilt_stage == None:
        a = math.radians(90+self.mapped_parameters.SEM.tilt_stage)
    else:
        a = math.radians(90+tilt_stage)
        
    if azimuth_angle == None:
        b = math.radians(self.mapped_parameters.SEM.EDS.azimuth_angle)
    else:
        b = math.radians(azimuth_angle)
        
    if elevation_angle == None:
        c = math.radians(self.mapped_parameters.SEM.EDS.elevation_angle)
    else:
        c = math.radians(elevation_angle)
    
    return math.degrees( np.arcsin (-math.cos(a)*math.cos(b)*math.cos(c) \
    + math.sin(a)*math.sin(c)))
    
def phase_inspector(self,bins=[20,20,20],plot_result=True):
    """
    Generate an binary image of different channel
    """
    bins=[20,20,20]
    minmax = []
    
    #generate the bins
    for s in self:    
        minmax.append([s.data.min(),s.data.max()])
    center = []
    for i, mm in enumerate(minmax):
        temp = list(mlab.frange(mm[0],mm[1],(mm[1]-mm[0])/bins[i]))
        temp[-1]+= 1
        center.append(temp)
        
    #calculate the Binary images
    dataBin = []
    if len(self) ==1:
        for x in range(bins[0]):
            temp = self[0].deepcopy()
            dataBin.append(temp)
            dataBin[x].data = ((temp.data >= center[0][x])*
              (temp.data < center[0][x+1])).astype('int')
    elif len(self) == 2 :    
        for x in range(bins[0]):
            dataBin.append([])
            for y in range(bins[1]):
                temp = self[0].deepcopy()
                temp.data = np.ones_like(temp.data)
                dataBin[-1].append(temp)
                a = [x,y]
                for i, s in enumerate(self):
                    dataBin[x][y].data *= ((s.data >= center[i][a[i]])*
                     (s.data < center[i][a[i]+1])).astype('int')
            dataBin[x] = utils.stack(dataBin[x])
    elif len(self) == 3 :    
        for x in range(bins[0]):
            dataBin.append([])
            for y in range(bins[1]):
                dataBin[x].append([])                    
                for z in range(bins[2]):
                    temp = self[0].deepcopy()
                    temp.data = np.ones_like(temp.data)
                    dataBin[-1][-1].append(temp)
                    a = [x,y,z]
                    for i, s in enumerate(self):
                        dataBin[x][y][z].data *= ((s.data >=
                         center[i][a[i]])*(s.data < 
                         center[i][a[i]+1])).astype('int')
                dataBin[x][y] = utils.stack(dataBin[x][y])
            dataBin[x] = utils.stack(dataBin[x])
    img = utils.stack(dataBin)

    for i in range(len(self)):
        img.axes_manager[i].name = self[i].mapped_parameters.title
        img.axes_manager[i].scale = (minmax[i][1]-minmax[i][0])/bins[i]
        img.axes_manager[i].offest = minmax[i][0]
        img.axes_manager[i].units = '-'
    img.get_dimensions_from_data()
    return img 
    
    
def simulate_one_spectrum(nTraj,dose=100,mp='gui',
        elements='auto',
        composition='auto',
        density='auto',
        detector='Si(Li)'):
    """"
    Simulate a spectrum using DTSA-II
    
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
        
    composition: list of string
        Give the composition. If auto, equally parted
        
    detector: str
        Give the detector name defined in DTSA-II       
   
    """
    from hyperspy import signals
    if mp == 'gui':
        spec = signals.EDSSEMSpectrum(np.zeros(1024))
        spec.set_microscope_parameters()
        mp = spec.mapped_parameters
        dic = mp.as_dictionary()
        if elements == 'auto':
            print 'Elements need to be set with gui option'
            return 0
        else:
            spec.set_elements(elements)  
    dic = mp.as_dictionary()
    if hasattr(mp.Sample, 'Xray_lines'):
        dic['Sample']['Xray_lines'] = list(dic['Sample']['Xray_lines'])
        
    if hasattr(mp.Sample, 'elements'):
        dic['Sample']['elements'] = list(dic['Sample']['elements'])
    else:
        print 'Elements needs to be defined'
        return 0
        
    if density == 'auto':
        density = 7.0
        
    if composition == 'auto':
        composition = []
        for elm in dic['Sample']['elements']:
            composition.append(1./len(dic['Sample']['elements']))         
        

    import execnet
    gw = execnet.makegateway(
        "popen//python=C:\Users\pb565\Documents\Java\Jython2.7b\jython.bat")
    channel = gw.remote_exec("""   
        import dtsa2
        import math
        epq = dtsa2.epq 
        epu = dtsa2.epu
        nm = dtsa2.nm
       
        param = """ + str(dic) + """

        elements = param[u'Sample'][u'elements']
        elms = []
        for element in elements:
            elms.append(getattr(dtsa2.epq.Element,element))
        density = """ + str(density) + """
        composition = """ + str(composition) + """
        e0 = param[u'SEM'][u'beam_energy']
        tiltD = -1*param[u'SEM'][u'tilt_stage']
        live_time = param[u'SEM'][u'EDS'][u'live_time']

        nTraj = """ + str(nTraj) + """
        dose = """ + str(dose) + """
        
        IncrementF = 0.5
        pixSize = (4*1.0e-9,200*1.0e-9,100*1.0e-9)
        pixLat = (70, 5)
        pixTot = pixLat[0]*pixLat[1]
        tilt = math.radians(tiltD) # tilt angle radian

        det = dtsa2.findDetector('""" + detector + """')
        origin = epu.Math2.multiply(1.0e-3, epq.SpectrumUtils.getSamplePosition(det.getProperties()))
        z0 = origin[2]

        el = 0
        if len(elms) == 1:
            mat=epq.MaterialFactory.createPureElement(elms[el])
        else:            
            mat = epq.Material(epq.Composition(elms,composition ),
                                    epq.ToSI.gPerCC(density))


        # Create a simulator and initialize it
        monteb = nm.MonteCarloSS()
        monteb.setBeamEnergy(epq.ToSI.keV(e0))

        # top substrat
        monteb.addSubRegion(monteb.getChamber(), mat,      
                                  nm.MultiPlaneShape.createSubstrate([math.sin(tilt),0.0,-math.cos(tilt)], origin) )
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
        propsb.setTextProperty(epq.SpectrumProperties.SpectrumDisplayName, 
                              "%s std." % (elms[el]))
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
            channel.send(noisyb.getCounts(i))
               
    """)

    datas = []
    for i, item in enumerate(channel):
        if i == 0:
            scale = item
        elif i==1:
            offset = item
        elif i==2:
            reso = item
        else:
            datas.append(item)
        
    try:
        spec
    except:
        spec = signals.EDSSEMSpectrum(np.array(datas))
        spec.mapped_parameters = mp
    else:    
        spec.data = np.array(datas)
    spec.get_dimensions_from_data() 
    
    spec.mapped_parameters.SEM.EDS.energy_resolution_MnKa = reso
    spec.axes_manager[0].scale = scale / 1000
    spec.axes_manager[0].offset = offset
    spec.axes_manager[0].name = 'Energy'
    spec.axes_manager[0].unit = 'keV'
    spec.mapped_parameters.title = 'Simulated spectrum'
    
    

    return spec
    
    
