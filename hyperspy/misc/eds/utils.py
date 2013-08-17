import math
import numpy as np
import execnet
import os
import copy

from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.config_dir import config_path
from hyperspy import utils
import matplotlib.pyplot as plt

    
def _get_element_and_line(Xray_line):
    lim = Xray_line.find('_')
    return Xray_line[:lim], Xray_line[lim+1:]    
    
def xray_range(Xray_line,beam_energy,rho=None):
    '''Return the Anderson-Hasler X-ray range.
    
    Parameters
    ----------    
    Xray_line: str
        The X-ray line, e.g. 'Al_Ka'
        
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
        
def electron_range(element,beam_energy,rho=None,tilt=0):
    '''Return the Kanaya-Okayama electron range in a pure material.
    
    Parameters
    ----------    
    elements: str
        The element, e.g. 'Al'
        
    beam_energy: float (kV)
        The energy of the beam in kV. 
        
    rho: float (g/cm3)
        The density of the material. If None, the density of the pure 
        element is used.
        
    tilt: float (degree)
        the tilt of the sample.
        
    Returns
    -------
    Electron range in micrometer.
        
    Notes
    -----
    From Kanaya, K. and S. Okayama (1972). J. Phys. D. Appl. Phys. 5, p43
    
    See also the textbook of Goldstein et al., Plenum publisher, 
    third edition p 72
    '''

    if rho is None:
        rho = elements_db[element]['density']
        Z = elements_db[element]['Z']
        A = elements_db[element]['A']
    
    return (0.0276*A/np.power(Z,0.89)/rho*
        np.power(beam_energy,1.67)*math.cos(math.radians(tilt)))
        
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
        The tilt of the stage. The sample is facing a detector at 0 azimuth 
        when positively tilted. 

    azimuth_angle: float (Degree)
        The azimuth of the detector. 0 is perpendicular to the tilt 
        axis. A zero azimuth means that the tilt axis is normal to the 
        vertical place containing the axis of detection.

    elevation_angle: float (Degree)
        The elevation of the detector compared to a surface with 0 tilt.
        90 is SEM direction (perp to the surface)
                
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
        
    composition: list of string
        Give the composition. If auto, equally parted
        
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
    if mp == 'gui':
        spec = signals.EDSSEMSpectrum(np.zeros(1024))
        spec.set_microscope_parameters()
        mp = spec.mapped_parameters
        dic = mp.as_dictionary()
        if elements == 'auto':
            raise ValueError( 'Elements need to be set with gui option')
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
        

    
    if gateway == 'auto':
        gateway = get_link_to_jython()
    channel = gateway.remote_exec("""   
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

def get_link_to_jython():
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
    """Load the EDSSEMSpectrum and the result.
    
    See also
    --------
    
    load
    """
    from hyperspy.io import load
    
    s = load(filenames,record_by,signal_type,signal_origin,stack,
         stack_axis,new_axis_name,mmap,mmap_dir,**kwds)
         
    mp = s.mapped_parameters
    if hasattr(mp, 'Sample'):   
        for result in ['standard_spec','kratios','quant','quant_enh','intensities']:
            if hasattr(mp.Sample, result):
                _set_result_signal_list(mp,result)

    return s
    
def _set_result_signal_list(mp,result):
    std = mp.Sample[result]
    #if '_' in std.mapped_parameters.title:
    #    number_of_parts=len(mp.Sample.Xray_lines)
    #    is_xray = True
    #else:
    #    number_of_parts=len(mp.Sample.elements)
    #    is_xray = False
    number_of_parts=std.data.shape[0]
    if result =='standard_spec':
        #Need to change
        #number_of_parts=len(mp.Sample.elements)
        l_time = std.mapped_parameters.SEM.EDS.live_time
        #number_of_parts=len(mp.Sample.Xray_lines)
        temp = std.split(axis=0,number_of_parts=number_of_parts)        
    else:
        temp = std.split(axis=1,number_of_parts=number_of_parts)
    std = []
    for i, tp in enumerate(temp):
        tp = tp.squeeze()
        if result == 'standard_spec':
            #to change
            if number_of_parts==len(mp.Sample.Xray_lines):
                el, li = _get_element_and_line(mp.Sample.Xray_lines[i])
            else:
                el = mp.Sample.elements[i]
            tp.mapped_parameters.title = el + '_std'
            tp.mapped_parameters.SEM.EDS.live_time = l_time[i]
        elif number_of_parts==len(mp.Sample.Xray_lines):
            tp.mapped_parameters.title = result + ' ' + mp.Sample.Xray_lines[i]
        elif number_of_parts==len(mp.Sample.elements):
            tp.mapped_parameters.title = result + ' ' + mp.Sample.elements[i]
        std.append(tp)
    mp.Sample[result] = std


def align_with_stackReg(img,
    starting_slice=0,
    align_img=False,
    return_align_img=False,
    gateway='auto'):
    """Align a stack of images with stackReg from Imagej.
    
    store the shifts in mapped_parameters.align.shifts
    
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
        imgtemp.save(path_img,overwrite=True);
    else:
        img.save(path_img,overwrite=True);
    
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
        imp.setSlice("""+str(starting_slice)+"""+1)
        IJ.runPlugIn(imp, "MultiStackReg_", "")

        return_align_img="""+str(return_align_img)+"""
        if return_align_img:
            IJ.saveAs(imp,"Tiff",path_img_alnd[0])
        imp.close()
    
    """)
    
    for i in range(100):
        if os.path.exists(path_align_file):               
            break
        else:
            time.sleep(0.5)
            
    shifts = _read_alignement_file()
    mp = img.mapped_parameters
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
    
    if path_align_file=='auto':
        path_align_file = os.path.join(config_path, "imageJ\\TransfoMatrix.txt")
    f = open(path_align_file, 'r')
    for i in range(10):
        line = f.readline()
    middle = [float(line.split('\t')[0]),float(line.split('\t')[1][:-1])]
    #readshift
    f = open(path_align_file, 'r')
    shiftI = list()
    i=-1
    for line in f:
        if 'Source' in line:
            if i == -1:
                shiftI.append([int(line.split(' ')[-1]),middle])
            shiftI.append([int(line.split(' ')[2])])
            i=1
        elif i == 1:
            shiftI[-1].append([float(line.split('\t')[0]),float(line.split('\t')[1][:-1])])
            i = 0
    f.close()
    starting_slice = shiftI[0][0]
    shiftI.sort()
    a = []
    for i, shift in enumerate(shiftI):
        a.append(shift[1])        
    shiftI=(np.array(a)-middle)
    shiftIcumu = []
    for i, sh in enumerate(shiftI):
        if i < starting_slice:
            shiftIcumu.append(np.sum(shiftI[i:starting_slice],axis=0))
        else:
            shiftIcumu.append(np.sum(shiftI[starting_slice:i+1],axis=0))
    shiftIcumu = np.array(shiftIcumu)
    shiftIcumu=np.array([shiftIcumu[::,1],shiftIcumu[::,0]]).T
    
    return shiftIcumu
    
    
def compare_results(specs,results,sum_elements=False,
        normalize=False,plot_result=True):
    """
    Plot different results side by side
    
    The results are found in 'mapped.mapped_parameters.Sample['results_name']'.
    They need to have the same dimension
    
    Parameters
    ----------
    specs: list || list of list
        The list (list of list) of spectra containing the results.
        
    results: list || list of list
        The list (list of list) of name of the results (or a list of images).
        
    normalize: bool    
        If True, each result are normalized.
        
    plot_result : bool
        If True (default option), plot the result. If False, return 
        the result.           
    
    """
    if isinstance(specs[0], list):
        check = []
        for j, spec in enumerate(specs):
            check_temp = []
            for i,s in enumerate(spec):
                if isinstance(results[j][i],str) is False:
                    temp = results[j][i].deepcopy()
                elif normalize:
                    temp = s.normalize_result(results[j][i])
                else:
                    temp = copy.deepcopy(s.mapped_parameters.Sample[results[j][i]]) 
                temp = utils.stack(temp)
                if sum_elements:
                    temp = temp.sum(1)       
                check_temp.append(temp)            
            check.append(utils.stack(check_temp,
                axis=temp.axes_manager.signal_axes[0].name))
            
        check = utils.stack(check,axis=temp.axes_manager.signal_axes[1].name)
        
    elif isinstance(specs[0], list) is False:
        check = []
        for i,s in enumerate(specs):
            if isinstance(results[i],str) is False:
                temp = results[i].deepcopy()
            elif normalize:
                temp = s.normalize_result(results[i])
            else:
                temp = copy.deepcopy(s.mapped_parameters.Sample[results[i]]) 
            temp = utils.stack(temp) 
            if sum_elements:
                temp = temp.sum(1)       
            check.append(temp)
            
        check = utils.stack(check,axis=temp.axes_manager.signal_axes[0].name)
    else:
        raise ValueError("resutls are not a list")   
 
    
    
    check.mapped_parameters.title = 'Compared Results'
    if plot_result: 
        check.plot(navigator=None)
    else:
        return check
        
        
def _histo_data_plot(data,bins = 10):
    """Return data ready to plot an histogram, with a step style
    
    Parameters
    ----------    
    data: np.array
        the data to use
        
    bins: int
        the number of bins
        
    Returns
    -------    
    center: np.array
        the position of the bins
        
    hist: np.array
        the number of conts in the bins
        
    See also
    --------    
    np.histogram
    """
    
    hist1, bins = np.histogram(data,bins)
    hist = np.append(np.append(np.array([0]),
        np.array(zip(hist1,hist1)).flatten()),[0])
    center = np.array(zip(bins ,bins )).flatten()
    return center, hist
    
def plot_histogram_results(specs,element,results,bins = 10,normalize=True):
    """
    Plot the histrogram for different results for one element.
    
    The results are found in 'mapped.mapped_parameters.Sample['results_name']'.
        
    Paramters
    ---------
    
    specs: list
        The list of spectra containing the results.
        
    element: str
        The element to consider. 'all' return the sum over all elements.
        
    results: list 
        The list of name of the results (or a list of images).        
        
    bins: int
        the number of bins
        
    normalize: bool
        
    """
    
    fig = plt.figure()
    for i, spec in enumerate(specs):
        if element == 'all':
            re = copy.deepcopy(spec.mapped_parameters.Sample[results[i]])
            re = utils.stack(re)
            re = re.sum(1)
            re.mapped_parameters.title = 'Sum ' +  results[i] + ' ' + spec.mapped_parameters.title
        elif isinstance(results[i],str):
            re = spec.get_result(element,results[i])           
        else:
            re = results[i].deepcopy()
        data = re.data.flatten()
        center, hist1 = _histo_data_plot(data,bins)
        if normalize:
            hist1 = hist1 / float(hist1.sum())
        plt.plot(center, hist1, label = re.mapped_parameters.title)
    plt.legend()
    fig.show()
    
    return fig



    
    
        

    

    
    
    
        
    
    
    
