import math
import numpy as np
import execnet
import os
import copy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import hyperspy.utils

from hyperspy.misc.eds.elements import elements as elements_db
from hyperspy.misc.config_dir import config_path
import hyperspy.misc.units_converter as units_converter
import hyperspy.components as components
    
def _get_element_and_line(Xray_line):
    lim = Xray_line.find('_')
    return Xray_line[:lim], Xray_line[lim+1:]

def get_FWHM_at_Energy(energy_resolution_MnKa,E):
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
    E_ref = elements_db['Mn']['Xray_energy']['Ka']
    
    
    FWHM_e = 2.5*(E-E_ref)*1000 + FWHM_ref*FWHM_ref
   
    return math.sqrt(FWHM_e)/1000 # In mrad

    
def get_index_from_names(self,axis_names,index_name,axis_name_in_mp=True):
    """Get the index of an axis that is link to a list of names.
    
    Parameters
    ----------
    
    axis_names: list of str | str
        the list name corresponding to the axis
        
    index_name: str
        The name of the index to find
        
    axis_name_in_mp: bool
        if axis_name is in mapped_parameters.Sample.
        
    """
    if axis_name_in_mp==True:
        axis_names = self.mapped_parameters.Sample[axis_names]
    
    for i, name in enumerate(axis_names):
        if name == index_name:
            return i


    
def phase_inspector(self,bins=[20,20,20],plot_result=True):
    #must go in Image
    """
    Generate an binary image of different channel
    """
    from hyperspy import utils
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
        compo_at='auto',
        density='auto',
        detector='Si(Li)',
        gateway='auto'):
    #must create a class, EDS simulation
    #to be retested, det still here
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
    spec = signals.EDSSEMSpectrum(np.zeros(1024))
    
    if mp == 'gui':             
        if elements == 'auto':
            raise ValueError( 'Elements need to be set (set_elements) ' +  
             'with gui option')
            return 0
        else:
            spec.set_microscope_parameters()      
            spec.set_elements(elements) 
            spec.add_lines() 
        mp = spec.mapped_parameters        
    else :
        spec.mapped_parameters = mp.deepcopy()
        mp = spec.mapped_parameters
        
    if elements == 'auto':        
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError( 'Elements need to be set (set_elements)') 
            return 0  
    else: 
        mp.Sample.elements = elements
        
    if compo_at == 'auto':        
        compo_at = []
        #Not tested. atm vs w
        #if hasattr(mp.Sample, 'quant'):
        #    for elm in elements:
        #        compo_at.append(float(spec.get_result(elm,'quant').data))
        #else:       
        for elm in elements:
            compo_at.append(1./len(elements))
    mp.Sample.compo_at = compo_at
            
    if density == 'auto':
        density = utils.eds.density_from_composition(elements, compo_at)
    mp.Sample.density = density
        
    e0 = mp.SEM.beam_energy
    tilt = np.radians(mp.SEM.tilt_stage)
    #tilt = np.radians(abs(mp.SEM.tilt_stage))
    ltime = mp.SEM.EDS.live_time
    elevation =np.radians(mp.SEM.EDS.elevation_angle)
    azim = np.radians(90-mp.SEM.EDS.azimuth_angle)
    #if mp.SEM.EDS.azimuth_angle==90:
    #    tilt = np.radians(abs(mp.SEM.tilt_stage))
    TOangle = np.radians(spec.get_take_off_angle())
    #print TOA(spec)
    compo_wt = units_converter.atomic_to_weight(elements,compo_at)

        
    if gateway == 'auto':
        gateway = get_link_to_jython()
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
        e0 =  """ + str(e0) + """ 
        dose =  """ + str(dose) + """
        tilt = """ + str(tilt) + """ 
        tiltD = tilt
        if tilt < 0:
            #tilt cannot be negative
            tiltD = -tiltD
        live_time = """ + str(ltime) + """
        elevation = """ + str(elevation) + """
        azim = """ + str(azim) + """
        TOA = """ + str(TOangle) + """

        nTraj = """ + str(nTraj) + """          
        
        #Position of detector and sample (WD in km, d-to-crystal in m)
        prop = epq.SpectrumProperties()
        
        prop.setDetectorPosition(elevation, azim, 0.005, 2e-5)
        #if tilt < 0:
        #    prop.setDetectorPosition(TOA, 0.0, 0.005, 2e-5)
        #else : 
        #    prop.setDetectorPosition(TOA, 0.0, 0.005, 2e-5)
        posi = prop.getArrayProperty(epq.SpectrumProperties.DetectorPosition)
        posi = [posi[0]/1000.,posi[1]/1000.,posi[2]/1000.]
        origin = [0.0,0.0,2e-5]
        z0 = origin[2]
        
        det = dtsa2.findDetector('""" + detector + """')  
        prop = det.getDetectorProperties()
        prop.setPosition(posi)
        
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
            nm.MultiPlaneShape.createSubstrate([0.0,math.sin(tilt),-math.cos(tilt)], origin) )
        # Add event listeners to model characteristic radiation
        #monteb.rotate([0,0,z0], -tilt,0.0,0.0)
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
        elif i==1:
            offset = item
        elif i==2:
            reso = item
        else:
            datas.append(item)
        

    spec.data = np.array(datas)
    spec.get_dimensions_from_data() 
    
    spec.mapped_parameters.SEM.EDS.energy_resolution_MnKa = reso
    spec.axes_manager[0].scale = scale / 1000
    spec.axes_manager[0].offset = offset
    spec.axes_manager[0].name = 'Energy'
    spec.axes_manager[0].units = 'keV'
    spec.mapped_parameters.title = 'Simulated spectrum'
    
    spec.mapped_parameters.add_node('simulation')
    spec.mapped_parameters.simulation.nTraj = nTraj 
    #mp.signal_origin = "simulation"

    return spec
    
def simulate_Xray_depth_distribution(nTraj,bins=120,mp='gui',
        elements='auto',
        Xray_lines='auto',
        compo_at='auto',
        density='auto',
        detector='Si(Li)',
        gateway='auto'):
    #must create a class, EDS simulation
    #check if all param well stored
    #dim*cos(tilt)
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
    
    Xray_lines: list of str | 'auto'
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
            raise ValueError( 'Elements need to be set (set_elements) ' +  
             'with gui option')
            return 0
        else:
            spec.set_elements(elements) 
            spec.set_lines() 
        mp = spec.mapped_parameters        
    else :
        spec.mapped_parameters = mp.deepcopy()
        mp = spec.mapped_parameters
        
    if elements == 'auto':        
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError( 'Elements need to be set (set_elements)')   
            
    if Xray_lines == 'auto':
        if hasattr(mp.Sample, 'Xray_lines'):
            Xray_lines = list(mp.Sample.Xray_lines)
        else:
            raise ValueError( 'Xray_lines need to be set (set_lines)')
        
    if compo_at == 'auto':        
        compo_at = []
        #if hasattr(mp.Sample, 'quant'):
        #    for elm in elements:
        #        compo_at.append(float(spec.get_result(elm,'quant').data))
        #else:       
        for elm in elements:
            compo_at.append(1./len(elements))
            
    if density == 'auto':
        density = utils.eds.density_from_composition(elements, compo_at)
        
    e0 = mp.SEM.beam_energy
    tilt = np.radians(mp.SEM.tilt_stage)
    ltime = mp.SEM.EDS.live_time
    elevation =np.radians(mp.SEM.EDS.elevation_angle)
    azim = np.radians(90-mp.SEM.EDS.azimuth_angle)
    compo_wt = units_converter.atomic_to_weight(elements,compo_at)
 
        
    if gateway == 'auto':
        gateway = get_link_to_jython()
    channel = gateway.remote_exec("""   
        import dtsa2
        import math
        epq = dtsa2.epq 
        epu = dtsa2.epu
        nm = dtsa2.nm
        elements = """ + str(elements) + """
        Xray_lines = """ + str(Xray_lines) + """ 
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
        
        for Xray_line in Xray_lines:        
            lim = Xray_line.find('_')
            el = getattr(dtsa2.epq.Element,Xray_line[:lim])            
            li = Xray_line[lim+1:]
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
    datas = np.reshape(datas[:-1],(len(Xray_lines),2,bins))
    datas = np.rollaxis(datas,1,0)
        
    frz = signals.Spectrum(np.array(datas))
    frz.mapped_parameters.SEM = mp.SEM 
    mp = frz.mapped_parameters
    mp.add_node('Sample')
    mp.Sample.elements = elements
    mp.Sample.compo_at = compo_at
    mp.Sample.Xray_lines = Xray_lines 
    mp.Sample.density = density

    frz.axes_manager[0].name = 'Generated|Emitted'
    frz.axes_manager[1].name = 'Xray_lines'
    #frz.axes_manager[1].units = 'keV'
    frz.axes_manager[2].name = 'Depth'
    frz.axes_manager[2].units = 'nm'
    frz.axes_manager[2].scale = dim / bins * 1000000000
    mp.title = 'Simulated Depth distribution'
    
    mp.add_node('simulation')
    mp.simulation.nTraj = nTraj  
    #mp.signal_origin = "simulation"
    mp.simulation.software = 'NistMonte' 

    return frz


def get_link_to_jython():
    #must go in IO
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
    #must desappear, result == spec
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
        ##Need to change
        ##number_of_parts=len(mp.Sample.elements)
        l_time = std.mapped_parameters.SEM.EDS.live_time
        ##number_of_parts=len(mp.Sample.Xray_lines)
        temp = std.split(axis=0,number_of_parts=number_of_parts) 
    elif len(std.data.shape) == 1:
        temp = std.split(axis=0,number_of_parts=number_of_parts) 
    else:
        #temp = std.split(axis=1,number_of_parts=number_of_parts)
        temp = std.split(axis=-3,number_of_parts=number_of_parts)
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
    #must be in Image
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
        channel.send(1)
    
    """)
    for i, item in enumerate(channel):
        item = item  
            
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
        normalize=False,plot_result=True,expand=False):
    #must be the main function in Image, specs = image. EDSSpec for results
    """
    Plot different results side by side
    
    The results are found in 'mapped.mapped_parameters.Sample['results_name']'.
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
            results = [results]*len(specs)
            for i, spec in enumerate(specs):
                specs[i] = [specs[i]]*len(results[0])
        else:
            if isinstance(results[0], list):
                specs = [[specs]*len(results[0])]*len(results)
            else:
                specs = [specs]*len(results)
    
        
    if isinstance(specs[0], list):
        if isinstance(results,list) is False:
            results = [[results]*len(specs[0])]*len(specs) 
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
        check.axes_manager[-2].name += ' + results'
        check.axes_manager[-1].name += ' + specs'
        
    elif isinstance(specs, list):
        if isinstance(results,list) is False:
            results = [results]*len(specs)
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
        raise ValueError("specs is not a list")   
 
    
    
    check.mapped_parameters.title = 'Compared Results'
    if plot_result: 
        check.plot(navigator=None)
    else:
        return check

    
def compare_histograms_results(specs,
    element,
    results,
    bins = 10,
    normalizeI=False,
    normalizex=False,
    legend_labels='auto',
    colors='auto',
    line_styles='auto'):
    """
    Plot the histrogram for different results for one element.
    
    The results are found in 'mapped.mapped_parameters.Sample['results_name']'.
        
    Paramters
    ---------
    
    specs: list
        The list of spectra containing the results.
        
    element: str
        The element to consider. 'all' return the sum over all elements.
        
    results: list || str
        The list of name of the results (or a list of images).        
        
    bins: int
        the number of bins
        
    normalizeI: bool
        nomralize the intensity
        
    normalizex: bool
        nomralize over all the results
    
    legend_labels: 'auto' | list | None
        If legend_labels is auto, then the indexes are used.
        
    colors: list
        If 'auto', automatically selected, eg: ('red','blue')
        
    line_styles: list
        If 'auto', continuous lines, eg: ('-','--','steps','-.',':')
        
    """
    from hyperspy import utils
    specs = copy.deepcopy(specs)
    if isinstance(results,list) is False:
        results = [results]*len(specs)
    elif isinstance(specs,list) is False:        
        specs = [specs]*len(results)
    else:
        dim_results = len(results)        
        results = np.repeat(results,len(specs))
        specs = specs*dim_results
    hists=[]
    for i, spec in enumerate(specs):
        if element == 'all':
            re = copy.deepcopy(spec.mapped_parameters.Sample[results[i]])
            re = utils.stack(re)
            re = re.sum(1)
            re.mapped_parameters.title = 'Sum ' +  results[i] + ' ' + spec.mapped_parameters.title
        elif isinstance(results[i],str):
            if normalizex:
                re = spec.normalize_result(results[i])[list(spec.mapped_parameters.Sample.elements).index(element)]
            else:
                re = spec.get_result(element,results[i]) 
            re.mapped_parameters.title = element + ' ' +  results[i] + ' ' +  spec.mapped_parameters.title
        else:
            re = results[i].deepcopy()
            #print 'Normalise x not available yet'
            re.mapped_parameters.title = (element + ' ' +  
                re.mapped_parameters.title + ' ' +  spec.mapped_parameters.title)
        #data = re.data.flatten()
        #center, hist1 = _histo_data_plot(data,bins)
        hist_tmp = re.get_histogram(bins)
        if normalizeI:
            hist_tmp = hist_tmp / float(hist_tmp.sum(0).data)
        hists.append(hist_tmp)
        
    compare_signal(hists,legend_labels=legend_labels,colors=colors,
        line_styles=line_styles)
    
    
def compare_histograms(imgs,bins=10,
    legend_labels='auto',
    colors='auto',
    line_styles='auto'):
    """Compare the histogram of the list of image
    
    Parameters
    ----------
    
    bins: int
        the number of bins (channel)
        
    legend_labels: 'auto' | list | None
        If legend_labels is auto, then the indexes are used.
        
    colors: list
        If 'auto', automatically selected, eg: ('red','blue')
        
    line_styles: list
        If 'auto', continuous lines, eg: ('-','--','steps','-.',':')
        
    """
    hists=[]
    for img in imgs:
        hists.append(img.get_histogram(bins))
    compare_signal(hists,legend_labels=legend_labels,colors=colors,
        line_styles=line_styles)   

    
def compare_signal(specs,
    indexes=None,
    legend_labels='auto',
    colors='auto',
    line_styles='auto'):
    """Compare the signal from different indexes or|and from different
    spectra.
    
    Parameters
    ----------
    
    specs: list | spectrum
        A list of spectra or a spectrum
    
    indexes: list | None
        The list of indexes to compares. If None, specs is a list of 
        1D spectra that are ploted together
        
        
    legend_labels: 'auto' | list | None
        If legend_labels is auto, then the indexes are used.
        
    colors: list
        If 'auto', automatically selected, eg: ('red','blue')
        
    line_styles: list
        If 'auto', continuous lines, eg: ('-','--','steps','-.',':')
        
    Returns
    -------
    
    figure
        
    """

    if indexes == None:
        nb_signals = len(specs)
    elif isinstance(indexes[0],list) is False and isinstance(indexes[0],tuple) is False:     
        nb_signals = len(specs)
        indexes = [indexes]*nb_signals
    else :
        nb_signals=len(indexes)
            
    if colors == 'auto':
        colors = ['red','blue','green','orange','violet','magenta',
        'cyan','violet','black','yellow','pink']
        colors+=colors
        colors+=colors
    elif isinstance(colors,list) is False:
        colors = [colors]* nb_signals
    if line_styles == 'auto':
        line_styles = ['-']* nb_signals
    elif isinstance(line_styles,list) is False:
        line_styles = [line_styles]* nb_signals

    fig = plt.figure()
    if legend_labels == 'auto': 
        legend_labels = []
        if isinstance(specs,list) or isinstance(specs,tuple):
            for spec in specs: legend_labels.append(spec.mapped_parameters.title)
        else:
            for index in indexes:  legend_labels.append(str(index))
    #for i, index in enumerate(indexes):
    for i in range(nb_signals):
        if isinstance(specs,list) or isinstance(specs,tuple):
            tmp = specs[i]
        else :
            tmp = specs
            
        if indexes != None:
            for ind in indexes[i]: tmp = tmp[ind]

        maxx = (len(tmp.data)-1)*tmp.axes_manager[0].scale+tmp.axes_manager[0].offset
        xdata = mlab.frange(tmp.axes_manager[0].offset,maxx,
                            tmp.axes_manager[0].scale,npts=len(tmp.data))
        plt.plot(xdata,tmp.data, color = colors[i],ls=line_styles[i])
    plt.ylabel('Intensity')
     
    plt.xlabel(str(tmp.axes_manager[0].name) + ' (' + str(tmp.axes_manager[0].units) + ')')
    
    if legend_labels is not None:
        plt.legend(legend_labels) 
    fig.show() 
    
    return fig



    
def simulate_linescan(nTraj,
    compos_at,
    min_max,
    lscan_scale,
    lscan_axis='x',
    elements = 'auto',    
    density = 'auto',
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
            raise ValueError( 'Elements need to be set (set_elements) ' +  
             'with gui option')
            return 0
        else:
            spec.set_elements(elements) 
            spec.set_lines() 
        mp = spec.mapped_parameters        
    else :
        spec.mapped_parameters = mp.deepcopy()
        mp = spec.mapped_parameters
        
    if elements == 'auto':        
        if hasattr(mp.Sample, 'elements'):
            elements = list(mp.Sample.elements)
        else:
            raise ValueError( 'Elements need to be set (set_elements)')  
            return 0
    else: 
        mp.Sample.elements = elements
            
    if density == 'auto':
        density = []
        for compo_at in compos_at:
            density.append(utils.eds.density_from_composition(elements, compo_at))
            
    mp.Sample.compo_at = compo_at
    mp.Sample.density = density
        
    e0 = mp.SEM.beam_energy
    tilt = np.radians(mp.SEM.tilt_stage)
    ltime = mp.SEM.EDS.live_time
    elevation =np.radians(mp.SEM.EDS.elevation_angle)
    azim = np.radians(90-mp.SEM.EDS.azimuth_angle)
    compos_wt = []
    for compo_at in compos_at:
        compos_wt.append(units_converter.atomic_to_weight(elements,compo_at))
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
            interface_xyz =""" + str(interface_xyz*1e-6) + """
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
            elif i==1:
                offset = item
            elif i==2:
                reso = item
            else:
                datas.append(item) 
        return datas, scale, offset, reso
        
        
    spec_datas = []
    for thck in mlab.frange(min_max[0],min_max[1],lscan_scale): 
        if thck <0 and lscan_axis == 'z':
            tmp, scale, offset, reso = simu_film(0)
        else:
            tmp, scale, offset, reso  = simu_film(thck)
        spec_datas.append(tmp)
    spec = signals.EDSSEMSpectrum(np.array(spec_datas))    
    spec.mapped_parameters = mp   
     
    mp.SEM.EDS.energy_resolution_MnKa = reso
    spec.axes_manager[-1].scale = scale / 1000
    spec.axes_manager[-1].offset = offset
    spec.axes_manager[-1].name = 'Energy'
    spec.axes_manager[-1].units = 'keV'
    spec.axes_manager[0].scale = lscan_scale
    spec.axes_manager[0].offset = min_max[0]
    spec.axes_manager[0].name = 'Scan'
    spec.axes_manager[0].units = '${\mu}m$'
    spec.mapped_parameters.title = 'Simulated linescan along ' + lscan_axis 
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
    bottom, top = (int(np.floor(shifts[:,0].min())) if 
                            shifts[:,0].min() < 0 else None,
                   int(np.ceil(shifts[:,0].max())) if 
                            shifts[:,0].max() > 0 else 0)
    right, left = (int(np.floor(shifts[:,1].min())) if 
                            shifts[:,1].min() < 0 else None,
                   int(np.ceil(shifts[:,1].max())) if 
                            shifts[:,1].max() > 0 else 0)
    shifts = -shifts
    return top, bottom, left, right
    
def plot_orthoview(image,
    index,
    plot_index=False,
    space=2,
    plot_result=True):
    """
    Plot an orthogonal view of a 3D images
    
    Parameters
    ---------
    
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
    """
    from hyperspy import signals
    from hyperspy import utils
    image = image.deepcopy()
    dim = image.axes_manager.shape
    if len(dim)!=3:
        raise ValueError('Needs a 3D image')   
    
    scalez = image.axes_manager[0].scale
    scalex = image.axes_manager[1].scale
    if scalez > scalex:
        scale_fact= int(scalez / scalex)                
        image.data = np.repeat(image.data,int(scalez / scalex),axis=0)
        image.get_dimensions_from_data()
        dim = image.axes_manager.shape

    map_color = plt.get_cmap()
    if map_color.name == 'RdYlBu_r':
        mean_img= image.mean(0).mean(0).mean(0).data
    else:
        mean_img= image.max(0).max(0).max(0).data*0.88
    a=image[index[2]*scale_fact].deepcopy()
    b= image[::,index[0]].as_image([0,1]).deepcopy()
    c = image[::,::,index[1]].as_image([1,0]).deepcopy()
    if plot_index:
        #a.data[index[0]] = np.ones(dim[1])*mean_img
        #a.data[::,index[1]] = np.ones(dim[2])*mean_img   
        a.data[::,index[0]] = np.ones(dim[2])*mean_img
        a.data[index[1]] = np.ones(dim[1])*mean_img   
        b.data[index[1]] = np.ones(dim[0])*mean_img          
        b.data[::,index[2]*scale_fact] = np.ones(dim[2])*mean_img          
        c.data[index[2]*scale_fact] = np.ones(dim[1])*mean_img
        c.data[::,index[0]] = np.ones(dim[0])*mean_img
        
    im= utils.stack([a,
        signals.Image(np.ones([dim[2],space])*mean_img),b],axis=0)
    im2= utils.stack([c,
        signals.Image(np.ones([dim[0],dim[0]+space])*mean_img)],axis=0)
    im = utils.stack([im,
        signals.Image(np.ones([space,dim[1]+dim[0]+space])*mean_img),im2],axis=1)
    #Why I need to do that
    im.axes_manager[0].offset=0
    im.axes_manager[0].offset=0
    
    if plot_result:
        fig = im.plot()
        return fig 
    else:
        return im
        
def get_contrast_brightness_from(img,reference):
    """Set the contrast/brightness of an image to be the same as a reference.
    
    Fit the histogram of the image on the histogram of the reference to 
    get the change in contrast bightness
    
    Parameters
    ---------
    
    img: Signal
        The signal fo which the contrast need to be adjsuted
        
    reference: Signal
        The contrast/brightness reference
    """
    from hyperspy.hspy import create_model  
    
    img = img.deepcopy()
    
    hist_img=img.get_histogram(bins=50)
    hist_ref=reference.get_histogram(bins=50)
    
    posmax_ref=list(hist_ref.data).index(max(hist_ref.data))
    posmax_img=list(hist_img.data).index(max(hist_img.data))

    m = create_model(hist_img)
    fp = components.ScalableFixedPattern(hist_ref)

    fp.xscale.value= (hist_ref.axes_manager[0].scale/
                hist_img.axes_manager[0].scale)
    fp.shift.value = (hist_ref.axes_manager[0].scale*posmax_ref/
                     hist_img.axes_manager[0].scale/posmax_img)

    fp.set_parameters_free(['xscale','shift'])
    fp.set_parameters_not_free(['yscale'])
    m.append(fp)          
    m.multifit() 

    img*=fp.xscale.value
    img-=fp.shift.value

    return img
