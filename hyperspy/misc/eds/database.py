import os
import numpy as np

from hyperspy.misc.config_dir import config_path
from hyperspy.misc.utils import DictionaryTreeBrowser


def _load_in_database(name, result=False):
    from hyperspy.misc.eds import utils as utils_eds
    from hyperspy.io import load
    foldername = os.path.join(config_path, 'database//' + name)
    if result:
        return utils_eds.load_EDSSEMSpectrum(foldername)
    else:
        return load(foldername)


def spec1D(which_spec='BAM'):
    """
    load 1D spec

    Parameters
    ----------

    which_spec: {'BAM','msa','noisy','TEM'}
        if BAM: bam sample
        if msa: GnMeba test, coin of euro
        if noisy: 1 pixel in AlZn
        if TEM: sum spec from Robert A5
        if TEM_nico: IMT sample from Nico

    """

    if which_spec == 'BAM':
        return _load_in_database('SEM/1DS_bam.hdf5')
    elif which_spec == 'msa':
        return _load_in_database('SEM/1DS_GNmeba_test.msa')
    elif which_spec == 'noisy':
        return _load_in_database('SEM/1DS_1pix_AlZn.msa')
    elif which_spec == 'TEM':
        return _load_in_database('TEM/1Ds_Robert.hdf5')
    elif which_spec == 'TEM_nico':
        return _load_in_database('TEM/1Ds_Nico_IMT.hdf5')


def spec3D(which_spec='PCA_SEM'):
    """
    load 3D spec

    Parameters
    ----------

    which_spec: {'PCA_SEM','SEM','Ti_SEM','rpl','noisy'}
        if 'PCA_SEM', load RR 46 PCA rec
        if 'TEM', 0 degree roabert A5
        if 'SEM', load RR 46 no PCA
        ifs 'Ti_SEM', load TiFeNi no PCA jonas1h croped (::,:12)
        if 'rpl', jonas1h raw
        if 'noisy', AlZn 40 .rpl, see noisy 1D
    """

    if which_spec == 'PCA_SEM':
        return _load_in_database('SEM/3Ds_specImg3DBinPCAre46.hdf5')
    elif which_spec == 'TEM':
        return _load_in_database('TEM/3Ds_robert_a5.hdf5')
    elif which_spec == 'SEM':
        return _load_in_database('SEM/3Ds_specImg3D46.hdf5')
    elif which_spec == 'Ti_SEM':
        return _load_in_database('SEM/3Ds_TiFeNi1h.hdf5')
    elif which_spec == 'rpl':
        return _load_in_database('SEM/3Ds_jonas1h.rpl')
    elif which_spec == 'noisy':
        return _load_in_database('SEM/3Ds_AlZn__040.rpl')


def spec4D(which_spec='PCA_SEM'):
    """
    load RR PCA rec (10:15) or Cat (TEM) no PCA

    Parameters
    ----------

    which_spec: {'PCA_SEM','TEM'}
        if 'PCA_SEM', load RR (slices 10:15) PCA rec
        if 'TEM', load Cat (TEM) no PCA
    """
    if which_spec == 'PCA_SEM':
        return _load_in_database('SEM/4Ds_specImg3DBinPCArec.hdf5')
    elif which_spec == 'TEM':
        return _load_in_database('TEM/4Ds_cate_bin_reduced.hdf5')


def image2D(which_spec='SEM'):
    """
    load 2D image

    Parameters
    ----------
    which_spec: {'SEM','Ti_SEM','lena'}
        if SEM, RR SE 46 (TLD SE)
        if Ti_SEM, jonas1h SE image (inLens, bck corrected, croped)
        if lena, scipy.misc.lena
    """
    if which_spec == 'SEM':
        return _load_in_database('SEM/2Dim_img46.hdf5')
    elif which_spec == 'Ti_SEM':
        return _load_in_database('SEM/2Dim_SE_imTiFeNi1h.hdf5')
    elif which_spec == 'lena':
        import scipy.ndimage
        from hyperspy.signals import Image
        return Image(scipy.misc.lena())


def image3D(which_spec='SEM'):
    """
    load 3D image

    Parameters
    ----------
    which_spec: {'SEM','tilt_TEM'}
        if 'SEM': RR SE (10:20)
        if 'tilt_TEM' : nico tilt series 1 NP sample
    """
    if which_spec == 'SEM':
        return _load_in_database('SEM/3Dim_2img3DA.hdf5')
    elif which_spec == 'tilt_TEM':
        return _load_in_database('TEM/3Dim_tilt_nico_adf.hdf5')


def result3D():
    """
    load RR 2 3D
    """
    return _load_in_database('SEM/3Dres_2res3DrsAH.hdf5', result=True)


def detector_efficiency_INCA(index=4):
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
        det = signals.EDSSEMSpectrum(data[645:5646])
        det.axes_manager[-1].scale = 0.001
        det.axes_manager[-1].offset = 0.0
    else:
        det = signals.Spectrum(data[645:3646])
        det.axes_manager[-1].scale = 0.01
        det.axes_manager[-1].offset = 0.0
    det.metadata.General.title = det_name[index]
    det.axes_manager[-1].units = "keV"
    det.axes_manager[-1].name = "Energy"
    return det


def detector_layers_brucker(microscope_name='osiris'):
    """
    Import detector layers descrption from brucker file

    Parameters
    ----------
    microscope_name: str
        name of the microscope ("from_p_buffat",'osiris')

    Return
    ------
    elements (str),thicknesses_layer (nm),thickness_detector (mum)

    """

    if microscope_name == "from_p_buffat":
        return ['Al', 'Si', 'Si', 'O'], np.array([30., 40., 80., 40.]), 0.45
    else:
        from hyperspy import utils

        foldername = os.path.join(config_path,
                                  'database//brucker\\SpectraList_' + microscope_name + '.xml')

        import base64
        import zlib

        f = open(foldername)
        a = f.readlines()
        b = filter(lambda x: '<DetLayers>' in x, a)[0].split(
            '<DetLayers>')[1].split('</DetLayers>')[0].decode('base64').decode('zlib')
        atom = [int(c.split('"')[0]) for c in b.split('Atom="')[1:]]
        thicknesses = [float(c.split('"')[0]) for c in b.split('ss="')[1:]]
        b = filter(lambda x: 'Layer0 Atom' in x, a)[0]
        atom.append(int(b.split('Atom="')[1].split('"')[0]))
        thicknesses.append(float(b.split('ss="')[1].split('"')[0]))
        elements = []
        for at in atom:
            for el in utils.material.elements.keys():
                if utils.material.elements[el].General_properties.Z == at:
                    elements.append(el)
        thicknesses = np.array(thicknesses) * 1e3
        thickness_det = float(filter(lambda x: '<DetectorThickness>' in x, a
                                     )[0].split('ss>')[1].split('</D')[0])

        return elements, thicknesses, thickness_det


def kfactors_brucker(xray_lines='all', microscope_name='osiris_200'):
    """
    Import kfactors from brucker file

    Parameters
    ----------
    xray_lines: list of str
        The name of the X-ray line. If All return a full dictionnaries of value
    microscope_name: str
        name of the microscope

    Return
    ------
    dictionary of kfactors or kfactor, kerror

    """
    from hyperspy import utils
    from hyperspy.misc.eds import utils as utils_eds
    foldername = os.path.join(config_path,
                              'database//brucker\\Current_' + microscope_name + '.esl')
    f = open(foldername)
    a = f.readlines()
    kfactors = []
    for line in ['K', 'L', 'M']:
        kfactors.append(filter(lambda x: '<' + line + '_Factors>' in x, a)[0].split(
            '<' + line + '_Factors>')[1].split('</' + line + '_Factors>')[0].split(','))
        kfactors[-1] = [float(c) for c in kfactors[-1]]
    kerrors = []
    for line in ['K', 'L', 'M']:
        kerrors.append(filter(lambda x: '<' + line + '_Errors>' in x, a)[0].split(
            '<' + line + '_Errors>')[1].split('</' + line + '_Errors>')[0].split(','))
        kerrors[-1] = [float(c) for c in kerrors[-1]]

    dic_el = utils.material.elements
    if hasattr(xray_lines, '__iter__'):
        kfactor = []
        kerror = []
        for xray_line in xray_lines:
            elem, line = utils_eds._get_element_and_line(xray_line)
            iZ = dic_el[elem].General_properties.Z
            if line == 'Ka':
                iline = 0
            elif line == 'La':
                iline = 1
            elif line == 'Ma':
                iline = 2
            kfactor.append(kfactors[iline][iZ])
            kerror.append(kerrors[iline][iZ])
        return kfactor, kerror
    else:
        dic = DictionaryTreeBrowser()
        for el in dic_el.keys():
            iZ = dic_el[el].General_properties.Z
            if 'Xray_lines' in dic_el[el].Atomic_properties:
                for line in dic_el[el].Atomic_properties.Xray_lines.keys():
                    if line == 'Ka':
                        iline = 0
                    elif line == 'La':
                        iline = 1
                    elif line == 'Ma':
                        iline = 2
                    else:
                        iline = None
                    if iline is not None:
                        dic.set_item(
                            el +
                            '.' +
                            line +
                            '.kfactor',
                            kfactors[iline][iZ])
                        dic.set_item(
                            el +
                            '.' +
                            line +
                            '.kfactor_error',
                            kerrors[iline][iZ])
        return dic
