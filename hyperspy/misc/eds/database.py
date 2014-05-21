import os
import numpy as np

from hyperspy.misc.config_dir import config_path

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
        return _load_in_database('specImg3DBinPCAre46.hdf5')
    elif which_spec == 'TEM':
        return _load_in_database('robert_a5.hdf5')
    elif which_spec == 'SEM':
        return _load_in_database('specImg3D46.hdf5')
    elif which_spec == 'Ti_SEM':
        return _load_in_database('TiFeNi1h.hdf5')
    elif which_spec == 'rpl':
        return _load_in_database('jonas1h.rpl')
    elif which_spec == 'noisy':
        return _load_in_database('AlZn__040.rpl')


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
        return _load_in_database('specImg3DBinPCArec.hdf5')
    elif which_spec == 'TEM':
        return _load_in_database('cate_3D_bin_reduced.hdf5')


def image2D(which_spec='SEM'):
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


def image3D():
    """
    load RR SE (10:20)
    """
    return _load_in_database('2img3DA.hdf5')


def result3D():
    """
    load RR 2 3D
    """
    return _load_in_database('2res3DrsAH.hdf5', result=True)


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
