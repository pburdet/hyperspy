import numpy as np
import math
import execnet
import os
import copy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import hyperspy.utils
from hyperspy.misc.config_dir import config_path
import hyperspy.components as components
from hyperspy.misc.elements import elements as elements_db
from hyperspy.misc.eds import database
from functools import reduce
from hyperspy.misc.eds import utils as utils_eds


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
        gateway = utils_eds.get_link_to_jython()
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

    >>> im = database.image2D()
    >>> image_eds.tv_denoise(im,method='chambolle',
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
