# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from hyperspy.signal import Signal
from hyperspy.misc.eds import utils as utils_eds


class Image(Signal):

    """
    """
    _record_by = "image"

    def __init__(self, *args, **kw):
        super(Image, self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)

    def to_spectrum(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_spectrum : a method for the same purpose with more options.
        signals.Image.to_spectrum : performs the inverse operation on images.

        """
        return self.as_spectrum(0 + 3j)

    def plot_3D_iso_surface(self,
                            threshold,
                            outline=True,
                            figure=None,
                            color=None,
                            **kwargs):
        """
        Generate an iso-surface with Mayavi of a stack of images.

        The method uses the mlab.pipeline.iso_surface from mayavi.

        Parameters
        ----------
        threshold: float or list
            The threshold value(s) used to generate the contour(s).
            Between 0 (min intensity) and 1 (max intensity).
        figure: None or mayavi.core.scene.Scene
            If None, generate a new scene/figure.
        outline: bool
            If True, draw an outline.
        colors: None or  (r,g,b)
            None generate different color
        kwargs:
            other keyword arguments of mlab.pipeline.iso_surface (eg.
            'name=','opacity=','transparent=',...)

        Example
        --------

        >>> img = database.image3D()
        >>> # Plot two iso-surfaces from one stack of images
        >>> fig,src,iso = img.plot_3D_iso_surface([0.2,0.8])
        >>> # Plot an iso-surface from another stack of images
        >>> s = database.result3D()
        >>> img2 = s.get_result('Ni','quant')
        >>> fig,src2,iso2 = img2.plot_3D_iso_surface(0.2, figure=fig,
        >>>     outline=False)
        >>> # Change the threshold of the second iso-surface
        >>> iso2.contour.contours=[0.73, ]

        Return
        ------
        figure: mayavi.core.scene.Scene
        src: mayavi.sources.array_source.ArraySource
        iso: mayavi.modules.iso_surface.IsoSurface

        """
        from mayavi import mlab

        if len(self.axes_manager.shape) != 3:
            raise ValueError("This functions works only for 3D stack of "
                             "images.")

        if figure is None:
            figure = mlab.figure()

        colors = [(1, 0, 0), (0, 1, 0), (0, 0.1, 0.9),
                  (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
                  (0.3, 0, 0.7), (0.7, 0, 0.3), (0.0, 0.3, 0.7),
                  (0.0, 0.7, 0.3), (0.3, 0.7, 0.0), (0.7, 0.3, 0.0)]

        img_res = self.deepcopy()

        src = img_res._get_mayavi_scalar_field()
        img_data = self.data

        if hasattr(threshold, "__iter__") is False:
            threshold = [threshold]

            if color is None:
                color = colors[len(figure.children) - 1]

        threshold = [img_data.max() - thr * img_data.ptp()
                     for thr in threshold]

        if color is None:
            iso = mlab.pipeline.iso_surface(src,
                                            contours=threshold, **kwargs)
        else:
            iso = mlab.pipeline.iso_surface(src,
                                            contours=threshold, color=color, **kwargs)
        iso.compute_normals = False

        if outline:
            mlab.outline(color=(0.5, 0.5, 0.5))
            # mlab.outline()

        return figure, src, iso

    def _get_mayavi_scalar_field(self, return_data=False):
        """
        Return a mayavi scalar field from an image

        Parameters
        ----------
        return_data:bool
            If return_data is True, return the data
            if False return the scalarfield

        """
        from mayavi import mlab

        scale = [self.axes_manager[i].scale for i in [1, 2, 0]]

        img_data = self.data
        img_data = np.rollaxis(img_data, 0, 3)
        img_data = np.rollaxis(img_data, 0, 2)
        src = mlab.pipeline.scalar_field(img_data)
        src.name = self.metadata.General.title
        src.spacing = scale
        if return_data:
            return img_data
        else:
            return src

    def tomographic_reconstruction(self,
                                   algorithm='FBP',
                                   tilt_stages='auto',
                                   iteration=1,
                                   parallel=None,
                                   **kwargs):
        """
        Reconstruct a 3D tomogram from a sinogram

        Parameters
        ----------
        algorithm: {'FBP','SART'}
            FBP, filtered back projection
            SART, Simultaneous Algebraic Reconstruction Technique
        tilt_stages: list or 'auto'
            the angles of the sinogram. If 'auto', takes the angles in
            Acquisition_instrument.TEM.tilt_stage
        iteration: int
            The numebr of iteration used for SART
        parallel : {None, int}
            If None or 1, does not parallelise multifit. If >1, will look for
            ipython clusters. If no ipython clusters are running, it will
            create multiprocessing cluster.

        Return
        ------
        The reconstruction as a 3D image

        Examples
        --------
        >>> adf_tilt = database.image3D('tilt_TEM')
        >>> adf_tilt.change_dtype('float')
        >>> rec = adf_tilt.tomographic_reconstruction()
        """
        from hyperspy._signals.spectrum import Spectrum
        # import time
        if parallel is None:
            sinogram = self.to_spectrum().data
        if tilt_stages == 'auto':
            tilt_stages = self.axes_manager[0].axis
        # a = time.time()
        if algorithm == 'FBP':
            # from skimage.transform import iradon
            from hyperspy.misc.borrowed.scikit_image_dev.radon_transform \
                import iradon
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon(sinogram[i], theta=tilt_stages,
                                output_size=sinogram.shape[1], **kwargs)
        elif algorithm == 'SART' and parallel is None:
            from hyperspy.misc.borrowed.scikit_image_dev.radon_transform\
                import iradon_sart
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                     **kwargs)
                for j in range(iteration - 1):
                    rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                         image=rec[i], **kwargs)
        elif algorithm == 'SART':
            from hyperspy._signals.image import isart_multi
            sino, pool, pool_type = \
                utils_eds.get_multi_processing_pool(parallel,
                                                    self.to_spectrum())

            kwargs.update({'theta': tilt_stages})
            data = [[si.data, iteration, kwargs] for si in sino]
            res = pool.map_async(isart_multi, data)
            if pool_type == 'mp':
                pool.close()
                pool.join()
            res = res.get()
            rec = res[0]
            for i in range(len(res)-1):
                rec = np.append(rec, res[i+1], axis=0)

        # print time.time() - a

        rec = Spectrum(rec).as_image([2, 1])
        rec.axes_manager = self.axes_manager.deepcopy()
        rec.axes_manager[0].scale = rec.axes_manager[1].scale
        rec.axes_manager[0].offset = rec.axes_manager[1].offset
        rec.axes_manager[0].units = rec.axes_manager[1].units
        rec.axes_manager[0].name = 'z'
        rec.get_dimensions_from_data()
        return rec


def isart_multi(args):
    from hyperspy.misc.borrowed.scikit_image_dev.radon_transform\
        import iradon_sart
    import numpy as np
    sinogram, iteration, kwargs = args
    rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                    sinogram.shape[1]])
    for i in range(sinogram.shape[0]):
        rec[i] = iradon_sart(sinogram[i], **kwargs)
        for j in range(iteration - 1):
            rec[i] = iradon_sart(sinogram[i],
                                 image=rec[i], **kwargs)
    return rec
