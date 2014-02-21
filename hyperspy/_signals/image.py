# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import numpy as np

from hyperspy.signal import Signal


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

        See also:
        -----

        skimage.filter.denoise_tv_chambolle
        skimage.filter.denoise_tv_bregman

        """

        import skimage.filter

        if method == 'bregman':
            img = self.apply(skimage.filter.denoise_tv_bregman, weight=weight,
                             eps=eps, max_iter=n_iter_max)
        elif method == 'chambolle':
            img = self.apply(skimage.filter.denoise_tv_chambolle, img.data,
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
        img = self.apply(scipy.ndimage.convolve, weights=kernel)
        return img

    def plot_3D_iso_surface(self,
                            threshold,
                            outline=True,
                            figure=None,
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
        kwargs:
            other keyword arguments of mlab.pipeline.iso_surface (eg.
            'color=(R,G,B)','name=','opacity=','transparent=',...)

        Example
        --------

        >>> # Plot two iso-surfaces from one stack of images
        >>> [fig,src,iso] = img.plot_3D_iso_surface([0.2,0.8])
        >>> # Plot an iso-surface from another stack of images
        >>> [fig,src2,iso2] = img2.plot_3D_iso_surface(0.2,figure=fig)
        >>> # Change the threshold of the second iso-surface
        >>> iso2.contour.contours=[0.3, ]

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

        img_res = self.deepcopy()

        img_data = img_res.data
        img_data = np.rollaxis(img_data, 0, 3)
        img_data = np.rollaxis(img_data, 0, 2)
        src = mlab.pipeline.scalar_field(img_data)
        src.name = img_res.metadata.title

        if hasattr(threshold, "__iter__") is False:
            threshold = [threshold]

        threshold = [img_data.max() - thr * img_data.ptp()
                     for thr in threshold]

        scale = [1 / img_res.axes_manager[i].scale for i in [1, 2, 0]]
        src.spacing = scale

        iso = mlab.pipeline.iso_surface(src,
                                        contours=threshold, **kwargs)
        iso.compute_normals = False

        if outline:
            # mlab.outline(color=(0.5,0.5,0.5))
            mlab.outline()

        return figure, src, iso
