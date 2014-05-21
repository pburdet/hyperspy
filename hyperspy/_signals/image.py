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
