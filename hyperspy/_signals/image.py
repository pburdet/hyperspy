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
                                            contours=threshold,
                                            color=color, **kwargs)
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
                                   relaxation=0.15,
                                   **kwargs):
        """
        Reconstruct a 3D tomogram from a sinogram

        The siongram has x and y as signal axis and tilt as navigation axis

        Parameters
        ----------
        algorithm: {'FBP','SART'}
            FBP, filtered back projection
            SART, Simultaneous Algebraic Reconstruction Technique
        tilt_stages: list or 'auto'
            the angles of the sinogram. If 'auto', take the navigation axis
            value.
        iteration: int
            The numebr of iteration used for SART
        relaxation: float
            For SART: Relaxation parameter for the update step. A higher value
            can improve the convergence rate, but one runs the risk of
            instabilities. Values close to or higher than 1 are not
            recommended.
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
        if parallel is None:
            sinogram = self.to_spectrum().data
        if tilt_stages == 'auto':
            tilt_stages = self.axes_manager[0].axis
        if algorithm == 'FBP':
            from skimage.transform import iradon
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon(sinogram[i], theta=tilt_stages,
                                output_size=sinogram.shape[1], **kwargs)
        elif algorithm == 'SART' and parallel is None:
            from skimage.transform import iradon_sart
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                     **kwargs)
                for j in range(iteration - 1):
                    rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                         image=rec[i], **kwargs)
        elif algorithm == 'SART':
            from hyperspy.misc import multiprocessing
            pool, pool_type = multiprocessing.pool(parallel)
            sino = multiprocessing.split(self.to_spectrum(), parallel, axis=1)
            kwargs.update({'theta': tilt_stages})
            data = [[si.data, iteration, kwargs] for si in sino]
            res = pool.map_sync(multiprocessing.isart, data)
            if pool_type == 'mp':
                pool.close()
                pool.join()
            rec = res[0]
            for i in range(len(res)-1):
                rec = np.append(rec, res[i+1], axis=0)

        rec = Spectrum(rec).as_image([2, 1])
        rec.metadata.General.title = 'Reconstruction from ' + \
            self.metadata.General.title
        rec.axes_manager = self.axes_manager.deepcopy()
        rec.axes_manager[0].scale = rec.axes_manager[1].scale
        rec.axes_manager[0].offset = rec.axes_manager[1].offset
        rec.axes_manager[0].units = rec.axes_manager[1].units
        rec.axes_manager[0].name = 'z'
        rec.get_dimensions_from_data()
        return rec

    def plot_orthoview(self, isotropic_voxel=True):
        """
        Plot an orthogonal view of a 3D images

        Parameters
        ---------
        image: signals.Image
            An image in 3D.
        isotropic_voxel:
            If True, generate a new image, scaling z in order to obtain
            isotropic voxel.
        """
        from hyperspy.misc.eds.image_eds import plot_orthoview_animated
        plot_orthoview_animated(self, isotropic_voxel=isotropic_voxel)

    def plot_orthoview_new(image):
        """
        Plot an orthogonal view of a 3D images

        Parameters
        ---------
        image: signals.Image
            An image in 3D.
        isotropic_voxel:
            If True, generate a new image, scaling z in order to obtain
            isotropic voxel.
        """
        if len(image.axes_manager.shape) != 3:
            raise ValueError("image must have 3 dimension.")

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

        im_xy.axes_manager[0].index = (im_xy.axes_manager[0].high_index -
                                       im_xy.axes_manager[0].low_index)/2
        im_xy.axes_manager[1].index = (im_xy.axes_manager[1].high_index -
                                       im_xy.axes_manager[1].low_index)/2
        im_xy.axes_manager[2].index = (im_xy.axes_manager[2].high_index -
                                       im_xy.axes_manager[2].low_index)/2

        im_xz.axes_manager._update_attributes()
        im_yz.axes_manager._update_attributes()
        im_xy.plot()
        im_xz.plot()
        im_yz.plot()

    def plot(self,
             colorbar=True,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             auto_contrast=True,
             saturated_pixels=0.2,
             vmin=None,
             vmax=None,
             no_nans=False,
             **kwargs
             ):
        """Plot image.

        For multidimensional datasets an optional figure,
        the "navigator", with a cursor to navigate that data is
        raised. In any case it is possible to navigate the data using
        the sliders. Currently only signals with signal_dimension equal to
        0, 1 and 2 can be plotted.

        Parameters
        ----------
        navigator : {"auto", None, "slider", "spectrum", Signal}
            If "auto", if navigation_dimension > 0, a navigator is
            provided to explore the data.
            If navigation_dimension is 1 and the signal is an image
            the navigator is a spectrum obtained by integrating
            over the signal axes (the image).
            If navigation_dimension is 1 and the signal is a spectrum
            the navigator is an image obtained by stacking horizontally
            all the spectra in the dataset.
            If navigation_dimension is > 1, the navigator is an image
            obtained by integrating the data over the signal axes.
            Additionaly, if navigation_dimension > 2 a window
            with one slider per axis is raised to navigate the data.
            For example,
            if the dataset consists of 3 navigation axes X, Y, Z and one
            signal axis, E, the default navigator will be an image
            obtained by integrating the data over E at the current Z
            index and a window with sliders for the X, Y and Z axes
            will be raised. Notice that changing the Z-axis index
            changes the navigator in this case.
            If "slider" and the navigation dimension > 0 a window
            with one slider per axis is raised to navigate the data.
            If "spectrum" and navigation_dimension > 0 the navigator
            is always a spectrum obtained by integrating the data
            over all other axes.
            If None, no navigator will be provided.
            Alternatively a Signal instance can be provided. The signal
            dimension must be 1 (for a spectrum navigator) or 2 (for a
            image navigator) and navigation_shape must be 0 (for a static
            navigator) or navigation_shape + signal_shape must be equal
            to the navigator_shape of the current object (for a dynamic
            navigator).
            If the signal dtype is RGB or RGBA this parameters has no
            effect and is always "slider".
        axes_manager : {None, axes_manager}
            If None `axes_manager` is used.
        colorbar : bool, optional
             If true, a colorbar is plotted for non-RGB images.
        scalebar : bool, optional
            If True and the units and scale of the x and y axes are the same a
            scale bar is plotted.
        scalebar_color : str, optional
            A valid MPL color string; will be used as the scalebar color.
        axes_ticks : {None, bool}, optional
            If True, plot the axes ticks. If None axes_ticks are only
            plotted when the scale bar is not plotted. If False the axes ticks
            are never plotted.
        auto_contrast : bool, optional
            If True, the contrast is stretched for each image using the
            `saturated_pixels` value. Default True.
        saturated_pixels: scalar
            The percentage of pixels that are left out of the bounds.  For example,
            the low and high bounds of a value of 1 are the 0.5% and 99.5%
            percentiles. It must be in the [0, 100] range.
        vmin, vmax : scalar, optional
            `vmin` and `vmax` are used to normalize luminance data. If at least one of them is given
            `auto_contrast` is set to False and any missing values are calculated automatically.
        no_nans : bool, optional
            If True, set nans to zero for plotting.
        **kwargs, optional
            Additional key word arguments passed to matplotlib.imshow()

        """
        super(Image, self).plot(
            colorbar=colorbar,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            auto_contrast=auto_contrast,
            saturated_pixels=saturated_pixels,
            vmin=vmin,
            vmax=vmax,
            no_nans=no_nans,
            **kwargs
        )
