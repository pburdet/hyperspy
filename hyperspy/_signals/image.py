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

from hyperspy.signal import Signal


class Image(Signal):
    """
    """
    _record_by = "image"
    
    def __init__(self, *args, **kw):
        super(Image,self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)
        
    def to_spectrum(self):
        """Returns the image as a spectrum.
        
        See Also
        --------
        as_spectrum : a method for the same purpose with more options.  
        signals.Image.to_spectrum : performs the inverse operation on images.

        """
        return self.as_spectrum(0+3j)
        
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

        if method=='bregman':
            img = self.apply(skimage.filter.denoise_tv_bregman,weight=weight,
                eps=eps, max_iter=n_iter_max)
        elif method=='chambolle':
            img = self.apply(skimage.filter.denoise_tv_chambolle,img.data,
                weight=weight, eps=eps, n_iter_max=n_iter_max)        
        return img
        
    def mean_filter(self,size):
        """ Apply a mean filter.

        Parameters
        ----------
        
        size : int | list or tuple
            `size` gives the shape that is taken from the input array, 
            at every element position, to define the input to the filter
            function.
        
        """
        import scipy.ndimage
        dim=self.axes_manager.shape
        if isinstance(size,int):
            kernel=np.ones([size]*len(dim))        
        else:
            kernel=np.ones(size)
        kernel=kernel/kernel.sum()       
        img = self.apply(scipy.ndimage.convolve,weights=kernel)    
        return img
        
        
        
        
        
        
        
        
