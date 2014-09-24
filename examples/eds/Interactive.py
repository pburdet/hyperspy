#Zoom in notebook
import mpld3
%matplotlib inline
mpld3.enable_notebook()


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


import IPython.html.widgets as widgets
from IPython.html.widgets import interact, interactive, fixed


# function
@interact(fun=('sin', 'cos', 'arctan'),  # dropdown menu
          title='my function',  # text area
          dashed=False,  # checkbox
          xscale=(.1, 100.))  # float slider
def complex_plot(fun='sin',
                 title='sine',
                 dashed=False,
                 xscale=5.):

    f = getattr(np, fun)
    t = np.linspace(-xscale, +xscale, 1000)
    s = '--' if dashed else '-'

    plt.plot(t, f(t), s, lw=3)
    plt.xlim(-xscale, +xscale)
    plt.title(title)


#threshold and binary
import IPython.html.widgets as widgets
from scipy import ndimage
from IPython.html.widgets import interact, interactive, fixed
im2 = database.image2D()
im = im2.deepcopy()
im = ndimage.gaussian_filter(im.data, 2)
im2 = im2.data
min_i, max_i = float(im.min()), float(im.max())


figure()


@interact(thr=(0., 1., 0.0001))
def thr_plot(thr=0.5):
    data = (im > (max_i - min_i) * thr + min_i)
    subplot(211)
    plt.imshow(data, interpolation='nearest')
    subplot(212)
    plt.imshow(data * im2, interpolation='nearest', vmin=min_i, vmax=max_i)

figure()


@interact(thr=(0., 1., 0.0001),
          mor=('binary_opening', 'binary_closing'),
          shape={'No': [[1]],
                 'cross': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                 'square': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]})
def complex_plot(thr=0.5, mor='binary_opening', shape=[[1]]):

    f = getattr(ndimage, mor)
    data = f((im > (max_i - min_i) * thr + min_i), shape)
    subplot(211)
    plt.imshow(data,
               interpolation='nearest')
    subplot(212)
    plt.imshow(data * im2,
               interpolation='nearest', vmin=min_i, vmax=max_i)
