#Zoom in notebook
import mpld3
%matplotlib inline
mpld3.enable_notebook()


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


import IPython.html.widgets as widgets
from IPython.html.widgets import interact, interactive, fixed


#function
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
    
    plt.plot(t, f(t), s, lw=3);
    plt.xlim(-xscale, +xscale);
    plt.title(title);


#threshold and binary
from scipy import ndimage
figure()
min_i, max_i=float(im.min()), float(im.max())
@interact(thr=(min_i, max_i),
          mor=('binary_opening','binary_closing'),
          shape={'No':[[1]],
                 'cross':[[0,1,0],[1,1,1],[0,1,0]],
                 'square':[[1,1,1],[1,1,1],[1,1,1]]})

def complex_plot(thr= (min_i+ max_i)/2.,mor='binary_opening',shape=[[1]]):

    f = getattr(ndimage,mor)

    plt.imshow(f((im > thr), shape),interpolation='nearest');
    

