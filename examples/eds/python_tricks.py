
# useful Function

>> > np.repeat([1, 2], 2, axis=0)
    [1, 1, 2, 2]

>> > [i + j for (i, j) in zip([1, 2, 3], [3, 4, 5])]
    [4, 6, 8]

>> > np.insert([1, 2], 1, 0)
    [1, 0, 2]

>> > np.count_nonzero([1, 2, 0, 0])
    2

>> > plt.mlab.frange(0, 1, 0.5)
    array([0., 0.5, 1.])

>> > map(sum, [[1, 2], [3, 4]])
    [3, 7]

>> > np.floor([0.5, 1.1])  # np.ceil
    array([0., 1.])

>> > 'abcababckd'.split('a')
    ['', 'bc', 'b', 'bckd']

>> > [i + j for i, j in enumerate([3, 4, 5])]
    [3, 5, 7]

# structured array
>> > a = np.array(([1, 2, 3], [3, 4, 5]), dtype=[('x1', object), ('x2', object)])
>> > a['x1'].item()
    [1, 2, 3]

>> > np.array([[1, 2], [3, 4]]).T
    array([[1, 3], [2, 4]])

>> > np.array([True, True, True]).all()
    True

>> > 13 % 4
    1

>> > a = [1, 2]
>> > hasattr(a, '__iter__')
    True

>> > isinstance(3, int)
    True

>> > np.zeros_like([1, 2, 3])
    array([0, 0, 0])

>> > np.searchsorted([1, 2, 3], 1.5)
    1

>> > set([0, 1, 0, 1, 4, 1, 1, 2, 3, 5, 6])
    {0, 1, 2, 3, 4, 5, 6}

>> > a = np.array([0., 1.])
>> > np.place(a, (a == 0.), 1.)
>> > a
    array([1., 1.])

>> > a = np.array(['po', 'apoa', 'piop'])
>> > filter(lambda x: 'po' in x, a)
['po', 'apoa']

>> > a = np.array([0., 1., 3.])
>> > a.argmax()
2

>> > a = np.array([0., 1., 3.])
>> > a[a>2] = 10.
np.array([10., 10., 3.])

#map a function with some fix adn some variant argument
>>> import functools
>>> map(functools.partial(f,x=1,y=2),a)

#save 2D data as csv
np.savetxt('a.csv',s.data)

.ravel()

get_backend()

Usefull magic function
----------------

%matplotlib inline

%matplotlib qt

%doctest_mode .... to write example

%edit misc.utils
to edit code

release memory(Out)
%reset - f out

all variable
%who_ls

copy paste part of code
%paste

%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\basic.py"
%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\PCA.py"
%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\quantification.py"
%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\simulation.py"
%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\visualisation.py"
%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\quantification_tem.py"
%load "C:\Users\pb565\Documents\Python\hyperspy\examples\eds\edsmodel.py"

# To configure lprune
# http://pynash.org/2013/03/06/timing-and-profiling.html

import time
a = time.time()
1 + 2
b = time.time()
print b - a


%time or % %time

%timeit

%prun

from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy.misc.utils import slugify
from hyperspy.misc.material import mass_absorption_coefficient
%lprun - f  mass_absorption_coefficient \
    - f EDSSEMSpectrum.compute_continuous_xray_generation \
    - f slugify \
    EDSSEMSpectrum.compute_continuous_xray_absorption(s)
    
%%file mp.py
if __name__ == "__main__":
    print "Hello"

%run mp.py

%%cmd
python mp.py
