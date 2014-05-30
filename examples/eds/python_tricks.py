#useful Function

>>> np.repeat([1, 2],2,axis=0)
    [1, 1, 2, 2]

>>> [i+j for (i,j) in zip([1, 2, 3],[3, 4, 5])]
    [4, 6, 8]

>>> np.insert([1,2],1,0)
    [1, 0, 2]

>>> np.count_nonzero([1, 2, 0, 0])
    2

>>> plt.mlab.frange(0,1, 0.5)
    array([0., 0.5, 1.])

>>> map(sum,[[1, 2],[3, 4]])
    [3, 7]

>>> np.floor([0.5, 1.1]) #np.ceil
    array([0., 1.])

>>> 'abcababckd'.split('a')
    ['', 'bc', 'b', 'bckd']

>>> [i+j for i,j in enumerate([3,4,5])]
    [3, 5, 7]
    
#structured array
>>> a = np.array(([1, 2, 3],[3, 4, 5]),dtype=[('x1', object),('x2', object)])
>>> a['x1'].item() 
    [1, 2, 3]

>>> np.array([[1, 2],[3, 4]]).T
    array([[1, 3],[2, 4]])

>>> np.array([True, True, True]).all()
    True

>>> 13 % 4
    1

>>> a = [1,2]
>>> hasattr(a,'__iter__')
    True

>>> isinstance(3, int)
    True

>>> np.zeros_like([1, 2, 3])
    array([0, 0, 0])

>>> np.searchsorted([1,2,3],1.5)
    1

>>> set([0,1,0,1,4,1,1,2,3,5,6])
    {0, 1, 2, 3, 4, 5, 6}

.ravel()

get_backend()

Usefull magic function
----------------

%doctest_mode .... to write example

%edit misc.utils 
to edit code

release memory (Out)
%reset -f out 

all variable
%who_ls

copy paste part of code
%paste
