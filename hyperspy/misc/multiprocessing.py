
def pool(parallel, pool_type=None, ipython_timeout=1.):
    """
    Create a pool for multiprocessing

    Parameters
    ----------
    pool_type: 'ipython' or'mp'
        the type of pool
    ipython_timeout : float
        Timeout to be passed for ipython parallel Client.
    """
    if pool_type is None:
        from IPython.parallel import Client, error
        try:
            c = Client(profile='hyperspy', timeout=ipython_timeout)
            pool = c[:parallel]
            pool_type = 'ipython'
        except (error.TimeoutError, IOError):
            from multiprocessing import Pool
            pool_type = 'mp'
            pool = Pool(processes=parallel)
    elif pool_type == 'iypthon':
        from IPython.parallel import Client
        c = Client(profile='hyperspy', timeout=ipython_timeout)
        pool = c[:parallel]
        pool_type = 'ipython'
    else:
        from multiprocessing import Pool
        pool_type = 'mp'
        pool = Pool(processes=parallel)
    return pool, pool_type


def split(self, parallel, axis=0):
    dim_split = self.axes_manager.shape[axis]
    step_sizes = [dim_split / parallel] * parallel
    for i in range(dim_split % parallel):
        step_sizes[i] += 1
    self_to_split = self.split(axis=axis, step_sizes=step_sizes)
    return self_to_split


def multifit(args):
    from hyperspy.model import Model
    model_dict, kwargs = args
    m = Model(model_dict)
    m.multifit(**kwargs)
    d = m.as_dictionary()
    del d['spectrum']
    # delete everything else that doesn't matter. Only maps of
    # parameters and chisq matter
    return d


def isart(args):
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


def rotate(args):
    from scipy.ndimage import rotate
    d = rotate(**args)
    return d


def absorption_correction_matrix(args):
    from hyperspy.misc.eds import physical_model
    d = physical_model.absorption_correction_matrix(**args)
    return d


def absorption_correction_matrix2(args):
    from hyperspy.misc.eds import physical_model
    d = physical_model.absorption_correction_matrix2(**args)
    return d
