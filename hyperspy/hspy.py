"""

All public packages, functions and classes are in this package. This package is
automatically imported in the user namespace when starting HyperSpy using the
starting script e.g. by typing ``hyperspy`` in a console, using the context
menu entries or using the links in the ``Start Menu``, the
:mod:`~hyperspy.hspy` package is imported in the user namespace. When using
HyperSpy as a library, it is reccommended to import the :mod:`~hyperspy.hspy`
package as follows:

    from hyperspy import hspy as hs

Functions:

    create_model
        Create a model for curve fitting.

    get_configuration_directory_path
        Return the configuration directory path.

    load
        Load data into Signal instaces from supported files.

    preferences
        Preferences class instance to configure the default value of different
        parameters. It has a CLI and a GUI that can be started by execting its
        `gui` method i.e. `preferences.gui()`.


The :mod:`~hyperspy.hspy` package contains the following subpackages:

    :mod:`~hyperspy.hspy.signals`
        Specialized Signal instances.

    :mod:`~hyperspy.hspy.utils`
        Functions that operate of Signal instances and other goodies.

    :mod:`~hyperspy.hspy.components`
        Components that can be used to create a model for curve fitting.

For more details see their doctrings.

"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'RdYlBu_r'

from hyperspy.Release import version as __version__
from hyperspy import components
from hyperspy import signals
from hyperspy.io import load
from hyperspy.defaults_parser import preferences
from hyperspy import utils

# dev
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.eds import database
from hyperspy.misc.eds import image_eds
from hyperspy.misc.eds import epq_database
#from hyperspy.misc.eds.MAC import MAC_db as MAC
#from hyperspy.misc.eds.kfactors_osiris import kfactors_db as kfactors


def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    return hyperspy.misc.config_dir.config_path


def create_model(signal, *args, **kwargs):
    """Create a model object

    Any extra argument is passes to the Model constructor.

    Parameters
    ----------
    signal: A signal class

    If the signal is an EELS signal the following extra parameters
    are available:

    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically
        a powerlaw to the model and estimate the parameters by the
        two-area method.
    auto_add_edges : boolean
        If True, and if spectrum is an EELS instance, it will
        automatically add the ionization edges as defined in the
        Spectrum instance. Adding a new element to the spectrum using
        the components.EELSSpectrum.add_elements method automatically
        add the corresponding ionisation edges to the model.
    ll : {None, EELSSpectrum}
        If an EELSSPectrum is provided, it will be assumed that it is
        a low-loss EELS spectrum, and it will be used to simulate the
        effect of multiple scattering by convolving it with the EELS
        spectrum.
    GOS : {'hydrogenic', 'Hartree-Slater', None}
        The GOS to use when auto adding core-loss EELS edges.
        If None it will use the Hartree-Slater GOS if
        they are available, otherwise it will use the hydrogenic GOS.

    If the signal is an EDS signal the following extra parameters
    are available:

    auto_add_lines : boolean
        If True, and if spectrum is an EDS instance, it will
        automatically add Gaussians for all X-rays generated in the energy
        range by an element.
    auto_background : boolean
        If True, and if spectrum is an EDS instance adds automatically
        a polynomial order 6 to the model.

    Examples
    --------

    With s an EDS instance
    >>> m = create_model(s)
    >>> m.mulifit()
    >>> m.plot()
    >>> m.get_lines_intensity(plot_result=True)

    Returns
    -------

    A Model class

    """

    from hyperspy._signals.eels import EELSSpectrum
    from hyperspy._signals.eds_sem import EDSSEMSpectrum
    from hyperspy._signals.eds_tem import EDSTEMSpectrum
    from hyperspy.models.eelsmodel import EELSModel
    from hyperspy.models.edssemmodel import EDSSEMModel
    from hyperspy.models.edstemmodel import EDSTEMModel
    from hyperspy.model import Model
    if isinstance(signal, EELSSpectrum):
        return EELSModel(signal, *args, **kwargs)
    elif isinstance(signal, EDSSEMSpectrum):
        return EDSSEMModel(signal, *args, **kwargs)
    elif isinstance(signal, EDSTEMSpectrum):
        return EDSTEMModel(signal, *args, **kwargs)
    else:
        return Model(signal, *args, **kwargs)

