API Reference
=============

This is the class and function reference of ``hmmlearn``.

Please refer to the :ref:`full user guide <user_guide>` for further details, as
the class and function raw specifications may not be enough to give full
guidelines on their uses.


hmmlearn.base
-------------

ConvergenceMonitor
~~~~~~~~~~~~~~~~~~

.. autoclass:: hmmlearn.base.ConvergenceMonitor

_BaseHMM
~~~~~~~~

.. autoclass:: hmmlearn.base._BaseHMM
   :exclude-members: set_params, get_params, _get_param_names
   :private-members:
   :no-inherited-members:

hmmlearn.hmm
------------

GaussianHMM
~~~~~~~~~~~

.. autoclass:: hmmlearn.hmm.GaussianHMM
   :exclude-members: covars_, set_params, get_params

GMMHMM
~~~~~~

.. autoclass:: hmmlearn.hmm.GMMHMM
   :exclude-members: set_params, get_params

MultinomialHMM
~~~~~~~~~~~~~~

.. autoclass:: hmmlearn.hmm.MultinomialHMM
   :exclude-members: set_params, get_params
