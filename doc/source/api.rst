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

_AbstractHMM
~~~~~~~~~~~~

.. autoclass:: hmmlearn.base._AbstractHMM
   :exclude-members: set_params, get_params, _get_param_names
   :private-members:
   :no-inherited-members:

BaseHMM
~~~~~~~

.. autoclass:: hmmlearn.base.BaseHMM
   :exclude-members: set_params, get_params, _get_param_names
   :private-members:
   :no-inherited-members:

VariationalBaseHMM
~~~~~~~~~~~~~~~~~~

.. autoclass:: hmmlearn.base.VariationalBaseHMM
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

CategoricalHMM
~~~~~~~~~~~~~~

.. autoclass:: hmmlearn.hmm.CategoricalHMM
   :exclude-members: set_params, get_params

PoissonHMM
~~~~~~~~~~

.. autoclass:: hmmlearn.hmm.PoissonHMM
   :exclude-members: set_params, get_params


hmmlearn.vhmm
-------------

VariationalCategoricalHMM
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hmmlearn.vhmm.VariationalCategoricalHMM
   :exclude-members: set_params, get_params

VariationalGaussianHMM
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: hmmlearn.vhmm.VariationalGaussianHMM
   :exclude-members: set_params, get_params
