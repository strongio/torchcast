torchcast
=====================================

``torchcast`` is a Python package for forecasting with state-space models built on top of |PyTorch|. Its focus is training and forecasting with *batches* of time-series, rather than training separate models for one time-series at a time. In addition, it provides robust support for *multivariate* time-series, where multiple correlated measures are being forecasted.

Currently the focus of ``torchcast`` is building models that are hybrids: they are classic state-space models with the twist that every part of these models is differentiable and can take advantage of PyTorch's flexibility. For :doc:`example <examples/electricity>`, we can use arbitrary PyTorch :class:`torch.nn.Modules` to predict the variance of forecasts or even to generate the underlying states themselves.

This repository is the work of `Strong Analytics <https://www.strong.io/>`_.

.. toctree::
   :maxdepth: 1

   quick_start
   examples/examples
   api/api

.. include:: macros.hrst

.. image:: examples_air_quality_6_2.png

Getting Started
---------------

``torchcast`` can be installed with `pip`:

.. code-block:: bash

    pip install git+https://github.com/strongio/torchcast.git#egg=torchcast

``torchcast`` requires Python >= 3.6 and |PyTorch| >= 1.8.

See the :doc:`quick_start` for a simple example that will get you up to speed, or delve into the :doc:`examples/examples` or :doc:`api/api`.
