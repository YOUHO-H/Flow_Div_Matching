Improving Flow Matching by Aligning Flow Divergence
===================================================

Trajectory Sampling for Dynamical Systems
-----------------------------------------

Official implementation of the flow and divergence matching (FDM) for trajectory sampling.

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. **Suggested:** Set the package cache directory of ``uv`` to a directory in a mounted drive.
   For example,

   .. code:: bash

      echo "export UV_CACHE_DIR=/root/workspace/out/uv-cache" >> ~/.bashrc
      source ~/.bashrc

#. Install ``cuDNN``:

   .. code:: bash

      apt-get install -y cudnn9-cuda-12

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Test your installation by running the unit tests:

   .. code:: bash

      pytest tests

Supplementary Documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line inferface configuration library for configuring the experiments in this project.
* `Optuna <https://optuna.readthedocs.io/en/v4.2.0/index.html>`_: Library for tuning the XGBoost hyperparameters.
* `Optuna Dashboard <https://optuna-dashboard.readthedocs.io/en/stable/index.html>`_: Library for visualizing the results of Optuna's optimization.
* `PyTorch <https://pytorch.org/docs/2.5/index.html>`_: Library for implementing Zgraggen algorithm.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.5.0/>`_: Library for handles training the Zgraggen algorithm.

Training the models
===================

To see all the configurable options, run

.. code:: bash

   python src/userfm/main.py dataset=GaussianMixture model=ModelDiffusion -c job

Change the default ``out_dir`` of ``class Config`` in ``src/cs.py`` to where you want the trained models to be saved, or override it with

.. code:: bash

   python src/userfm/main.py out_dir=<path/to/save/trained/models> <other_overrides...>

To train a model, run

.. code:: bash

   python src/userfm/main.py +experiment=<experiment> dataset=<dataset> model=<model>

where:

* ``<experiment>`` is replaced with either:

   * ``TrainInitialTimeStepConditioned`` for the relative error task.
   * ``TrainUnconditioned`` for the unconditional and a posteriori conditional generation tasks.

* ``<dataset>`` is replaced with either ``GaussianMixture``, ``SimpleHarmonicOscillator``, ``Lorenz``, or ``FitzHughNagumo``.
* ``<model>`` is replaced with either ``DiffusionVE``, ``FlowMatchingOT``, or ``FlowMatchingVE``.

To train FDM, we add the conditional divergence matching (CDM) loss to the conditional flow matching (CFM) loss.
In the code, adding the CDM loss is called regularization.
Add the CDM loss by adding the override **surrounded by single-quotes**

.. code:: bash

   'model.regularizations=[<regularization>,...]'

where:

* ``<regularization>`` is either ``{_target_:cs.RegularizationDivergence,coefficient:2.}``
* ``<regularization>`` is either ``{_target_:cs.RegularizationDerivative,coefficient:2.}``

Note, if training a model for ``dataset=GaussianMixture``, consider setting `use_ckpt_monitor=false``.

Tuning with Optuna
------------------

To tune model hyperparameters with Optuna, run

.. code:: bash

   python src/userfm/main_optuna.py +experiment=<experiment> dataset=<dataset> model=<model> <additional_overrides>

where:

* ``<experiment>`` is replaced with either:

   * ``OptunaTrainInitialTimeStepConditioned`` for the relative error task.
   * ``OptunaTrainUnconditioned`` for the unguided and a posteriori guided generation tasks.

* ``<dataset>`` is replaced with either ``GaussianMixture``, ``SimpleHarmonicOscillator``, ``Lorenz``, or ``FitzHughNagumo``.
* ``<model>`` is replaced with either ``DiffusionVE``, ``FlowMatchingOT``, or ``FlowMatchingVE``.

Notice that ``Optuna`` that now prefixes the experiment name.
Also, note that ``main_optuna.py`` will **NOT** save any model checkpoints to avoid excess disk space usage.

Use `Optuna Dashboard <https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html>`_ to view the results of Optuna.

``main_optuna.py`` is currently set up to tune the coefficent of the CDM loss and the learning rate using this command:

.. code:: bash

   python src/userfm/main_optuna.py +experiment=OptunaTrainInitialTimeStepConditioned dataset=Lorenz model=FlowMatchingOT 'model.regularizations=[{_target_:cs.RegularizationDivergence,coefficient:0}]'

``main_optuna.py`` overrides the value of ``coefficient`` so it does not matter that it is set to ``0`` above.

Original Code from Finzi et al. (mostly)
----------------------------------------

.. warning::

   This has not been tested recently, so it may take some work to get running again.
   We have checked that training a diffusion model using this code produces a model that is comparable to the one produced by our code.

Lorenz
^^^^^^
Change ``--workdir`` as needed.

.. code:: bash

   python src/userdiffusion/main.py --config=src/userdiffusion/config.py --config.dataset=LorenzDataset --workdir=../../out/diffusion-dynamics/pmlr-v202-finzi23a/runs/lorenz/

Fitzhugh-Nagumo
^^^^^^^^^^^^^^^
Change ``--workdir`` as needed.

.. code:: bash

   python src/userdiffusion/main.py --config=src/userdiffusion/config.py --config.dataset=FitzHughDataset --workdir=../../out/diffusion-dynamics/pmlr-v202-finzi23a/runs/fitzhugh/

Pendulum
^^^^^^^^
Change ``--workdir`` as needed.

.. code:: bash

   python src/userdiffusion/main.py --config=src/userdiffusion/config.py --config.dataset=NPendulum --workdir=../../out/diffusion-dynamics/pmlr-v202-finzi23a/runs/pendulum/


Evaluating the models
=====================

Look at the Jupyter notebooks in ``notebooks`` to produce some of the plots in the paper.
In particular:

   * ``event_histogram``
   * ``event_histogram_saved``
   * ``nll``
