.. _install:

########################
Installation
########################

The ScenarioNet repo contains tools for converting scenarios and building database from various data sources.
The simulation part is maintained in `MetaDrive <https://github.com/metadriverse/metadrive>`_ repo, and let's install MetaDrive first.

**1. Install MetaDrive**

The installation of MetaDrive on different platforms is straightforward and easy!
We recommend to use the following command to install::

    # Install MetaDrive Simulator
    git clone git@github.com:metadriverse/metadrive.git
    cd metadrive
    pip install -e.

It can also be installed from PyPI by::

 pip install "metadrive-simulator>=0.4.1.1"

To check whether MetaDrive is successfully installed, please run::

    python -m metadrive.examples.profile_metadrive

.. note:: Please do not run the above command in the folder that has a sub-folder called :code:`./metadrive`.

**2. Install ScenarioNet**

For ScenarioNet, we only provide Github installation::

    # Install ScenarioNet
    git clone git@github.com:metadriverse/scenarionet.git
    cd scenarionet
    pip install -e .

