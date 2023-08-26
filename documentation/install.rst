.. _install:

########################
Installation
########################

The ScenarioNet repo contains tools for converting scenarios and building database from various data sources.
We recommend to create a new conda environment and install Python>=3.8,<=3.9::

    conda create -n scenarionet python==3.9
    conda activate scenarionet

The simulation part is maintained in `MetaDrive <https://github.com/metadriverse/metadrive>`_ repo, and let's install MetaDrive first.

**1. Install MetaDrive**

The installation of MetaDrive on different platforms is straightforward and easy!
We recommend to install from Github in the following two ways::

    # Method 1 (Recommend, if you don't want to access the source code)
    pip install git+https://github.com/metadriverse/metadrive.git

    # Method 2
    git clone git@github.com:metadriverse/metadrive.git
    cd metadrive
    pip install -e.


A more stable version of MetaDrive can be installed from PyPI by::

    # More stable version than installing from Github
    pip install "metadrive-simulator>=0.4.1.1"

To check whether MetaDrive is successfully installed, please run::

    python -m metadrive.examples.profile_metadrive

.. note:: Please do not run the above command in the folder that has a sub-folder called :code:`./metadrive`.

**2. Install ScenarioNet**

For ScenarioNet, we only provide Github installation. Likewise, there are two ways to install from Github::

    # Method 1 (Recommend, if you don't want to access the source code)
    pip install git+https://github.com/metadriverse/scenarionet.git

    # Method 2
    git clone git@github.com:metadriverse/scenarionet.git
    cd scenarionet
    pip install -e .

