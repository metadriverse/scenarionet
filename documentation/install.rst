.. _install:

########################
Installation
########################


1. Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~

The ScenarioNet repo contains tools for converting scenarios and building database from various data sources.
We recommend to create a new conda environment and install Python>=3.8,<=3.9::

    conda create -n scenarionet python==3.9
    conda activate scenarionet

2. Make a New Folder (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition, the operations in ScenarioNet are executed as Python module ``python -m``, and thus we have to make sure
the working directory contains NO folders named ``metadrive`` or ``scenarionet``.
Therefore, we strongly recommend creating a new folder under your routine working directory.
For example, supposing you prefer working at ``/home/lee``,
it would be greate to have a new folder ``mdsn`` created under this path.
And the ``git clone`` and package installation should happen in this new directory.
As a result, the directory tree should look like this::

    /home/lee/
    ├──mdsn
        ├──metadrive
        ├──scenarionet
    ├──...

In this way, you can freely run the dataset operations at any places other than ``/home/lee/mdsn``.
Now, let's move to this new directory for further installation with ``cd mdsn``.

.. note::
    This step is optional. One can still ``git clone`` and ``pip install`` the following two packages at any places.
    If any ``python -m scenarionet.[command]`` fails to run, please check if there is a folder called `metadrive`
    or `scenarionet` contained in the current directory. If so, please switch to a new directory to avoid this issue.

3. Install MetaDrive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation part is maintained in `MetaDrive <https://github.com/metadriverse/metadrive>`_ repo, and let's install MetaDrive first.
The installation of MetaDrive on different platforms is straightforward and easy!
We recommend to install in the following ways::

    # Method 1 (Recommend, latest version, source code exposed)
    git clone git@github.com:metadriverse/metadrive.git
    cd metadrive
    pip install -e.

    # Method 2 (Stable version, source code hidden)
    pip install "metadrive-simulator>=0.4.1.1"

To check whether MetaDrive is successfully installed, please run::

    python -m metadrive.examples.profile_metadrive

.. note:: Please do not run the above command at a directory that has a sub-folder called :code:`./metadrive`.

4. Install ScenarioNet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ScenarioNet, we only provide Github installation::

    git clone git@github.com:metadriverse/scenarionet.git
    cd scenarionet

Anyone of the following commands will automatically install basic requirements with additional requirements
for specific datasets::

    # Install basic requirement only
    pip install -e .

    # Install  Waymo official toolkit
    pip install -e .[waymo]

    # Install nuScenes development tookit
    pip install -e .[nuscenes]

    # Install nuPlan development tookit
    pip install -e .[nuplan]

    # Install all toolkit for all datasets
    pip install -e .[all]


.. note::
    If you don't wanna access the source code, you can install these two packages with
    ``pip install git+https://github.com/metadriverse/scenarionet.git``
    and ``pip install git+https://github.com/metadriverse/metadrive.git``.
    Though it is more straightforward, one has to install additional requirements, like development
    toolkits, manually.
