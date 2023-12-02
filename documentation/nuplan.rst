#############################
nuPlan
#############################

| Website: https://www.nuplan.org/nuplan
| Download: https://www.nuplan.org/nuplan (Registration required)
| Paper: https://arxiv.org/pdf/2106.11810.pdf
| Documentation: https://nuplan-devkit.readthedocs.io/en/latest/

nuPlan is the world's first large-scale planning benchmark for autonomous driving.
It provides a large-scale dataset with 1200h of human driving data from 4 cities across the US and Asia with widely varying traffic patterns (Boston, Pittsburgh, Las Vegas and Singapore).
Our dataset is auto-labeled using a state-of-the-art Offline Perception system.
Contrary to existing datasets of this size, it not only contains the 3d boxes of the objects detected in the dataset,
but also provides 10% of the raw sensor data (120h).
We hope this large-scale sensor data can be used to make further progress in the field of end-to-end planning.

1. Install nuPlan Toolkit
==========================

First of all, we have to install the ``nuplan-devkit``.

.. code-block:: bash

    # 1. install from github (Recommend)
    git clone git@github.com:nutonomy/nuplan-devkit.git
    cd nuplan-devkit
    pip install -r requirements.txt
    pip install -e .

    # additional requirements
    pip install pytorch-lightning

    # 2. or install from PyPI
    pip install nuplan-devkit

By installing from github, you can access examples and source code the toolkit.
The examples are useful to verify whether the installation and dataset setup is correct or not.

2. Download nuPlan Data
===========================

The official data setup page is at https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html.
Despite this, we provide a simplified download instruction for convenient.
First of all, you need to register on the https://www.nuplan.org/nuplan and go to the Download section.
There are three types of data: Sensor, Map, Split.
We only use the last two kind of data, the sensor data is not required by ScenarioNet.
Thus please download the following files:

- nuPlan Maps
- nuPlan Mini(Train/Test/Val) Split

.. note::
    Please download the latest version (V1.1).


We recommend to download the mini split to test and make yourself familiar with the setup process.
All downloaded files are ``.zip`` files and can be uncompressed by ``unzip "*.zip"``.
All data should be placed to ``~/nuplan/dataset`` and the folder structure should comply `file hierarchy <https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html#filesystem-hierarchy>`_.

.. code-block:: text

    ~/nuplan
    ├── exp
    │   └── ${USER}
    │       ├── cache
    │       │   └── <cached_tokens>
    │       └── exp
    │           └── my_nuplan_experiment
    └── dataset
        ├── maps
        │   ├── nuplan-maps-v1.0.json
        │   ├── sg-one-north
        │   │   └── 9.17.1964
        │   │       └── map.gpkg
        │   ├── us-ma-boston
        │   │   └── 9.12.1817
        │   │       └── map.gpkg
        │   ├── us-nv-las-vegas-strip
        │   │   └── 9.15.1915
        │   │       └── map.gpkg
        │   └── us-pa-pittsburgh-hazelwood
        │       └── 9.17.1937
        │           └── map.gpkg
        └── nuplan-v1.1
            ├── splits
            │     ├── mini
            │     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
            │     │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
            │     │    ├── ...
            │     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
            │     └── train_boston
            │          ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
            │          ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
            │          ├── ...
            │          └── 2021.10.11.08.31.07_veh-50_01750_01948.db
            └── sensor_blobs
                  ├── 2021.05.12.22.00.38_veh-35_01008_01518
                  │    ├── CAM_F0
                  │    │     ├── c082c104b7ac5a71.jpg
                  │    │     ├── af380db4b4ca5d63.jpg
                  │    │     ├── ...
                  │    │     └── 2270fccfb44858b3.jpg
                  │    ├── CAM_B0
                  │    ├── CAM_L0
                  │    ├── CAM_L1
                  │    ├── CAM_L2
                  │    ├── CAM_R0
                  │    ├── CAM_R1
                  │    ├── CAM_R2
                  │    └──MergedPointCloud
                  │         ├── 03fafcf2c0865668.pcd
                  │         ├── 5aee37ce29665f1b.pcd
                  │         ├── ...
                  │         └── 5fe65ef6a97f5caf.pcd
                  │
                  ├── 2021.06.09.17.23.18_veh-38_00773_01140
                  ├── ...
                  └── 2021.10.11.08.31.07_veh-50_01750_01948


After downloading the data, you should add the following variables to ``~/.bashrc`` to make sure the ``nuplan-devkit`` can find the data::

    export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
    export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
    export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"

After this step, the examples in ``nuplan-devkit`` is supposed to work well.
Please try ``nuplan-devkit/tutorials/nuplan_scenario_visualization.ipynb`` and see if the demo code can successfully run.

3. Build nuPlan Database
============================

With all aforementioned steps finished, the nuPlan data can be stored in our internal format and composes a database.
Here we take converting raw data in ``nuplan-mini`` as an example::

    python -m scenarionet.convert_nuplan -d /path/to/your/database --raw_data_path ~/nuplan/dataset/nuplan-v1.1/splits/mini

The ``raw_data_path`` is the place to store ``.db`` files. Other arguments is available by using `-h` flag.
Now all converted scenarios will be placed at ``/path/to/your/database`` and are ready to be used in your work.

Known Issues: nuPlan
======================

N/A
