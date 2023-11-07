#############################
nuScenes
#############################

| Website: https://www.nuscenes.org/nuscenes
| Download: https://www.nuscenes.org/nuscenes (Registration required)
| Paper: https://arxiv.org/pdf/1903.11027.pdf

The nuScenes dataset (pronounced /nuːsiːnz/) is a public large-scale dataset for autonomous driving developed by the team at Motional (formerly nuTonomy).
Motional is making driverless vehicles a safe, reliable, and accessible reality.
By releasing a subset of our data to the public,
Motional aims to support public research into computer vision and autonomous driving.

For this purpose nuScenes contains 1000 driving scenes in Boston and Singapore,
two cities that are known for their dense traffic and highly challenging driving situations.
The scenes of 20 second length are manually selected to show a diverse and interesting set of driving maneuvers,
traffic situations and unexpected behaviors.
The rich complexity of nuScenes will encourage development of methods that enable safe driving in urban areas with dozens of objects per scene.
Gathering data on different continents further allows us to study the generalization of computer vision algorithms across different locations, weather conditions, vehicle types, vegetation, road markings and left versus right hand traffic.


1. Install nuScenes Toolkit
============================

First of all, we have to install the ``nuscenes-devkit``.

.. code-block:: bash

    # install from github (Recommend)
    git clone git@github.com:nutonomy/nuscenes-devkit.git
    cd nuscenes-devkit/setup
    pip install -e .

    # or install from PyPI
    pip install nuscenes-devkit

By installing from github, you can access examples and source code the toolkit.
The examples are useful to verify whether the installation and dataset setup is correct or not.


2. Download nuScenes Data
==============================

The official instruction is available at https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup.
Here we provide a simplified installation procedure.

First of all, please complete the registration on nuScenes website: https://www.nuscenes.org/nuscenes.
After this, go to the Download section and download the following files/expansions:

- mini/train/test splits
- Can bus expansion
- Map expansion

We recommend to download the mini split first to verify and get yourself familiar with the process.
All downloaded files are ``.tgz`` files and can be uncompressed by ``tar -zxf xyz.tgz``.

Secondly, all files should be organized to the following structure::

    /nuscenes/data/path/
    ├── maps/
    |   ├──basemap/
    |   ├──prediction/
    |   └──expansion/
    ├── can_bus/
    |   ├──scene-1110_meta.json
    |   └──...
    ├── samples/
    |   ├──CAM_BACK
    |   └──...
    ├── sweeps/
    |   ├──CAM_BACK
    |   └──...
    ├── v1.0-mini/
    |   ├──attribute.json
    |   ├──calibrated_sensor.json
    |   ├──map.json
    |   ├──log.json
    |   ├──ego_pose.json
    |   └──...
    └── v1.0-trainval/


The ``/nuscenes/data/path`` should be ``/data/sets/nuscenes`` by default according to the official instructions,
allowing the ``nuscens-devkit`` to find it.
But you can still place it to any other places and:

- build a soft link connect your data folder and ``/data/sets/nuscenes``
- or specify the ``dataroot`` when calling nuScenes APIs and our convertors.


After this step, the examples in ``nuscenes-devkit`` is supposed to work well.
Please try ``nuscenes-devkit/python-sdk/tutorials/nuscenes_tutorial.ipynb`` and see if the demo can successfully run.

3. Build nuScenes Database
===========================

After setup the raw data, convertors in ScenarioNet can read the raw data, convert scenario format and build the database.
Here we take converting raw data in ``nuscenes-mini`` as an example::

    python -m scenarionet.convert_nuscenes -d /path/to/your/database --split v1.0-mini --dataroot /nuscens/data/path

The ``split`` is to determine which split to convert. ``dataroot`` is set to ``/data/sets/nuscenes`` by default,
but you need to specify it if your data is stored in any other directory.
Now all converted scenarios will be placed at ``/path/to/your/database`` and are ready to be used in your work.


Known Issues: nuScenes
=======================

N/A
