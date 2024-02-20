#####################
Example Colab
#####################


.. |colab_sim| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Open In Colab
   :target: https://colab.research.google.com/github/metadriverse/scenarionet/blob/main/tutorial/simulation.ipynb


**Colab example for running simulation with ScenarioNet:** |colab_sim|


.. |colab_read| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Open In Colab
   :target: https://colab.research.google.com/github/metadriverse/scenarionet/blob/main/tutorial/read_established_scenarionet_dataset.ipynb

**Colab example for reading established ScenarioNet dataset:** |colab_read|



#######################
Waymo Example
#######################

In this example, we will show you how to convert a small batch of `Waymo <https://waymo.com/intl/en_us/open/>`_ scenarios into the internal **Scenario Description**.
After that, the scenarios will be loaded to MetaDrive simulator for closed-loop simulation.
First of all, please install `MetaDrive <https://github.com/metadriverse/metadrive>`_ and `ScenarioNet <https://github.com/metadriverse/scenarionet>`_ following these steps :ref:`installation`.


1. Setup Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For any dataset, the first step after installing ScenarioNet is to install the corresponding official toolkit as we need to use it to parse the original data and convert to our internal scenario description.
For Waymo data, we already have the parser in ScenarioNet so just install the TensorFlow and Protobuf via::

    pip install tensorflow==2.11.0
    conda install protobuf==3.20

.. note::
    You may fail to install ``protobuf`` if using ``pip install protobuf==3.20``. If so, install via ``conda install protobuf=3.20``.

For other datasets like nuPlan and nuScenes, you need to setup `nuplan-devkit <https://github.com/motional/nuplan-devkit>`_ and `nuscenes-devkit <https://github.com/nutonomy/nuscenes-devkit>`_ respectively.
Guidance on how to setup these datasets and connect them with ScenarioNet can be found at :ref:`datasets`.

2. Prepare Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access the Waymo motion data at `Google Cloud <https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_2_0>`_.
Download one tfrecord scenario file from ``waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s``.
In this tutorial, we only use the first file ``training_20s.tfrecord-00000-of-01000``.
Just click the download button |:arrow_down:| on the right side to download it.
And place the downloaded tfrecord file to a folder. Let's call it ``exp_waymo`` and the structure is like this::

    exp_waymo
    ├──training_20s.tfrecord-00000-of-01000

.. note::
    For building database from all scenarios, install ``gsutil`` and use this command:
    ``gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s" .``
    Likewise, place all downloaded tfrecord files to the same folder.


3. Build Mini Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following command to extract scenarios in ``exp_waymo`` to ``exp_converted``::

    python -m scenarionet.convert_waymo -d /path/to/exp_converted/ --raw_data_path /path/to/exp_waymo --num_files=1

.. note::
    When running ``python -m``, make sure the directory you are at doesn't contain a folder called ``scenarionet``.
    Otherwise, the running may fail. For more details about the command, use ``python -m scenarionet.convert_waymo -h``

Now all extracted scenarios will be placed in ``exp_converted`` directory.
If we list the directory with ``ll`` command, the structure will be like::

    exp_converted
    ├──exp_converted_0
    ├──exp_converted_1
    ├──exp_converted_2
    ├──exp_converted_3
    ├──exp_converted_4
    ├──exp_converted_5
    ├──exp_converted_6
    ├──exp_converted_7
    ├──dataset_mapping.pkl
    ├──dataset_summary.pkl

This is because we use 8 workers to extract the scenarios, and thus the converted scenarios will be stored in 8 subfolders.
If we go check ``exp_converted_0``, we will see the structure is like::

    ├──sd_waymo_v1.2_2085c5cffcd4727b.pkl
    ├──sd_waymo_v1.2_27997d88023ff2a2.pkl
    ├──sd_waymo_v1.2_3ece8d267ce5847c.pkl
    ├──sd_waymo_v1.2_53e9adfdac0eb822.pkl
    ├──sd_waymo_v1.2_8e40ffb80dd2f541.pkl
    ├──sd_waymo_v1.2_df72c5dc77a73ed6.pkl
    ├──sd_waymo_v1.2_f1f6068fabe77dc8.pkl
    ├──dataset_mapping.pkl
    ├──dataset_summary.pkl

Therefore, the subfolder produced by each worker is actually where the converted scenarios are placed.
To aggregate the scenarios produced by all workers, the ``exp_converted/dataset_mapping.pkl`` stores the mapping
from `scenario_id` to the path of the target scenario file relative to ``exp_converted``.
As a result, we can get all scenarios produced by 8 workers by loading the database `exp_converted`.

4. Database Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several basic operations are available and allow us to split, merge, move, and check the databases.
First of all, let's check how many scenarios are included in this database built from ``training_20s.tfrecord-00000-of-01000``::

    python -m scenarionet.num -d /path/to/exp_converted/

It will show that there are totally 61 scenarios.
For machine learning applications, we usually want to split training/test sets.
To this end, we can use the following command to build the training set::

    python -m scenarionet.split --from /path/to/exp_converted/ --to /path/to/exp_train --num_scenarios 40

Again, use the following commands to build the test set::

    python -m scenarionet.split --from /path/toexp_converted/ --to /path/to/exp_test --num_scenarios 21 --start_index 40

We add the ``start_index`` argument to select the last 21 scenarios as the test set.
To ensure that no overlap exists, we can run this command::

    python -m scenarionet.check_overlap --d_1 /path/to/exp_train/ --d_2 /path/to/exp_test/

It will report `No overlapping in two database!`.
Now, let's suppose that the ``/exp_train/`` and ``/exp_test/`` are two databases built
from different source and we want to merge them into a larger one.
This can be achieved by::

    python -m scenarionet.merge --from /path/to/exp_train/ /path/to/exp_test -d /path/to/exp_merged

Let's check if the merged database is the same as the original one::

    python -m scenarionet.check_overlap --d_1 /path/to/exp_merged/ --d_2 /path/to/exp_converted

It will show there are 61 overlapped scenarios.
Congratulations! Now you are already familiar with some common operations.
More operations and details is available at :ref:`operations`.

5. Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The database can be loaded to MetaDrive simulator for scenario replay or closed-loop simulation.
First of all, let's replay scenarios in the ``exp_converted`` database::

    python -m scenarionet.sim -d /path/to/exp_converted --render 2D


By adding ``--render 3D`` flag, we can use 3D renderer::

    python -m scenarionet.sim -d /path/to/exp_converted --render 3D

.. note::
    ``--render advanced`` enables the advanced deferred rendering pipeline,
    but an advanced GPU better than RTX 2060 is required.