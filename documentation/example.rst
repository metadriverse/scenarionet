#######################
Example
#######################

In this example, we will show you how to convert a small batch of `Waymo <https://waymo.com/intl/en_us/open/>`_ scenarios into the internal Scenario Description.
After that, the scenarios will be loaded to simulator for closed-loop simulation.
First of all, please install `MetaDrive <https://github.com/metadriverse/metadrive>`_ and `ScenarioNet <https://github.com/metadriverse/scenarionet>`_ following these steps :ref:`installation`.

Setup Waymo toolkit
********************
For any dataset, this step is necessary after installing ScenarioNet,
as we need to use the official toolkits of the data provider to parse the original scenario description and convert to our internal scenario description.
For Waymo data, please install the toolkit via::

    pip install waymo-open-dataset-tf-2-11-0==1.5.0

.. note::
    This package is only supported on Linux platform.

Prepare Data
*************

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


Convert to Scenario Description
*******************************

Run the following command to extract scenarios in ``exp_waymo``::

