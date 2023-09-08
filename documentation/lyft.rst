.. _lyft:

####################
Lyft (In Upgrading)
####################

| Website: https://woven.toyota/en/prediction-dataset
| Download: https://woven-planet.github.io/l5kit/dataset.html
| Paper: https://proceedings.mlr.press/v155/houston21a/houston21a.pdf

This dataset includes the logs of movement of cars, cyclists, pedestrians,
and other traffic agents encountered by our automated fleet.
These logs come from processing raw lidar, camera, and radar data through our team’s perception systems and are ideal
for training motion prediction models.
The dataset includes:

- 1000+ hours of traffic agent movement
- 16k miles of data from 23 vehicles
- 15k semantic map annotations

.. note::
    Currently, the old Lyft dataset can be read by ``nuscenes-toolkit`` and thus can share the nuScenes convertor.
    The new Lyft data is now maintained by Woven Planet and we are working on support the ``L5Kit`` for allowing
    using new Lyft data.


Known Issues: Lyft
===================

N/A

