##########################
ScenarioNet Documentation
##########################


Welcome to the ScenarioNet documentation!
ScenarioNet is an open-sourced platform for large-scale traffic scenario modeling and simulation with the following features:

* ScenarioNet defines a unified scenario description format containing HD maps and detailed object annotations.
* ScenarioNet provides tools to build and manage databases built from various data sources including real-world datasets like Waymo, nuScenes, Lyft L5, and nuPlan datasets and synthetic datasets like the procedural generated ones and safety-critical ones.
* Scenarios recorded in this format can be replayed in the digital twins with multiple views, ranging from Bird-Eye-View layout to realistic 3D rendering.

It can thus support several applications including large-scale scenario generation, AD testing, imitation learning, and reinforcement learning in both single-agent and multi-agent settings. The results imply scaling up the training data brings new research opportunities in machine learning and autonomous driving.

This documentation brings you the information on installation, usages and more of ScenarioNet!
You can also visit the `GitHub repo <https://github.com/metadriverse/scenarionet>`_ and `Webpage <https://metadriverse.github.io/scenarionet/>`_ for code and videos.
Please feel free to contact us if you have any suggestions or ideas!


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   install.rst
   example.rst
   operations.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Scenario Description

   description.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Supported Dataset

   datasets.rst
   nuplan.rst
   nuscenes.rst
   waymo.rst
   new_data.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Simulation



Citation
########

You can read `our white paper <https://arxiv.org/pdf/2306.12241.pdf>`_ describing the details of ScenarioNet! If you use ScenarioNet in your own work, please cite:

.. code-block:: latex

    @article{li2023scenarionet,
      title={ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling},
      author={Li, Quanyi and Peng, Zhenghao and Feng, Lan and Duan, Chenda and Mo, Wenjie and Zhou, Bolei and others},
      journal={arXiv preprint arXiv:2306.12241},
      year={2023}
    }

