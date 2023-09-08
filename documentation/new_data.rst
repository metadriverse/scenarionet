.. _new_data:

#############################
New dataset support
#############################

We believe it is the effort from community makes it closer to a `ImageNet` of autonomous driving field.
And thus we encourage the community to contribute convertors for new datasets.
To build a convertor compatible to our APIs, one should follow the following steps.
We recommend to take a look at our existing convertors.
They are good examples and can be adjusted to parse the new dataset.
Besides, **we are very happy to provide helps if you are working on supporting new datasets!**

**1. Convertor function input/output**

Take the ``convert_waymo(scenario:scenario_pb2.Scenario(), version:str)->metadrive.scenario.ScenarioDescription`` as an example.
It takes a scenario recorded in Waymo original format as example and returned a ``ScenarioDescription`` which is actually a nested Python ``dict``.
We just extend the functions of ``dict`` object to pre-define a structure with several required fields to fill out.

The required fields can be found at :ref:`desc`.
Apart from basic information like ``version`` and ``scenario_id``, there are mainly three fields that needs to fill:
``tracks``, ``map_features`` and ``dynamic_map_states``,
which stores objects information, map structure and traffic light states respectively.
These information can be extracted with the toolkit coupled with the original data.

**2. Fill in the object data**


By parsing the ``scenario`` with the official APIs, we can extract the history of all objects easily.
Generally, the object information is stored in a *frame-centric* way, which means the querying API takes the timestep as
input and returns all objects present in this frame.
However, ScenarioNet requires an *object-centric* object history.
In this way, we can easily know how many objects present in the scenario and retrieve the trajectory of each object with
its object_id.
Thus, a convert from *frame-centric* description to *object-centric* description is required.
A good example for this is the ``extract_traffic(scenario: NuPlanScenario, center)`` function in `nuplan/utils.py <https://github.com/metadriverse/scenarionet/blob/e6831ff972ed0cd57fdcb6a8a63650c12694479c/scenarionet/converter/nuplan/utils.py#L343>`_.
Similarly, the traffic light states can be extracted from the original data and represented in an *object-centric* way.

**3. Fill in the map data**

Map data consists of lane center lines and various kinds of boundaries such as yellow solid lines and white broken lines.
All of them are actually lines represented by a list of points.
To fill in this field, the first step is to query all line objects in the region where the scenario is collected.
By traversing the line list, and extracting the line type and point list, we can build a map which is ready to be
loaded into MetaDrive.

**4. Extend the description**

As long as all mandatory fields are filled, one can add new key-value pairs in each level of the nested-dict.
For example, the some scenarios are labeled with the type of scenario and the behavior of surrounding objects.
It is absolutely ok to include these information to the scenario descriptions and use them in the later experiments.


