from functools import partial

from metadrive.scenario.scenario_description import ScenarioDescription as SD


class ScenarioFilter:
    GREATER = "greater"
    SMALLER = "smaller"

    @staticmethod
    def sdc_moving_dist(metadata, target_dist, condition=GREATER):
        """
        This function filters the scenario based on SDC information.
        """
        assert condition in [ScenarioFilter.GREATER, ScenarioFilter.SMALLER], "Wrong condition type"
        sdc_info = metadata[SD.SUMMARY.OBJECT_SUMMARY][metadata[SD.SDC_ID]]
        moving_dist = sdc_info[SD.SUMMARY.MOVING_DIST]
        if moving_dist > target_dist and condition == ScenarioFilter.GREATER:
            return True
        if moving_dist < target_dist and condition == ScenarioFilter.SMALLER:
            return True
        return False

    @staticmethod
    def object_number(metadata, number_threshold, object_type=None, condition=SMALLER):
        """
        Return True if the scenario satisfying the object number condition
        :param metadata: metadata in each scenario
        :param number_threshold: number of objects threshold
        :param object_type: MetaDriveType.VEHICLE or other object type. If none, calculate number for all object types
        :param condition: SMALLER or GREATER
        :return: boolean
        """
        assert condition in [ScenarioFilter.GREATER, ScenarioFilter.SMALLER], "Wrong condition type"
        if object_type is not None:
            num = metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS_EACH_TYPE].get(object_type, 0)
        else:
            num = metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS]
        if num > number_threshold and condition == ScenarioFilter.GREATER:
            return True
        if num < number_threshold and condition == ScenarioFilter.SMALLER:
            return True
        return False

    @staticmethod
    def has_traffic_light(metadata):
        return metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_TRAFFIC_LIGHTS] > 0

    @staticmethod
    def make(func, **kwargs):
        """
        A wrapper for partial() for filling some parameters
        :param func: func in this class
        :param kwargs: kwargs for filter
        :return: func taking only metadat as input
        """
        assert "metadata" not in kwargs, "You should only fill conditions, metadata will be fill automatically"
        if "condition" in kwargs:
            assert kwargs["condition"] in [ScenarioFilter.GREATER, ScenarioFilter.SMALLER], "Wrong condition type"
        return partial(func, **kwargs)
