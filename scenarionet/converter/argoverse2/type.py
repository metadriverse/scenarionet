import logging

from metadrive.type import MetaDriveType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from av2.datasets.motion_forecasting.data_schema import ObjectType
    from av2.map.lane_segment import LaneType, LaneMarkType
except ImportError as e:
    logger.warning("Can not import av2-devkit: {}".format(e))


def get_traffic_obj_type(av2_obj_type):
    if av2_obj_type == ObjectType.VEHICLE or av2_obj_type == ObjectType.BUS:
        return MetaDriveType.VEHICLE
    # elif av2_obj_type == ObjectType.MOTORCYCLIST:
    #     return MetaDriveType.MOTORCYCLIST
    elif av2_obj_type == ObjectType.PEDESTRIAN:
        return MetaDriveType.PEDESTRIAN
    elif av2_obj_type == ObjectType.CYCLIST:
        return MetaDriveType.CYCLIST
    # elif av2_obj_type == ObjectType.BUS:
    #     return MetaDriveType.BUS
    # elif av2_obj_type == ObjectType.STATIC:
    #     return MetaDriveType.STATIC
    # elif av2_obj_type == ObjectType.CONSTRUCTION:
    #     return MetaDriveType.CONSTRUCTION
    # elif av2_obj_type == ObjectType.BACKGROUND:
    #     return MetaDriveType.BACKGROUND
    # elif av2_obj_type == ObjectType.RIDERLESS_BICYCLE:
    #     return MetaDriveType.RIDERLESS_BICYCLE
    # elif av2_obj_type == ObjectType.UNKNOWN:
    #     return MetaDriveType.UNKNOWN
    else:
        return MetaDriveType.OTHER


def get_lane_type(av2_lane_type):
    if av2_lane_type == LaneType.VEHICLE or av2_lane_type == LaneType.BUS:
        return MetaDriveType.LANE_SURFACE_STREET
    elif av2_lane_type == LaneType.BIKE:
        return MetaDriveType.LANE_BIKE_LANE
    else:
        raise ValueError("Unknown nuplan lane type: {}".format(av2_lane_type))


def get_lane_mark_type(av2_mark_type):
    conversion_dict = {
        LaneMarkType.DOUBLE_SOLID_YELLOW: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        LaneMarkType.DOUBLE_SOLID_WHITE: "ROAD_LINE_SOLID_DOUBLE_WHITE",
        LaneMarkType.SOLID_YELLOW: "ROAD_LINE_SOLID_SINGLE_YELLOW",
        LaneMarkType.SOLID_WHITE: "ROAD_LINE_SOLID_SINGLE_WHITE",
        LaneMarkType.DASHED_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        LaneMarkType.DASHED_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        LaneMarkType.DASH_SOLID_YELLOW: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        LaneMarkType.DASH_SOLID_WHITE: "ROAD_LINE_SOLID_DOUBLE_WHITE",
        LaneMarkType.DOUBLE_DASH_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        LaneMarkType.DOUBLE_DASH_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        LaneMarkType.SOLID_DASH_WHITE: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        LaneMarkType.SOLID_DASH_YELLOW: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        LaneMarkType.SOLID_BLUE: "UNKNOWN_LINE",
        LaneMarkType.NONE: "UNKNOWN_LINE",
        LaneMarkType.UNKNOWN: "UNKNOWN_LINE"
    }

    return conversion_dict.get(av2_mark_type, "UNKNOWN_LINE")
