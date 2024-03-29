import logging

import tqdm

from scenarionet.converter.utils import mph_to_kmh
import geopandas as gpd
from shapely.ops import unary_union
logger = logging.getLogger(__name__)
import numpy as np

from pathlib import Path
from tqdm import tqdm
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

from scenarionet.converter.argoverse2.type import get_traffic_obj_type, get_lane_type, get_lane_mark_type
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from typing import Final
from shapely.geometry import Point, Polygon

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_ESTIMATED_PEDESTRIAN_LENGTH_M: Final[float] = 0.5
_ESTIMATED_PEDESTRIAN_WIDTH_M: Final[float] = 0.5
_ESTIMATED_BUS_LENGTH_M: Final[float] = 12.0
_ESTIMATED_BUS_WIDTH_M: Final[float] = 2.5

_HIGHWAY_SPEED_LIMIT_MPH: Final[float] = 85.0


def extract_tracks(tracks, sdc_idx, track_length):
    ret = dict()

    def _object_state_template(object_id):
        return dict(type=None, state=dict(# Never add extra dim if the value is scalar.
            position=np.zeros([track_length, 3], dtype=np.float32), length=np.zeros([track_length], dtype=np.float32),
            width=np.zeros([track_length], dtype=np.float32), height=np.zeros([track_length], dtype=np.float32),
            heading=np.zeros([track_length], dtype=np.float32), velocity=np.zeros([track_length, 2], dtype=np.float32),
            valid=np.zeros([track_length], dtype=bool), ),
            metadata=dict(track_length=track_length, type=None, object_id=object_id, dataset="av2"))

    track_category = []

    for obj in tracks:
        object_id = obj.track_id
        track_category.append(obj.category.value)
        obj_state = _object_state_template(object_id)
        # Transform it to Waymo type string
        obj_state["type"] = get_traffic_obj_type(obj.object_type)
        if obj_state["type"] == MetaDriveType.VEHICLE:
            length = _ESTIMATED_VEHICLE_LENGTH_M
            width = _ESTIMATED_VEHICLE_WIDTH_M
        elif obj_state["type"] == MetaDriveType.PEDESTRIAN:
            length = _ESTIMATED_PEDESTRIAN_LENGTH_M
            width = _ESTIMATED_PEDESTRIAN_WIDTH_M
        elif obj_state["type"] == MetaDriveType.CYCLIST:
            length = _ESTIMATED_CYCLIST_LENGTH_M
            width = _ESTIMATED_CYCLIST_WIDTH_M
        # elif obj_state["type"] == MetaDriveType.BUS:
        #     length = _ESTIMATED_BUS_LENGTH_M
        #     width = _ESTIMATED_BUS_WIDTH_M
        else:
            length = 1
            width = 1

        for _, state in enumerate(obj.object_states):
            step_count = state.timestep
            obj_state["state"]["position"][step_count][0] = state.position[0]
            obj_state["state"]["position"][step_count][1] = state.position[1]
            obj_state["state"]["position"][step_count][2] = 0

            # l = [state.length for state in obj.states]
            # w = [state.width for state in obj.states]
            # h = [state.height for state in obj.states]
            # obj_state["state"]["size"] = np.stack([l, w, h], 1).astype("float32")
            obj_state["state"]["length"][step_count] = length
            obj_state["state"]["width"][step_count] = width
            obj_state["state"]["height"][step_count] = 1

            # heading = [state.heading for state in obj.states]
            obj_state["state"]["heading"][step_count] = state.heading

            obj_state["state"]["velocity"][step_count][0] = state.velocity[0]
            obj_state["state"]["velocity"][step_count][1] = state.velocity[1]

            obj_state["state"]["valid"][step_count] = True

        obj_state["metadata"]["type"] = obj_state["type"]

        ret[object_id] = obj_state

    return ret, track_category


def extract_lane_mark(lane_mark):
    line = dict()
    line["type"] = get_lane_mark_type(lane_mark.mark_type)
    line["polyline"] = lane_mark.polyline.astype(np.float32)
    return line


def extract_map_features(map_features):
    # with open(
    #         "/Users/fenglan/Desktop/vita-group/code/mdsn/scenarionet/data_sample/waymo_converted_0/sd_waymo_v1.2_7e8422433c66cc13.pkl",
    #         'rb') as f:
    #     waymo_sample = pickle.load(f)
    ret = {}
    vector_lane_segments = map_features.get_scenario_lane_segments()
    vector_drivable_areas = map_features.get_scenario_vector_drivable_areas()
    ped_crossings = map_features.get_scenario_ped_crossings()

    ids = map_features.get_scenario_lane_segment_ids()

    max_id = max(ids)
    for seg in vector_lane_segments:
        center = {}
        lane_id = str(seg.id)

        left_id = str(seg.id + max_id + 1)
        right_id = str(seg.id + max_id + 2)
        left_marking = extract_lane_mark(seg.left_lane_marking)
        right_marking = extract_lane_mark(seg.right_lane_marking)

        ret[left_id] = left_marking
        ret[right_id] = right_marking

        center["speed_limit_mph"] = _HIGHWAY_SPEED_LIMIT_MPH

        center["speed_limit_kmh"] = mph_to_kmh(_HIGHWAY_SPEED_LIMIT_MPH)

        center["type"] = get_lane_type(seg.lane_type)

        polyline = map_features.get_lane_segment_centerline(seg.id)
        center["polyline"] = polyline.astype(np.float32)

        center["interpolating"] = True

        center["entry_lanes"] = [str(id) for id in seg.predecessors]

        center["exit_lanes"] = [str(id) for id in seg.successors]

        center["left_boundaries"] = []

        center["right_boundaries"] = []

        center["left_neighbor"] = []

        center["right_neighbor"] = []
        center['width'] = np.zeros([len(polyline), 2], dtype=np.float32)

        ret[lane_id] = center

    polygons = []
    for polygon in vector_drivable_areas:
        # convert to shapely polygon
        points = polygon.area_boundary
        polygons.append(Polygon([(p.x, p.y) for p in points]))

    polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
    boundaries = gpd.GeoSeries(unary_union(polygons)).boundary.explode(index_parts=True)
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        for i in range(0, len(block_points), 20):
            id = f'boundary_{idx}{i}'
            ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points[i:i + 20]}

    for cross in ped_crossings:
        bound = dict()
        bound["type"] = MetaDriveType.CROSSWALK
        bound["polygon"] = cross.polygon.astype(np.float32)
        ret[str(cross.id)] = bound

    return ret


def get_av2_scenarios(av2_data_directory, start_index, num):
    # parse raw data from input path to output path,
    # there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    logger.info("\nReading raw data")

    all_scenario_files = sorted(Path(av2_data_directory).rglob("*.parquet"))

    return all_scenario_files


def convert_av2_scenario(scenario, version):
    md_scenario = SD()

    md_scenario[SD.ID] = scenario.scenario_id
    md_scenario[SD.VERSION] = version

    # Please note that SDC track index is not identical to sdc_id.
    # sdc_id is a unique indicator to a track, while sdc_track_index is only the index of the sdc track
    # in the tracks datastructure.

    track_length = scenario.timestamps_ns.shape[0]

    tracks, category = extract_tracks(scenario.tracks, scenario.focal_track_id, track_length)

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario.static_map)
    md_scenario[SD.MAP_FEATURES] = map_features

    # compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = 'AV'
    md_scenario[SD.METADATA]["dataset"] = "av2"
    md_scenario[SD.METADATA]["scenario_id"] = scenario.scenario_id
    md_scenario[SD.METADATA]["source_file"] = scenario.scenario_id
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 49

    # obj id
    obj_keys = list(tracks.keys())
    md_scenario[SD.METADATA]["objects_of_interest"] = [obj_keys[idx] for idx, cat in enumerate(category) if cat == 2]

    md_scenario[SD.METADATA]["sdc_track_index"] = obj_keys.index('AV')

    track_index = [obj_keys.index(scenario.focal_track_id)]
    track_id = [scenario.focal_track_id]
    track_difficulty = [0]
    track_obj_type = [tracks[id]["type"] for id in track_id]
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": track_difficulty[count],
            "object_type": track_obj_type[count]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return md_scenario


def preprocess_av2_scenarios(files, worker_index):
    """
    Convert the waymo files into scenario_pb2. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenario_pb2
    """

    for scenario_path in tqdm(files, desc="Process av2 scenarios for worker {}".format(worker_index)):
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = (scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json")
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        scenario.static_map = static_map
        yield scenario

    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))  # return scenarios
