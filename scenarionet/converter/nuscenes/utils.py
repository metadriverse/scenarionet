import copy
import logging

import geopandas as gpd
import numpy as np
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from shapely.ops import unary_union

from scenarionet.converter.nuscenes.type import ALL_TYPE, HUMAN_TYPE, BICYCLE_TYPE, VEHICLE_TYPE

logger = logging.getLogger(__name__)
try:
    import logging

    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    from nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.eval.common.utils import quaternion_yaw
    from nuscenes.map_expansion.arcline_path_utils import discretize_lane
    from nuscenes.map_expansion.map_api import NuScenesMap
    from pyquaternion import Quaternion
except ImportError as e:
    logger.warning("Can not import nuscenes-devkit: {}".format(e))

EGO = "ego"


def get_metadrive_type(obj_type):
    meta_type = obj_type
    md_type = None
    if ALL_TYPE[obj_type] == "barrier":
        md_type = MetaDriveType.TRAFFIC_BARRIER
    elif ALL_TYPE[obj_type] == "trafficcone":
        md_type = MetaDriveType.TRAFFIC_CONE
    elif obj_type in VEHICLE_TYPE:
        md_type = MetaDriveType.VEHICLE
    elif obj_type in HUMAN_TYPE:
        md_type = MetaDriveType.PEDESTRIAN
    elif obj_type in BICYCLE_TYPE:
        md_type = MetaDriveType.CYCLIST

    # assert meta_type != MetaDriveType.UNSET and meta_type != "noise"
    return md_type, meta_type


def parse_frame(frame, nuscenes: NuScenes):
    ret = {}
    for obj_id in frame["anns"]:
        obj = nuscenes.get("sample_annotation", obj_id)
        # velocity = nuscenes.box_velocity(obj_id)[:2]
        # if np.nan in velocity:
        velocity = np.array([0.0, 0.0])
        ret[obj["instance_token"]] = {
            "position": obj["translation"],
            "obj_id": obj["instance_token"],
            "heading": quaternion_yaw(Quaternion(*obj["rotation"])),
            "rotation": obj["rotation"],
            "velocity": velocity,
            "size": obj["size"],
            "visible": obj["visibility_token"],
            "attribute": [nuscenes.get("attribute", i)["name"] for i in obj["attribute_tokens"]],
            "type": obj["category_name"]
        }
    ego_token = nuscenes.get("sample_data", frame["data"]["LIDAR_TOP"])["ego_pose_token"]
    ego_state = nuscenes.get("ego_pose", ego_token)
    ret[EGO] = {
        "position": ego_state["translation"],
        "obj_id": EGO,
        "heading": quaternion_yaw(Quaternion(*ego_state["rotation"])),
        "rotation": ego_state["rotation"],
        "type": "vehicle.car",
        "velocity": np.array([0.0, 0.0]),
        # size https://en.wikipedia.org/wiki/Renault_Zoe
        "size": [4.08, 1.73, 1.56],
    }
    return ret


def interpolate_heading(heading_data, old_valid, new_valid, num_to_interpolate=5):
    new_heading_theta = np.zeros_like(new_valid)
    for k, valid in enumerate(old_valid[:-1]):
        if abs(valid) > 1e-1 and abs(old_valid[k + 1]) > 1e-1:
            diff = (heading_data[k + 1] - heading_data[k] + np.pi) % (2 * np.pi) - np.pi
            # step = diff
            interpolate_heading = np.linspace(heading_data[k], heading_data[k] + diff, 6)
            new_heading_theta[k * num_to_interpolate:(k + 1) * num_to_interpolate] = interpolate_heading[:-1]
        elif abs(valid) > 1e-1 and abs(old_valid[k + 1]) < 1e-1:
            new_heading_theta[k * num_to_interpolate:(k + 1) * num_to_interpolate] = heading_data[k]
    new_heading_theta[-1] = heading_data[-1]
    return new_heading_theta * new_valid


def _interpolate_one_dim(data, old_valid, new_valid, num_to_interpolate=5):
    new_data = np.zeros_like(new_valid)
    for k, valid in enumerate(old_valid[:-1]):
        if abs(valid) > 1e-1 and abs(old_valid[k + 1]) > 1e-1:
            diff = data[k + 1] - data[k]
            # step = diff
            interpolate_data = np.linspace(data[k], data[k] + diff, num_to_interpolate + 1)
            new_data[k * num_to_interpolate:(k + 1) * num_to_interpolate] = interpolate_data[:-1]
        elif abs(valid) > 1e-1 and abs(old_valid[k + 1]) < 1e-1:
            new_data[k * num_to_interpolate:(k + 1) * num_to_interpolate] = data[k]
    new_data[-1] = data[-1]
    return new_data * new_valid


def interpolate(origin_y, valid, new_valid):
    if len(origin_y.shape) == 1:
        ret = _interpolate_one_dim(origin_y, valid, new_valid)
    elif len(origin_y.shape) == 2:
        ret = []
        for dim in range(origin_y.shape[-1]):
            new_y = _interpolate_one_dim(origin_y[..., dim], valid, new_valid)
            new_y = np.expand_dims(new_y, axis=-1)
            ret.append(new_y)
        ret = np.concatenate(ret, axis=-1)
    else:
        raise ValueError("Y has shape {}, Can not interpolate".format(origin_y.shape))
    return ret


def get_tracks_from_frames(nuscenes: NuScenes, scene_info, frames, num_to_interpolate=5):
    episode_len = len(frames)
    # Fill tracks
    all_objs = set()
    for frame in frames:
        all_objs.update(frame.keys())
    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(episode_len, 3)),
                heading=np.zeros(shape=(episode_len, )),
                velocity=np.zeros(shape=(episode_len, 2)),
                valid=np.zeros(shape=(episode_len, )),
                length=np.zeros(shape=(episode_len, 1)),
                width=np.zeros(shape=(episode_len, 1)),
                height=np.zeros(shape=(episode_len, 1))
            ),
            metadata=dict(track_length=episode_len, type=MetaDriveType.UNSET, object_id=k, original_id=k)
        )
        for k in list(all_objs)
    }

    tracks_to_remove = set()

    for frame_idx in range(episode_len):
        # Record all agents' states (position, velocity, ...)
        for id, state in frames[frame_idx].items():
            # Fill type
            md_type, meta_type = get_metadrive_type(state["type"])
            tracks[id]["type"] = md_type
            tracks[id][SD.METADATA]["type"] = meta_type
            if md_type is None or md_type == MetaDriveType.UNSET:
                tracks_to_remove.add(id)
                continue

            tracks[id]["type"] = md_type
            tracks[id][SD.METADATA]["type"] = meta_type

            # Introducing the state item
            tracks[id]["state"]["position"][frame_idx] = state["position"]
            tracks[id]["state"]["heading"][frame_idx] = state["heading"]
            tracks[id]["state"]["velocity"][frame_idx] = tracks[id]["state"]["velocity"][frame_idx]
            tracks[id]["state"]["valid"][frame_idx] = 1

            tracks[id]["state"]["length"][frame_idx] = state["size"][1]
            tracks[id]["state"]["width"][frame_idx] = state["size"][0]
            tracks[id]["state"]["height"][frame_idx] = state["size"][2]

            tracks[id]["metadata"]["original_id"] = id
            tracks[id]["metadata"]["object_id"] = id

    for track in tracks_to_remove:
        track_data = tracks.pop(track)
        obj_type = track_data[SD.METADATA]["type"]
        print("\nWARNING: Can not map type: {} to any MetaDrive Type".format(obj_type))

    new_episode_len = (episode_len - 1) * num_to_interpolate + 1

    # interpolate
    interpolate_tracks = {}
    for id, track, in tracks.items():
        interpolate_tracks[id] = copy.deepcopy(track)
        interpolate_tracks[id]["metadata"]["track_length"] = new_episode_len

        # valid first
        new_valid = np.zeros(shape=(new_episode_len, ))
        if track["state"]["valid"][0]:
            new_valid[0] = 1
        for k, valid in enumerate(track["state"]["valid"][1:], start=1):
            if valid:
                if abs(new_valid[(k - 1) * num_to_interpolate] - 1) < 1e-2:
                    start_idx = (k - 1) * num_to_interpolate + 1
                else:
                    start_idx = k * num_to_interpolate
                new_valid[start_idx:k * num_to_interpolate + 1] = 1
        interpolate_tracks[id]["state"]["valid"] = new_valid

        # position
        interpolate_tracks[id]["state"]["position"] = interpolate(
            track["state"]["position"], track["state"]["valid"], new_valid
        )
        if id == "ego" and not scene_info.get("prediction", False):
            assert "prediction" not in scene_info
            # We can get it from canbus
            try:
                canbus = NuScenesCanBus(dataroot=nuscenes.dataroot)
                imu_pos = np.asarray([state["pos"] for state in canbus.get_messages(scene_info["name"], "pose")[::5]])
                min_len = min(len(imu_pos), new_episode_len)
                interpolate_tracks[id]["state"]["position"][:min_len] = imu_pos[:min_len]
            except:
                logger.info("Fail to get canbus data for {}".format(scene_info["name"]))

        # velocity
        interpolate_tracks[id]["state"]["velocity"] = interpolate(
            track["state"]["velocity"], track["state"]["valid"], new_valid
        )
        vel = interpolate_tracks[id]["state"]["position"][1:] - interpolate_tracks[id]["state"]["position"][:-1]
        interpolate_tracks[id]["state"]["velocity"][:-1] = vel[..., :2] / 0.1
        for k, valid in enumerate(new_valid[1:], start=1):
            if valid == 0 or not valid or abs(valid) < 1e-2:
                interpolate_tracks[id]["state"]["velocity"][k] = np.array([0., 0.])
                interpolate_tracks[id]["state"]["velocity"][k - 1] = np.array([0., 0.])
        # speed outlier check
        max_vel = np.max(np.linalg.norm(interpolate_tracks[id]["state"]["velocity"], axis=-1))
        if max_vel > 30:
            print("\nWARNING: Too large speed for {}: {}".format(id, max_vel))

        # heading
        # then update position
        new_heading = interpolate_heading(track["state"]["heading"], track["state"]["valid"], new_valid)
        interpolate_tracks[id]["state"]["heading"] = new_heading
        if id == "ego" and not scene_info.get("prediction", False):
            assert "prediction" not in scene_info
            # We can get it from canbus
            try:
                canbus = NuScenesCanBus(dataroot=nuscenes.dataroot)
                imu_heading = np.asarray(
                    [
                        quaternion_yaw(Quaternion(state["orientation"]))
                        for state in canbus.get_messages(scene_info["name"], "pose")[::5]
                    ]
                )
                min_len = min(len(imu_heading), new_episode_len)
                interpolate_tracks[id]["state"]["heading"][:min_len] = imu_heading[:min_len]
            except:
                logger.info("Fail to get canbus data for {}".format(scene_info["name"]))

        for k, v in track["state"].items():
            if k in ["valid", "heading", "position", "velocity"]:
                continue
            else:
                interpolate_tracks[id]["state"][k] = interpolate(v, track["state"]["valid"], new_valid)
        # if id == "ego":
        # ego is valid all time, so we can calculate the velocity in this way
    return interpolate_tracks


def get_map_features(scene_info, nuscenes: NuScenes, map_center, radius=500, points_distance=1, only_lane=False):
    """
    Extract map features from nuscenes data. The objects in specified region will be returned. Sampling rate determines
    the distance between 2 points when extracting lane center line.
    """
    ret = {}
    map_name = nuscenes.get("log", scene_info["log_token"])["location"]
    map_api = NuScenesMap(dataroot=nuscenes.dataroot, map_name=map_name)

    layer_names = [
        # "line",
        # "polygon",
        # "node",
        'drivable_area',
        'road_segment',
        'road_block',
        'lane',
        'ped_crossing',
        'walkway',
        # 'stop_line',
        # 'carpark_area',
        'lane_connector',
        'road_divider',
        'lane_divider',
        'traffic_light'
    ]
    # road segment includes all roadblocks (a list of lanes in the same direction), intersection and unstructured road

    map_objs = map_api.get_records_in_radius(map_center[0], map_center[1], radius, layer_names)

    if not only_lane:
        # build map boundary
        polygons = []
        for id in map_objs["drivable_area"]:
            seg_info = map_api.get("drivable_area", id)
            assert seg_info["token"] == id
            for polygon_token in seg_info["polygon_tokens"]:
                polygon = map_api.extract_polygon(polygon_token)
                polygons.append(polygon)
        # for id in map_objs["road_segment"]:
        #     seg_info = map_api.get("road_segment", id)
        #     assert seg_info["token"] == id
        #     polygon = map_api.extract_polygon(seg_info["polygon_token"])
        #     polygons.append(polygon)
        # for id in map_objs["road_block"]:
        #     seg_info = map_api.get("road_block", id)
        #     assert seg_info["token"] == id
        #     polygon = map_api.extract_polygon(seg_info["polygon_token"])
        #     polygons.append(polygon)
        polygons = [geom if geom.is_valid else geom.buffer(0) for geom in polygons]
        boundaries = gpd.GeoSeries(unary_union(polygons)).boundary.explode(index_parts=True)
        for idx, boundary in enumerate(boundaries[0]):
            block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
            id = "boundary_{}".format(idx)
            ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points}

        # broken line
        for id in map_objs["lane_divider"]:
            line_info = map_api.get("lane_divider", id)
            assert line_info["token"] == id
            line = map_api.extract_line(line_info["line_token"]).coords.xy
            line = np.asarray([[line[0][i], line[1][i]] for i in range(len(line[0]))])
            ret[id] = {SD.TYPE: MetaDriveType.LINE_BROKEN_SINGLE_WHITE, SD.POLYLINE: line}

        # solid line
        for id in map_objs["road_divider"]:
            line_info = map_api.get("road_divider", id)
            assert line_info["token"] == id
            line = map_api.extract_line(line_info["line_token"]).coords.xy
            line = np.asarray([[line[0][i], line[1][i]] for i in range(len(line[0]))])
            ret[id] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_YELLOW, SD.POLYLINE: line}

        # crosswalk
        for id in map_objs["ped_crossing"]:
            info = map_api.get("ped_crossing", id)
            assert info["token"] == id
            boundary = map_api.extract_polygon(info["polygon_token"]).exterior.xy
            boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])
            ret[id] = {
                SD.TYPE: MetaDriveType.CROSSWALK,
                SD.POLYGON: boundary_polygon,
            }

        # walkway
        for id in map_objs["walkway"]:
            info = map_api.get("walkway", id)
            assert info["token"] == id
            boundary = map_api.extract_polygon(info["polygon_token"]).exterior.xy
            boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])
            ret[id] = {
                SD.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
                SD.POLYGON: boundary_polygon,
            }

    # normal lane
    for id in map_objs["lane"]:
        lane_info = map_api.get("lane", id)
        assert lane_info["token"] == id
        boundary = map_api.extract_polygon(lane_info["polygon_token"]).boundary.xy
        boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])
        # boundary_polygon += [[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))]
        ret[id] = {
            SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
            SD.POLYLINE: np.asarray(discretize_lane(map_api.arcline_path_3[id], resolution_meters=points_distance)),
            SD.POLYGON: boundary_polygon,
            SD.ENTRY: map_api.get_incoming_lane_ids(id),
            SD.EXIT: map_api.get_outgoing_lane_ids(id),
            SD.LEFT_NEIGHBORS: [],
            SD.RIGHT_NEIGHBORS: [],
        }

    # intersection lane
    for id in map_objs["lane_connector"]:
        lane_info = map_api.get("lane_connector", id)
        assert lane_info["token"] == id
        # boundary = map_api.extract_polygon(lane_info["polygon_token"]).boundary.xy
        # boundary_polygon = [[boundary[0][i], boundary[1][i], 0.1] for i in range(len(boundary[0]))]
        # boundary_polygon += [[boundary[0][i], boundary[1][i], 0.] for i in range(len(boundary[0]))]
        ret[id] = {
            SD.TYPE: MetaDriveType.LANE_SURFACE_UNSTRUCTURE,
            SD.POLYLINE: np.asarray(discretize_lane(map_api.arcline_path_3[id], resolution_meters=points_distance)),
            # SD.POLYGON: boundary_polygon,
            "speed_limit_kmh": 100,
            SD.ENTRY: map_api.get_incoming_lane_ids(id),
            SD.EXIT: map_api.get_outgoing_lane_ids(id)
        }

    # # stop_line
    # for id in map_objs["stop_line"]:
    #     info = map_api.get("stop_line", id)
    #     assert info["token"] == id
    #     boundary = map_api.extract_polygon(info["polygon_token"]).exterior.xy
    #     boundary_polygon = np.asarray([[boundary[0][i], boundary[1][i]] for i in range(len(boundary[0]))])
    #     ret[id] = {
    #         SD.TYPE: MetaDriveType.STOP_LINE,
    #         SD.POLYGON: boundary_polygon ,
    #     }

    #         'stop_line',
    #         'carpark_area',

    return ret


def convert_nuscenes_scenario(
    token, version, nuscenes: NuScenes, map_radius=500, prediction=False, past=2, future=6, only_lane=False
):
    """
    Data will be interpolated to 0.1s time interval, while the time interval of original key frames are 0.5s.
    """
    if prediction:
        past_num = int(past / 0.5)
        future_num = int(future / 0.5)
        nusc = nuscenes
        instance_token, sample_token = token.split("_")
        current_sample = last_sample = next_sample = nusc.get("sample", sample_token)
        past_samples = []
        future_samples = []
        for _ in range(past_num):
            if last_sample["prev"] == "":
                break
            last_sample = nusc.get("sample", last_sample["prev"])
            past_samples.append(parse_frame(last_sample, nusc))

        for _ in range(future_num):
            if next_sample["next"] == "":
                break
            next_sample = nusc.get("sample", next_sample["next"])
            future_samples.append(parse_frame(next_sample, nusc))
        frames = past_samples[::-1] + [parse_frame(current_sample, nusc)] + future_samples
        scene_info = copy.copy(nusc.get("scene", current_sample["scene_token"]))
        scene_info["name"] = scene_info["name"] + "_" + token
        scene_info["prediction"] = True
        frames_scene_info = [frames, scene_info]
    else:
        frames_scene_info = extract_frames_scene_info(token, nuscenes)

    scenario_log_interval = 0.1
    frames, scene_info = frames_scene_info
    result = SD()
    result[SD.ID] = scene_info["name"]
    result[SD.VERSION] = "nuscenes" + version
    result[SD.LENGTH] = (len(frames) - 1) * 5 + 1
    result[SD.METADATA] = {}
    result[SD.METADATA]["dataset"] = "nuscenes"
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    result[SD.METADATA]["map"] = nuscenes.get("log", scene_info["log_token"])["location"]
    result[SD.METADATA]["date"] = nuscenes.get("log", scene_info["log_token"])["date_captured"]
    result[SD.METADATA]["coordinate"] = "right-handed"
    # result[SD.METADATA]["dscenario_token"] = scene_token
    result[SD.METADATA][SD.ID] = scene_info["name"]
    result[SD.METADATA]["scenario_id"] = scene_info["name"]
    result[SD.METADATA]["sample_rate"] = scenario_log_interval
    result[SD.METADATA][SD.TIMESTEP] = np.arange(0., (len(frames) - 1) * 0.5 + 0.1, 0.1)
    # interpolating to 0.1s interval
    result[SD.TRACKS] = get_tracks_from_frames(nuscenes, scene_info, frames, num_to_interpolate=5)
    result[SD.METADATA][SD.SDC_ID] = "ego"

    # No traffic light in nuscenes at this stage
    result[SD.DYNAMIC_MAP_STATES] = {}
    if prediction:
        track_to_predict = result[SD.TRACKS][instance_token]
        result[SD.METADATA]["tracks_to_predict"] = {
            instance_token: {
                "track_index": list(result[SD.TRACKS].keys()).index(instance_token),
                "track_id": instance_token,
                "difficulty": 0,
                "object_type": track_to_predict['type']
            }
        }

    # map
    map_center = np.array(result[SD.TRACKS]["ego"]["state"]["position"][0])
    result[SD.MAP_FEATURES] = get_map_features(scene_info, nuscenes, map_center, map_radius, only_lane=only_lane)
    del frames_scene_info
    del frames
    del scene_info
    return result


def extract_frames_scene_info(scene, nuscenes):
    scene_token = scene["token"]
    scene_info = nuscenes.get("scene", scene_token)
    frames = []
    current_frame = nuscenes.get("sample", scene_info["first_sample_token"])
    while current_frame["token"] != scene_info["last_sample_token"]:
        frames.append(parse_frame(current_frame, nuscenes))
        current_frame = nuscenes.get("sample", current_frame["next"])
    frames.append(parse_frame(current_frame, nuscenes))
    assert current_frame["next"] == ""
    assert len(frames) == scene_info["nbr_samples"], "Number of sample mismatches! "
    return frames, scene_info


def get_nuscenes_scenarios(dataroot, version, num_workers=2):
    nusc = NuScenes(version=version, dataroot=dataroot)

    def _get_nusc():
        return NuScenes(version=version, dataroot=dataroot)

    return nusc.scene, [nusc for _ in range(num_workers)]


def get_nuscenes_prediction_split(dataroot, version, past, future, num_workers=2):
    def _get_nusc():
        return NuScenes(version="v1.0-mini" if "mini" in version else "v1.0-trainval", dataroot=dataroot)

    nusc = _get_nusc()
    return get_prediction_challenge_split(version, dataroot=dataroot), [nusc for _ in range(num_workers)]
