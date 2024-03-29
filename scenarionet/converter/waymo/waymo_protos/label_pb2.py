# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scenarionet/converter/waymo/waymo_protos/label.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

from scenarionet.converter.waymo.waymo_protos import keypoint_pb2 as scenarionet_dot_converter_dot_waymo_dot_waymo__protos_dot_keypoint__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n4scenarionet/converter/waymo/waymo_protos/label.proto\x12\x12waymo.open_dataset\x1a\x37scenarionet/converter/waymo/waymo_protos/keypoint.proto\"\xbd\t\n\x05Label\x12*\n\x03\x62ox\x18\x01 \x01(\x0b\x32\x1d.waymo.open_dataset.Label.Box\x12\x34\n\x08metadata\x18\x02 \x01(\x0b\x32\".waymo.open_dataset.Label.Metadata\x12,\n\x04type\x18\x03 \x01(\x0e\x32\x1e.waymo.open_dataset.Label.Type\x12\n\n\x02id\x18\x04 \x01(\t\x12M\n\x1a\x64\x65tection_difficulty_level\x18\x05 \x01(\x0e\x32).waymo.open_dataset.Label.DifficultyLevel\x12L\n\x19tracking_difficulty_level\x18\x06 \x01(\x0e\x32).waymo.open_dataset.Label.DifficultyLevel\x12\x1f\n\x17num_lidar_points_in_box\x18\x07 \x01(\x05\x12#\n\x1bnum_top_lidar_points_in_box\x18\r \x01(\x05\x12G\n\x0flaser_keypoints\x18\x08 \x01(\x0b\x32,.waymo.open_dataset.keypoints.LaserKeypointsH\x00\x12I\n\x10\x63\x61mera_keypoints\x18\t \x01(\x0b\x32-.waymo.open_dataset.keypoints.CameraKeypointsH\x00\x12:\n\x0b\x61ssociation\x18\n \x01(\x0b\x32%.waymo.open_dataset.Label.Association\x12 \n\x18most_visible_camera_name\x18\x0b \x01(\t\x12\x38\n\x11\x63\x61mera_synced_box\x18\x0c \x01(\x0b\x32\x1d.waymo.open_dataset.Label.Box\x1a\xbf\x01\n\x03\x42ox\x12\x10\n\x08\x63\x65nter_x\x18\x01 \x01(\x01\x12\x10\n\x08\x63\x65nter_y\x18\x02 \x01(\x01\x12\x10\n\x08\x63\x65nter_z\x18\x03 \x01(\x01\x12\x0e\n\x06length\x18\x05 \x01(\x01\x12\r\n\x05width\x18\x04 \x01(\x01\x12\x0e\n\x06height\x18\x06 \x01(\x01\x12\x0f\n\x07heading\x18\x07 \x01(\x01\"B\n\x04Type\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x0b\n\x07TYPE_3D\x10\x01\x12\x0b\n\x07TYPE_2D\x10\x02\x12\x0e\n\nTYPE_AA_2D\x10\x03\x1ap\n\x08Metadata\x12\x0f\n\x07speed_x\x18\x01 \x01(\x01\x12\x0f\n\x07speed_y\x18\x02 \x01(\x01\x12\x0f\n\x07speed_z\x18\x05 \x01(\x01\x12\x0f\n\x07\x61\x63\x63\x65l_x\x18\x03 \x01(\x01\x12\x0f\n\x07\x61\x63\x63\x65l_y\x18\x04 \x01(\x01\x12\x0f\n\x07\x61\x63\x63\x65l_z\x18\x06 \x01(\x01\x1a&\n\x0b\x41ssociation\x12\x17\n\x0flaser_object_id\x18\x01 \x01(\t\"`\n\x04Type\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x10\n\x0cTYPE_VEHICLE\x10\x01\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x02\x12\r\n\tTYPE_SIGN\x10\x03\x12\x10\n\x0cTYPE_CYCLIST\x10\x04\"8\n\x0f\x44ifficultyLevel\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0b\n\x07LEVEL_1\x10\x01\x12\x0b\n\x07LEVEL_2\x10\x02\x42\x11\n\x0fkeypoints_oneof\"2\n\x0ePolygon2dProto\x12\t\n\x01x\x18\x01 \x03(\x01\x12\t\n\x01y\x18\x02 \x03(\x01\x12\n\n\x02id\x18\x03 \x01(\t'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'scenarionet.converter.waymo.waymo_protos.label_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _globals['_LABEL']._serialized_start = 134
    _globals['_LABEL']._serialized_end = 1347
    _globals['_LABEL_BOX']._serialized_start = 827
    _globals['_LABEL_BOX']._serialized_end = 1018
    _globals['_LABEL_BOX_TYPE']._serialized_start = 952
    _globals['_LABEL_BOX_TYPE']._serialized_end = 1018
    _globals['_LABEL_METADATA']._serialized_start = 1020
    _globals['_LABEL_METADATA']._serialized_end = 1132
    _globals['_LABEL_ASSOCIATION']._serialized_start = 1134
    _globals['_LABEL_ASSOCIATION']._serialized_end = 1172
    _globals['_LABEL_TYPE']._serialized_start = 1174
    _globals['_LABEL_TYPE']._serialized_end = 1270
    _globals['_LABEL_DIFFICULTYLEVEL']._serialized_start = 1272
    _globals['_LABEL_DIFFICULTYLEVEL']._serialized_end = 1328
    _globals['_POLYGON2DPROTO']._serialized_start = 1349
    _globals['_POLYGON2DPROTO']._serialized_end = 1399
# @@protoc_insertion_point(module_scope)
