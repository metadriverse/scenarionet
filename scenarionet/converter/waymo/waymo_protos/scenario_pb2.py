# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scenarionet/converter/waymo/waymo_protos/scenario.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from scenarionet.converter.waymo.waymo_protos import compressed_lidar_pb2 as scenarionet_dot_converter_dot_waymo_dot_waymo__protos_dot_compressed__lidar__pb2
from scenarionet.converter.waymo.waymo_protos import map_pb2 as scenarionet_dot_converter_dot_waymo_dot_waymo__protos_dot_map__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n7scenarionet/converter/waymo/waymo_protos/scenario.proto\x12\x12waymo.open_dataset\x1a?scenarionet/converter/waymo/waymo_protos/compressed_lidar.proto\x1a\x32scenarionet/converter/waymo/waymo_protos/map.proto\"\xba\x01\n\x0bObjectState\x12\x10\n\x08\x63\x65nter_x\x18\x02 \x01(\x01\x12\x10\n\x08\x63\x65nter_y\x18\x03 \x01(\x01\x12\x10\n\x08\x63\x65nter_z\x18\x04 \x01(\x01\x12\x0e\n\x06length\x18\x05 \x01(\x02\x12\r\n\x05width\x18\x06 \x01(\x02\x12\x0e\n\x06height\x18\x07 \x01(\x02\x12\x0f\n\x07heading\x18\x08 \x01(\x02\x12\x12\n\nvelocity_x\x18\t \x01(\x02\x12\x12\n\nvelocity_y\x18\n \x01(\x02\x12\r\n\x05valid\x18\x0b \x01(\x08\"\xe6\x01\n\x05Track\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x39\n\x0bobject_type\x18\x02 \x01(\x0e\x32$.waymo.open_dataset.Track.ObjectType\x12/\n\x06states\x18\x03 \x03(\x0b\x32\x1f.waymo.open_dataset.ObjectState\"e\n\nObjectType\x12\x0e\n\nTYPE_UNSET\x10\x00\x12\x10\n\x0cTYPE_VEHICLE\x10\x01\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x02\x12\x10\n\x0cTYPE_CYCLIST\x10\x03\x12\x0e\n\nTYPE_OTHER\x10\x04\"R\n\x0f\x44ynamicMapState\x12?\n\x0blane_states\x18\x01 \x03(\x0b\x32*.waymo.open_dataset.TrafficSignalLaneState\"\xac\x01\n\x12RequiredPrediction\x12\x13\n\x0btrack_index\x18\x01 \x01(\x05\x12J\n\ndifficulty\x18\x02 \x01(\x0e\x32\x36.waymo.open_dataset.RequiredPrediction.DifficultyLevel\"5\n\x0f\x44ifficultyLevel\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07LEVEL_1\x10\x01\x12\x0b\n\x07LEVEL_2\x10\x02\"\xcb\x03\n\x08Scenario\x12\x13\n\x0bscenario_id\x18\x05 \x01(\t\x12\x1a\n\x12timestamps_seconds\x18\x01 \x03(\x01\x12\x1a\n\x12\x63urrent_time_index\x18\n \x01(\x05\x12)\n\x06tracks\x18\x02 \x03(\x0b\x32\x19.waymo.open_dataset.Track\x12?\n\x12\x64ynamic_map_states\x18\x07 \x03(\x0b\x32#.waymo.open_dataset.DynamicMapState\x12\x34\n\x0cmap_features\x18\x08 \x03(\x0b\x32\x1e.waymo.open_dataset.MapFeature\x12\x17\n\x0fsdc_track_index\x18\x06 \x01(\x05\x12\x1b\n\x13objects_of_interest\x18\x04 \x03(\x05\x12\x41\n\x11tracks_to_predict\x18\x0b \x03(\x0b\x32&.waymo.open_dataset.RequiredPrediction\x12Q\n\x1b\x63ompressed_frame_laser_data\x18\x0c \x03(\x0b\x32,.waymo.open_dataset.CompressedFrameLaserDataJ\x04\x08\t\x10\n')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'scenarionet.converter.waymo.waymo_protos.scenario_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_OBJECTSTATE']._serialized_start=197
  _globals['_OBJECTSTATE']._serialized_end=383
  _globals['_TRACK']._serialized_start=386
  _globals['_TRACK']._serialized_end=616
  _globals['_TRACK_OBJECTTYPE']._serialized_start=515
  _globals['_TRACK_OBJECTTYPE']._serialized_end=616
  _globals['_DYNAMICMAPSTATE']._serialized_start=618
  _globals['_DYNAMICMAPSTATE']._serialized_end=700
  _globals['_REQUIREDPREDICTION']._serialized_start=703
  _globals['_REQUIREDPREDICTION']._serialized_end=875
  _globals['_REQUIREDPREDICTION_DIFFICULTYLEVEL']._serialized_start=822
  _globals['_REQUIREDPREDICTION_DIFFICULTYLEVEL']._serialized_end=875
  _globals['_SCENARIO']._serialized_start=878
  _globals['_SCENARIO']._serialized_end=1337
# @@protoc_insertion_point(module_scope)
