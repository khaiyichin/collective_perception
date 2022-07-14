
'Generated protocol buffer code.'
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nutil.proto\x12\x1fcollective_perception_cpp.proto"\x9f\x02\n\x16RepeatedTrialAgentData\x12[\n\x0bmultiagents\x18\x01 \x03(\x0b2F.collective_perception_cpp.proto.RepeatedTrialAgentData.MultiAgentData\x1aC\n\tAgentData\x12\x1c\n\x10tile_occurrences\x18\x01 \x03(\rB\x02\x10\x01\x12\x18\n\x0cobservations\x18\x02 \x03(\rB\x02\x10\x01\x1ac\n\x0eMultiAgentData\x12Q\n\x06agents\x18\x01 \x03(\x0b2A.collective_perception_cpp.proto.RepeatedTrialAgentData.AgentData"\x97\x05\n\x12RepeatedTrialStats\x12M\n\nlocal_vals\x18\x01 \x03(\x0b29.collective_perception_cpp.proto.RepeatedTrialStats.Stats\x12N\n\x0bsocial_vals\x18\x02 \x03(\x0b29.collective_perception_cpp.proto.RepeatedTrialStats.Stats\x12P\n\rinformed_vals\x18\x03 \x03(\x0b29.collective_perception_cpp.proto.RepeatedTrialStats.Stats\x12\x18\n\x0csp_mean_vals\x18\x04 \x03(\x02B\x02\x10\x01\x12d\n\x13swarm_informed_vals\x18\x05 \x03(\x0b2G.collective_perception_cpp.proto.RepeatedTrialStats.SwarmInformedValues\x1a[\n\x05Stats\x12\x12\n\x06x_mean\x18\x01 \x03(\x02B\x02\x10\x01\x12\x15\n\tconf_mean\x18\x02 \x03(\x02B\x02\x10\x01\x12\x11\n\x05x_std\x18\x03 \x03(\x02B\x02\x10\x01\x12\x14\n\x08conf_std\x18\x04 \x03(\x02B\x02\x10\x01\x1a6\n\x13AgentInformedValues\x12\r\n\x01x\x18\x01 \x03(\x02B\x02\x10\x01\x12\x10\n\x04conf\x18\x02 \x03(\x02B\x02\x10\x01\x1a{\n\x13SwarmInformedValues\x12d\n\x13agent_informed_vals\x18\x01 \x03(\x0b2G.collective_perception_cpp.proto.RepeatedTrialStats.AgentInformedValues"\x87\x02\n\x06Packet\x12\x10\n\x08sim_type\x18\x01 \x01(\t\x12\x0b\n\x03tfr\x18\x02 \x01(\x01\x12\x0e\n\x06b_prob\x18\x03 \x01(\x01\x12\x0e\n\x06w_prob\x18\x04 \x01(\x01\x12\x12\n\nnum_agents\x18\x05 \x01(\r\x12\x12\n\nnum_trials\x18\x06 \x01(\r\x12\x13\n\x0bcomms_graph\x18\t \x01(\t\x12\x0f\n\x07num_obs\x18\n \x01(\r\x12\x14\n\x0ccomms_period\x18\x0b \x01(\r\x12\x12\n\ncomms_prob\x18\x0c \x01(\x02\x12\x11\n\tnum_steps\x18\r \x01(\r\x12\x13\n\x0bcomms_range\x18\x0e \x01(\x02\x12\x0f\n\x07density\x18\x0f \x01(\x02\x12\r\n\x05speed\x18\x10 \x01(\x02"\x88\x01\n\x0bStatsPacket\x127\n\x06packet\x18\x01 \x01(\x0b2\'.collective_perception_cpp.proto.Packet\x12@\n\x03rts\x18\x02 \x01(\x0b23.collective_perception_cpp.proto.RepeatedTrialStats"\x91\x01\n\x0fAgentDataPacket\x127\n\x06packet\x18\x01 \x01(\x0b2\'.collective_perception_cpp.proto.Packet\x12E\n\x04rtad\x18\x02 \x01(\x0b27.collective_perception_cpp.proto.RepeatedTrialAgentDatab\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'util_pb2', globals())
if (_descriptor._USE_C_DESCRIPTORS == False):
    DESCRIPTOR._options = None
    _REPEATEDTRIALAGENTDATA_AGENTDATA.fields_by_name['tile_occurrences']._options = None
    _REPEATEDTRIALAGENTDATA_AGENTDATA.fields_by_name['tile_occurrences']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALAGENTDATA_AGENTDATA.fields_by_name['observations']._options = None
    _REPEATEDTRIALAGENTDATA_AGENTDATA.fields_by_name['observations']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS_STATS.fields_by_name['x_mean']._options = None
    _REPEATEDTRIALSTATS_STATS.fields_by_name['x_mean']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS_STATS.fields_by_name['conf_mean']._options = None
    _REPEATEDTRIALSTATS_STATS.fields_by_name['conf_mean']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS_STATS.fields_by_name['x_std']._options = None
    _REPEATEDTRIALSTATS_STATS.fields_by_name['x_std']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS_STATS.fields_by_name['conf_std']._options = None
    _REPEATEDTRIALSTATS_STATS.fields_by_name['conf_std']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS_AGENTINFORMEDVALUES.fields_by_name['x']._options = None
    _REPEATEDTRIALSTATS_AGENTINFORMEDVALUES.fields_by_name['x']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS_AGENTINFORMEDVALUES.fields_by_name['conf']._options = None
    _REPEATEDTRIALSTATS_AGENTINFORMEDVALUES.fields_by_name['conf']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALSTATS.fields_by_name['sp_mean_vals']._options = None
    _REPEATEDTRIALSTATS.fields_by_name['sp_mean_vals']._serialized_options = b'\x10\x01'
    _REPEATEDTRIALAGENTDATA._serialized_start = 48
    _REPEATEDTRIALAGENTDATA._serialized_end = 335
    _REPEATEDTRIALAGENTDATA_AGENTDATA._serialized_start = 167
    _REPEATEDTRIALAGENTDATA_AGENTDATA._serialized_end = 234
    _REPEATEDTRIALAGENTDATA_MULTIAGENTDATA._serialized_start = 236
    _REPEATEDTRIALAGENTDATA_MULTIAGENTDATA._serialized_end = 335
    _REPEATEDTRIALSTATS._serialized_start = 338
    _REPEATEDTRIALSTATS._serialized_end = 1001
    _REPEATEDTRIALSTATS_STATS._serialized_start = 729
    _REPEATEDTRIALSTATS_STATS._serialized_end = 820
    _REPEATEDTRIALSTATS_AGENTINFORMEDVALUES._serialized_start = 822
    _REPEATEDTRIALSTATS_AGENTINFORMEDVALUES._serialized_end = 876
    _REPEATEDTRIALSTATS_SWARMINFORMEDVALUES._serialized_start = 878
    _REPEATEDTRIALSTATS_SWARMINFORMEDVALUES._serialized_end = 1001
    _PACKET._serialized_start = 1004
    _PACKET._serialized_end = 1267
    _STATSPACKET._serialized_start = 1270
    _STATSPACKET._serialized_end = 1406
    _AGENTDATAPACKET._serialized_start = 1409
    _AGENTDATAPACKET._serialized_end = 1554
