
'Generated protocol buffer code.'
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from . import util_pb2 as util__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14simulation_set.proto\x12\x1fcollective_perception_cpp.proto\x1a\nutil.proto"\xa2\x02\n\rSimulationSet\x12\x10\n\x08sim_type\x18\x01 \x01(\t\x12\x12\n\nnum_agents\x18\x02 \x01(\r\x12\x12\n\nnum_trials\x18\x03 \x01(\r\x12\x15\n\ttfr_range\x18\x04 \x03(\x01B\x02\x10\x01\x12\x14\n\x08sp_range\x18\x05 \x03(\x01B\x02\x10\x01\x12\x13\n\x0bcomms_graph\x18\x07 \x01(\t\x12\x0f\n\x07num_obs\x18\x08 \x01(\r\x12\x14\n\x0ccomms_period\x18\t \x01(\r\x12\x12\n\ncomms_prob\x18\n \x01(\x02\x12\x11\n\tnum_steps\x18\x0b \x01(\r\x12\x13\n\x0bcomms_range\x18\x0c \x01(\x02\x12\x0f\n\x07density\x18\r \x01(\x02\x12\r\n\x05speed\x18\x0e \x01(\x02\x12\x12\n\nassumed_sp\x18\x0f \x01(\x02"\x9a\x01\n\x12SimulationStatsSet\x12?\n\x07sim_set\x18\x01 \x01(\x0b2..collective_perception_cpp.proto.SimulationSet\x12C\n\rstats_packets\x18\x02 \x03(\x0b2,.collective_perception_cpp.proto.StatsPacket"\xa7\x01\n\x16SimulationAgentDataSet\x12?\n\x07sim_set\x18\x01 \x01(\x0b2..collective_perception_cpp.proto.SimulationSet\x12L\n\x12agent_data_packets\x18\x02 \x03(\x0b20.collective_perception_cpp.proto.AgentDataPacketb\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'simulation_set_pb2', globals())
if (_descriptor._USE_C_DESCRIPTORS == False):
    DESCRIPTOR._options = None
    _SIMULATIONSET.fields_by_name['tfr_range']._options = None
    _SIMULATIONSET.fields_by_name['tfr_range']._serialized_options = b'\x10\x01'
    _SIMULATIONSET.fields_by_name['sp_range']._options = None
    _SIMULATIONSET.fields_by_name['sp_range']._serialized_options = b'\x10\x01'
    _SIMULATIONSET._serialized_start = 70
    _SIMULATIONSET._serialized_end = 360
    _SIMULATIONSTATSSET._serialized_start = 363
    _SIMULATIONSTATSSET._serialized_end = 517
    _SIMULATIONAGENTDATASET._serialized_start = 520
    _SIMULATIONAGENTDATASET._serialized_end = 687
