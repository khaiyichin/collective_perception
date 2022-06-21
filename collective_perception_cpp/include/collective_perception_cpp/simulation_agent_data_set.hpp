#ifndef SIMULATION_AGENT_SET_HPP
#define SIMULATION_AGENT_SET_HPP

#include <iterator>
#include <algorithm>

#include "data_common.hpp"
#include "simulation_set.hpp"
#include "simulation_set.pb.h"

using RTAgentDataProtoMsg = collective_perception_cpp::proto::RepeatedTrialAgentData;
using MultiAgentDataProtoMsg = collective_perception_cpp::proto::RepeatedTrialAgentData::MultiAgentData;
using AgentDataProtoMsg = collective_perception_cpp::proto::RepeatedTrialAgentData::AgentData;
using SimSetMsg = collective_perception_cpp::proto::SimulationSet;
using PacketMsg = collective_perception_cpp::proto::Packet;

/**
 * @brief Class to store all statistics from all the experiments
 *
 * A SimulationAgentDataSet object stores sets of statistics from simulations (i.e., `AgentDataPacket`s) with the same:
 *  - number of experiments,
 *  - number of agents,
 *  - number of steps,
 *  - communication range,
 *  - robot speed, and
 *  - swarm density
 * and with varying:
 *  - target fill ratios, and
 *  - sensor probabilities.
 *
 */
class SimulationAgentDataSet : public SimulationSet
{
public:
    SimulationAgentDataSet() {}

    /**
     * @brief Construct a new SimulationAgentDataSet object from SimulationAgentDataSet protobuf message
     *
     * @param sim_agent_data_set_msg
     */
    SimulationAgentDataSet(collective_perception_cpp::proto::SimulationAgentDataSet &sim_agent_data_set_msg);

    /**
     * @brief Serialize into SimulationAgentDataSet protobuf message
     *
     * @param sim_agent_data_set_msg
     */
    void Serialize(collective_perception_cpp::proto::SimulationAgentDataSet &sim_agent_data_set_msg);

    inline void InsertAgentDataPacket(const AgentDataPacket &packet) { SimulationSet::InsertPacket(packet, packets_); }

    inline AgentDataPacket GetAgentDataPacket(const float &tfr, const float &sp) { return SimulationSet::GetPacket(tfr, sp, packets_); }

private:

    /**
     * @brief Extract RepeatedTrialAgentData protobuf message
     *
     * @param vec All AgentData objects for all repeated trials
     * @return RTAgentDataProtoMsg protobuf message
     */
    RTAgentDataProtoMsg ExtractRepeatedTrialAgentDataMsg(const RepeatedTrials<std::vector<AgentData>> &vec);

    PacketDict<AgentDataPacket> packets_; ///< Unordered map storing T structs
};

#endif