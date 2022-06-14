#ifndef SIMULATION_DATA_SET_HPP
#define SIMULATION_DATA_SET_HPP

#include <iterator>
#include <algorithm>

#include "data_common.hpp"
#include "simulation_data_set.pb.h"

using RTAgentDataProtoMsg = collective_perception_cpp::proto::RepeatedTrialAgentData;
using MultiAgentDataProtoMsg = collective_perception_cpp::proto::RepeatedTrialAgentData::MultiAgentData;
using AgentDataProtoMsg = collective_perception_cpp::proto::RepeatedTrialAgentData::AgentData;

using RTStatsProtoMsg = collective_perception_cpp::proto::RepeatedTrialStats;
using StatsProtoMsg = collective_perception_cpp::proto::RepeatedTrialStats::Stats;

/**
 * @brief Class to store all data from all the experiments
 *
 * A SimulationDataSet object stores sets of data from simulations (i.e., `SimPacket`s) with the same:
 *  - number of experiments,
 *  - number of agents,
 *  - number of steps,
 *  - communication range, and
 *  - swarm density
 * and with varying:
 *  - target fill ratios, and
 *  - sensor probabilities.
 *
 */
class SimulationDataSet
{
public:
    SimulationDataSet() {}

    /**
     * @brief Construct a new SimulationDataSet object from SimulationDataSet protobuf message
     *
     * @param sim_data_set_msg
     */
    SimulationDataSet(collective_perception_cpp::proto::SimulationDataSet &sim_data_set_msg);

    /**
     * @brief Serialize into SimulationDataSet protobuf message
     *
     * @param sim_data_set_msg
     */
    void Serialize(collective_perception_cpp::proto::SimulationDataSet &sim_data_set_msg);

    /**
     * @brief Get the SimPacket object
     *
     * @param tfr
     * @param sp
     * @return SimPacket
     */
    SimPacket GetSimPacket(const float &tfr, const float &sp);

    /**
     * @brief Insert a SimPacket object
     *
     * @param packet SimPacket object
     */
    void InsertSimPacket(const SimPacket &packet);

    std::string simulation_type_ = "dynamic";

    unsigned int num_agents_;

    unsigned int num_trials_;

    unsigned int num_steps_;

    float comms_range_;

    float density_;

    std::vector<float> tfr_range_;

    std::vector<float> sp_range_;

private:
    /**
     * @brief Convert the provided value into internal units: 0.005 = 5
     *
     * @param val The value in physical units to be converted
     * @return int The converted value in internal units
     */
    const inline int ConvertToInternalUnits(const float &val) { return std::round(val * 1e3); }

    /**
     * @brief Round an integer to the next multiple of a specific base
     *
     * Examples:
     *      roundToMultiple(13, 10) = 10
     *      roundToMultiple(14, 4) = 16
     *
     * @param value The value to be rounded
     * @param base Rounding base such that the result is a multiple of the base
     * @return int The rounded value
     */
    int RoundToMultiple(const int &value, const int &base);

    /**
     * @brief Extract RepeatedTrialAgentData protobuf message
     *
     * @param vec All AgentData objects for all repeated trials
     * @return RTAgentDataProtoMsg protobuf message
     */
    RTAgentDataProtoMsg ExtractRepeatedTrialAgentDataMsg(const RepeatedTrials<std::vector<AgentData>> &vec);

    /**
     * @brief Extract RepeatedTrialStats protobuf message
     *
     * @param arr Array of solver values in the order of local, social, and informed for all trials
     * @return RTStatsProtoMsg protobuf message
     */
    RTStatsProtoMsg ExtractRepeatedTrialStatsMsg(const std::array<RepeatedTrials<Stats>, 3> &arr);

    /**
     * @brief Get all of the SimPacket structs across different simulation parameters
     *
     * @return std::vector<SimPacket> Vector of SimPacket structs
     */
    std::vector<SimPacket> GetAllSimPackets();

    std::unordered_map<unsigned int, std::unordered_map<unsigned int, SimPacket>> sim_packets_; ///< Unordered map storing SimPacket structs
};

#endif