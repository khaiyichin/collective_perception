#ifndef SIMULATION_STATS_SET_HPP
#define SIMULATION_STATS_SET_HPP

#include <iterator>
#include <algorithm>

#include "data_common.hpp"
#include "simulation_set.hpp"
#include "simulation_set.pb.h"

using RTStatsProtoMsg = collective_perception_cpp::proto::RepeatedTrialStats;
using StatsProtoMsg = collective_perception_cpp::proto::RepeatedTrialStats::Stats;
using SimSetMsg = collective_perception_cpp::proto::SimulationSet;
using PacketMsg = collective_perception_cpp::proto::Packet;

/**
 * @brief Class to store all statistics from all the experiments
 *
 * A SimulationStatsSet object stores sets of statistics from simulations (i.e., `StatsPacket`s) with the same:
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
class SimulationStatsSet : public SimulationSet
{
public:
    SimulationStatsSet() {}

    /**
     * @brief Construct a new SimulationStatsSet object from SimulationStatsSet protobuf message
     *
     * @param sim_stats_set_msg
     */
    SimulationStatsSet(collective_perception_cpp::proto::SimulationStatsSet &sim_stats_set_msg);

    /**
     * @brief Serialize into SimulationStatsSet protobuf message
     *
     * @param sim_stats_set_msg
     */
    void Serialize(collective_perception_cpp::proto::SimulationStatsSet &sim_stats_set_msg);

    inline void InsertStatsPacket(const StatsPacket &packet) { SimulationSet::InsertPacket(packet, packets_); }

    inline StatsPacket GetStatsPacket(const float &tfr, const float &sp) { return SimulationSet::GetPacket(tfr, sp, packets_); }

private:
    /**
     * @brief Extract RepeatedTrialStats protobuf message
     *
     * @param arr Array of solver values in the order of local, social, and informed for all trials
     * @return RTStatsProtoMsg protobuf message
     */
    RTStatsProtoMsg ExtractRepeatedTrialStatsMsg(const std::array<RepeatedTrials<Stats>, 3> &arr);

    PacketDict<StatsPacket> packets_; ///< Unordered map storing T structs
};

#endif