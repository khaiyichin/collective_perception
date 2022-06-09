#ifndef SIMULATION_DATA_SET_HPP
#define SIMULATION_DATA_SET_HPP

#include "data_common.hpp"
#include "simulation_data_set.pb.h"

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
     * @brief Construct a new SimulationDataSet object from SimulationDataSet protobuf
     *
     * @param exp_data_proto
     */
    SimulationDataSet(collective_perception_cpp::proto::SimulationDataSet &exp_data_proto);

    void Serialize(collective_perception_cpp::proto::SimulationDataSet &exp_data_proto);

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

private:
    /**
     * @brief Convert the provided value into internal units: 0.005 = 5
     *
     * @param val The value in physical units to be converted
     * @return int The converted value in internal units
     */
    int ConvertToInternalUnits(const float &val) { return std::round(val * 1e3); }

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

    unsigned int num_agents_;

    unsigned int num_experiments_;

    unsigned int num_steps_;

    float comms_range_;

    float density_;

    std::vector<float> tfr_range_;

    std::vector<float> sp_range_;

    std::unordered_map<unsigned int, std::unordered_map<unsigned int, SimPacket>> sim_packets_;
};

#endif