#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

#include <google/protobuf/stubs/common.h>

/**
 * @brief Alias for storing multiple elements in a single experiment
 *
 * @tparam T Any type to be stored
 */
template <class T>
using RepeatedExperimentData = std::vector<T>;

/**
 * @brief Struct to store a single agent's data in one experiment
 *
 */
struct AgentData
{
    AgentData() {}

    std::vector<unsigned int> tile_occurrences;

    std::vector<unsigned int> observations;
};

/**
 * @brief Struct to store statistics across all agents
 *
 */
struct Stats
{
    Stats() {}

    std::vector<float> x_sample_mean; ///< mean estimate across all agents

    std::vector<float> confidence_sample_mean; ///< mean confidence across all agents

    std::vector<float> x_sample_std; ///< 1 sigma estimate across all agents

    std::vector<float> confidence_sample_std; ///< 1 sigma confidence across all agents
};

/**
 * @brief Struct to store simulation data and statistics
 *
 * A SimPacket stores data from repeated experiments with the same:
 *  - number of agents,
 *  - number of steps,
 *  - communication range,
 *  - swarm density,
 *  - target fill ratio, and
 *  - sensor probability.
 *
 */
struct SimPacket
{
    SimPacket() {}

    float comms_range;

    float target_fill_ratio;

    float b_prob;

    float w_prob;

    unsigned int num_agents;

    unsigned int num_trials;

    unsigned int num_steps;

    float density;

    std::string sim_type = "dynamic";

    RepeatedExperimentData<Stats> local_values_vec; ///< Local values for repeated experiments

    RepeatedExperimentData<Stats> social_values_vec; ///< Social values for repeated experiments

    RepeatedExperimentData<Stats> informed_values_vec; ///< Social values for repeated experiments

    RepeatedExperimentData<AgentData> agent_data_vec; ///< Agent data for repeated experiments
};

template <class T>
void WriteProtoToDisk(const T &proto_obj, const std::string &path)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION; // verify version of library linked is compatible with the version of header compiled

    std::ofstream pb_stream(path, std::ios::trunc | std::ios::binary);

    if (!proto_obj.SerializeToOstream(&pb_stream))
    {
        throw std::runtime_error("Failed to serialize!");
    }
}

template <class T>
void LoadProtoFromDisk(T &proto_obj, const std::string &path)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION; // verify version of library linked is compatible with the version of header compiled

    std::ifstream pb_stream(path, std::ios::binary);

    if (!proto_obj.ParseFromIstream(&pb_stream))
    {
        throw std::runtime_error("Failed to read!");
    }
}

#endif