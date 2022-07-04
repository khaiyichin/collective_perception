#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <fstream>

#include <google/protobuf/stubs/common.h>

/**
 * @brief Alias for storing multiple trial data
 *
 * @tparam T Any type to be stored
 */
template <class T>
using RepeatedTrials = std::vector<T>;

/**
 * @brief Struct to store a single agent's data in one trial (mostly for debugging purposes)
 *
 */
struct AgentData
{
    AgentData() {}

    std::vector<unsigned int> tile_occurrences;

    std::vector<unsigned int> observations;
};

/**
 * @brief Struct to store statistics across all agents per trial
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
 * @brief Struct to serve as parent struct to store simulation data and statistics
 *
 * A Packet stores data from repeated trials with the same:
 *  - number of agents,
 *  - number of steps,
 *  - communication range,
 *  - swarm density,
 *  - robot speed,
 *  - target fill ratio, and
 *  - sensor probability.
 *
 */
struct Packet
{
    Packet() {}

    float comms_range;

    float target_fill_ratio;

    float b_prob;

    float w_prob;

    unsigned int num_agents;

    unsigned int num_trials;

    unsigned int num_steps;

    float density;

    float speed;

    std::string sim_type = "dynamic";
};

struct StatsPacket : Packet
{
    StatsPacket() {}

    /**
     * @brief Construct a new StatsPacket object
     * 
     * @param n Number of trials
     */
    inline StatsPacket(const unsigned int &n)
        : repeated_local_values(n), repeated_social_values(n), repeated_informed_values(n) {}

    RepeatedTrials<Stats> repeated_local_values; ///< Local values for repeated trials

    RepeatedTrials<Stats> repeated_social_values; ///< Social values for repeated trials

    RepeatedTrials<Stats> repeated_informed_values; ///< Social values for repeated trials

    RepeatedTrials<float> sp_mean_values; ///< mean values for random sensor probabilities
};

struct AgentDataPacket : Packet
{
    AgentDataPacket() {}

    /**
     * @brief Construct a new AgentDataPacket object
     * 
     * @param n Number of trials
     * @param m Number of agents
     */
    inline AgentDataPacket(const unsigned int &n, const unsigned int &m)
        : repeated_agent_data_vec(n, std::vector<AgentData>(m)) {}

    RepeatedTrials<std::vector<AgentData>> repeated_agent_data_vec; ///< Agent data vector for repeated trials
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