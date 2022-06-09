#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

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
 * @brief Struct to store statistics
 *
 */
struct Stats
{
    Stats() {}

    std::vector<float> x_sample_mean;

    std::vector<float> confidence_sample_mean;

    std::vector<float> x_sample_std;

    std::vector<float> confidence_sample_std;
};

/**
 * @brief Struct to store simulation data and statistics
 *
 * An SimPacket stores data from repeated experiments with the same:
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

    unsigned int num_experiments;

    unsigned int num_steps;

    float density;

    std::string sim_type = "dynamic";

    RepeatedExperimentData<Stats> local_vals_vec; ///< Local values for repeated experiments

    RepeatedExperimentData<Stats> social_values_vec; ///< Social values for repeated experiments

    RepeatedExperimentData<Stats> informed_values_vec; ///< Social values for repeated experiments

    RepeatedExperimentData<AgentData> agent_data_vec; ///< Agent data for repeated experiments
};

#endif