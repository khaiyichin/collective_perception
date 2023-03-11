#ifndef BENCHMARKING_LOOP_FUNCTIONS_HPP
#define BENCHMARKING_LOOP_FUNCTIONS_HPP

#include <vector>
#include <random>
// #include <unordered_map>
// #include <ctime>
// #include <cmath>
// #include <filesystem>
// #include <iomanip>
// #include <sstream>
// #include <algorithm>
// #include <numeric>
#include <nlohmann/json.hpp>

// Buzz and ARGoS headers
// #include <buzz/argos/buzz_loop_functions.h>
// #include <buzz/buzzvm.h>
// #include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/plugins/simulator/entities/rab_equipped_entity.h>
#include <argos3/plugins/simulator/entities/box_entity.h>

// Local headers
#include "util.hpp"
// #include "arena.hpp"
// #include "brain.hpp"
// #include "simulation_set.hpp"
// #include "simulation_stats_set.hpp"
// #include "simulation_agent_data_set.hpp"
// #include "dac_plugin.hpp"
// #include "robot_disability.hpp"
#include "benchmark_crosscombe_2017.hpp"

using namespace argos;
using namespace nlohmann;

/**
 * @brief Class to implement loop functions for the collective perception simulation
 *
 */
class BenchmarkingLoopFunctions : public CBuzzLoopFunctions
{
public:
    /**
     * @brief Initialize loop functions
     *
     * @param t_tree Pointer to the XML config node
     */
    virtual void Init(TConfigurationNode &t_tree);

    /**
     * @brief Reset loop functions (triggered by simulation reset)
     *
     */
    inline void Reset() { SetupExperiment(); }

    /**
     * @brief Execute post step activities
     *
     */
    virtual void PostStep();

    /**
     * @brief Execute post experiment activities
     *
     */
    virtual void PostExperiment();

    /**
     * @brief Check if experiment is over
     *
     * @return true
     * @return false
     */
    inline bool IsExperimentFinished() { return finished_; }

private:
    /**
     * @brief Compute the swarm statistics
     *
     */
    void ComputeStats();

    /**
     * @brief Setup experiment
     *
     */
    void SetupExperiment();

    /**
     * @brief Create a new data packet objects
     *
     */
    void CreateNewPacket();

    /**
     * @brief Write data to disk
     *
     */
    void SaveData();

    std::vector<int> SampleRobotIdsWithoutReplacement();

    void InitializeBenchmarkAlgorithm(TConfigurationNode &t_tree);

    bool finished_ = false; ///< Flag to indicate whether all simulation parameters have been executed

    bool output_datetime_; ///< Flag to enable datetime in output data filename

    int trial_counter_ = 0; ///< Counter to keep track of trials

    int id_base_num_; ///< Starting ID for robots

    std::vector<std::pair<double, double>> paired_parameter_ranges_;

    std::vector<std::pair<double, double>>::iterator curr_paired_parameter_range_itr_;

    std::string algorithm_str_id_; ///< String identifier for desired benchmark algorithm

    std::string verbose_level_; ///< Output verbosity level

    // std::string id_prefix_; ///< Prefix in the robot IDs

    std::string output_folder_; ///< Folder to output data to

    std::string output_filename_; ///< Filename of output data
    
    std::shared_ptr<BenchmarkAlgorithmBase> benchmark_algo_ptr_; ///< Pointer to benchmark algorithm object

    std::shared_ptr<BenchmarkDataBase> benchmark_data_ptr_; ///< Pointer to benchmark data
};

#endif