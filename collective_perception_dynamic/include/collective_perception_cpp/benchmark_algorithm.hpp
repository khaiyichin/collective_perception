#ifndef BENCHMARK_ALGORITHM_HPP
#define BENCHMARK_ALGORITHM_HPP

#include <vector>
#include <string>
#include <random>
#include <functional>
#include <filesystem>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <buzz/buzzvm.h>
#include <argos3/core/utility/configuration/argos_configuration.h>

// Local includes
#include "json.hpp"
#include "util.hpp"
#include "data_common.hpp"

using namespace argos;
using namespace nlohmann;
using BuzzForeachVMFunc = std::function<void(CBuzzLoopFunctions::COperation &)>; // type for the CBuzzLoopFunctions::BuzzForeachVM function

/**
 * @brief Base class for the BenchmarkData struct to provide common variables
 *
 */
struct BenchmarkDataBase
{
    /**
     * @brief Construct a new BenchmarkDataBase object
     *
     */
    BenchmarkDataBase() {}

    /**
     * @brief Construct a new BenchmarkDataBase object
     *
     * @param sim_type Identifying keyword for the simulation type (specific benchmark algorithm)
     */
    BenchmarkDataBase(const std::string &sim_type) : simulation_type(sim_type) {}

    std::string simulation_type; ///< Keyword identifying the desired benchmark algorithm to use

    unsigned int num_agents; ///< Number of robots in the simulation

    unsigned int num_trials; ///< Number of trials per experiment (fixed paired parameters)

    unsigned int num_steps; ///< Number of steps per trial

    float comms_range; ///< Communication range of robots in m

    float density; ///< Swarm density

    float speed; ///< Robot maximum linear speed in cm/s (Buzz code uses cm/s)

    std::string output_folder; ///< High-level output folder name

    std::string output_filename; ///< Filename prefix for output JSON files

    std::vector<double> tfr_range; ///< Range of target fill ratios

    std::vector<double> benchmark_param_range; ///< Range for parameter specific to the benchmark algorithm

    std::vector<ordered_json> json_data; ///< Vector of JSON objects
};

/**
 * @brief Base class for the BenchmarkAlgorithm class
 *
 */
class BenchmarkAlgorithmBase
{
public:
    /**
     * @brief Construct a new BenchmarkAlgorithmBase object
     *
     */
    BenchmarkAlgorithmBase() {}

    /**
     * @brief Construct a new BenchmarkAlgorithmBase object
     *
     * @param buzz_foreach_vm_func CBuzzLoopFunctions::BuzzForeachVM functor
     * @param t_tree XML node tree with `algorithm` as the root node
     * @param robot_id_vec Vector of all robot IDs
     */
    BenchmarkAlgorithmBase(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                           TConfigurationNode &t_tree,
                           const std::vector<std::string> robot_id_vec)
        : buzz_foreach_vm_func_(buzz_foreach_vm_func), robot_id_vec_(robot_id_vec) {}

    /**
     * @brief Destroy the BenchmarkAlgorithmBase object
     *
     */
    virtual ~BenchmarkAlgorithmBase() {}

    /**
     * @brief Initialize the algorithm
     *
     */
    virtual void Init() = 0;

    /**
     * @brief Execute post Step() (provided in argos::CSimulator) operations
     *
     */
    virtual void PostStep() = 0;

    /**
     * @brief Execute post experiment operations
     *
     * @param final_experiment Flag to indicate whether this has been the last experiment
     */
    virtual void PostExperiment(const bool &final_experiment = false) = 0;

    /**
     * @brief Setup the experiment
     *
     * @param trial_ind Current trial index
     * @param curr_paired_parameters Current pair of parameters to simulate
     */
    virtual void SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters) = 0;

    /**
     * @brief Initialize JSON object
     *
     */
    virtual void InitializeJson() = 0;

    /**
     * @brief Save the JSON data
     *
     * @param foldername_prefix Prefix for the high-level folder
     */
    virtual void SaveData(const std::string &foldername_prefix = "") = 0;

    /**
     * @brief Get the benchmark data
     *
     * @return BenchmarkDataBase& Benchmark data struct
     */
    virtual BenchmarkDataBase &GetData() = 0;

    /**
     * @brief Get the range of the benchmark-specific parameter
     *
     * @return std::vector<double> Range of the parameter
     */
    virtual std::vector<double> GetParameterRange() = 0;

    /**
     * @brief Get the benchmark-specific parameter keyword
     * This keyword should have been defined as a macro in the benchmark-specific class header
     *
     * @return std::string Parameter keyword
     */
    virtual std::string GetParameterKeyword() = 0;

    /**
     * @brief Get the abbreviated benchmark-specific parameter keyword
     * This abbreviated keyword should have been defined as a macro in the benchmark-specific class header
     *
     * @return std::string Abbreviated parameter keyword
     */
    virtual std::string GetParameterKeywordAbbr() = 0;

protected:
    ordered_json curr_json_; //< Current JSON data object

    int curr_trial_ind_; ///< Current trial index

    std::pair<double, double> curr_paired_parameters_ = {-1, -1}; ///< Current paired parameters

    BuzzForeachVMFunc buzz_foreach_vm_func_; ///< BuzzForeachVM functor

    std::vector<std::string> robot_id_vec_; ///< Vector of robot IDs (provides full list of robot IDs)
};

/**
 * @brief Template for creating the BenchmarkAlgorithm class specific to the desired benchmark
 *
 * @tparam T BenchmarkData type
 */
template <typename T>
class BenchmarkAlgorithmTemplate : public BenchmarkAlgorithmBase
{
public:
    /**
     * @brief Construct a new BenchmarkAlgorithmTemplate object
     *
     */
    BenchmarkAlgorithmTemplate() {}

    /**
     * @brief Construct a new BenchmarkAlgorithmTemplate object
     *
     * @param buzz_foreach_vm_func CBuzzLoopFunctions::BuzzForeachVM functor
     * @param t_tree XML node tree with `algorithm` as the root node
     * @param robot_id_vec Vector of all robot IDs
     */
    BenchmarkAlgorithmTemplate(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                               TConfigurationNode &t_tree,
                               const std::vector<std::string> &robot_id_vec)
        : BenchmarkAlgorithmBase(buzz_foreach_vm_func, t_tree, robot_id_vec) {}

    /**
     * @brief Get the benchmark data
     *
     * @return T& Benchmark data (a subclass of BenchmarkDataBase)
     */
    T &GetData() { return data_; }

protected:
    /**
     * @brief Draw a sample of robot IDs without replacement
     *
     * @param num_robots_to_sample Number of robots to sample
     * @return std::vector<std::string> Drawn robot IDs
     */
    std::vector<std::string> SampleRobotIdsWithoutReplacement(const unsigned int &num_robots_to_sample)
    {
        // Sample random robot IDs (without replacement)
        std::vector<std::string> sampled_robot_ids;

        std::sample(robot_id_vec_.begin(),
                    robot_id_vec_.end(),
                    std::back_inserter(sampled_robot_ids),
                    num_robots_to_sample,
                    std::mt19937{std::random_device{}()});

        return sampled_robot_ids;
    }

    T data_; ///< Benchmark data (subclass of BenchmarkDataBase)
};

#endif