#ifndef BENCHMARKING_LOOP_FUNCTIONS_HPP
#define BENCHMARKING_LOOP_FUNCTIONS_HPP

#include <vector>
#include <random>
#include <filesystem>

// Buzz and ARGoS headers
#include <argos3/plugins/simulator/entities/rab_equipped_entity.h>
#include <argos3/plugins/simulator/entities/box_entity.h>

// Local headers
#include "util.hpp"
#include "benchmark_crosscombe_2017.hpp"

using namespace argos;

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
     * @brief Setup experiment
     *
     */
    void SetupExperiment();

    /**
     * @brief Write data to disk
     *
     */
    void SaveData();

    /**
     * @brief Initialize the desired benchmark algorithm
     *
     * @param t_tree XML node tree with `algorithm` as the root node
     */
    void InitializeBenchmarkAlgorithm(TConfigurationNode &t_tree);

    bool finished_ = false; ///< Flag to indicate whether all simulation parameters have been executed

    bool output_datetime_; ///< Flag to enable datetime in output data filename

    int curr_trial_ind_ = 0; ///< Counter to keep track of trials

    std::vector<std::pair<double, double>> paired_parameter_ranges_; ///< Paired parameter ranges to be simulated

    std::vector<std::pair<double, double>>::iterator curr_paired_parameter_range_itr_; ///< Current pair of parameters to be simulated

    std::string algorithm_str_id_; ///< String identifier for desired benchmark algorithm

    std::string verbose_level_; ///< Output verbosity level

    std::string output_folder_; ///< Folder to output data to

    std::string output_filename_; ///< Filename of output data

    std::shared_ptr<BenchmarkAlgorithmBase> benchmark_algo_ptr_; ///< Pointer to benchmark algorithm object
};

#endif