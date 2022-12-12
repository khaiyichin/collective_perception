#ifndef DAC_PLUGIN_HPP
#define DAC_PLUGIN_HPP

#include <string>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <numeric>

// Local includes
#include "brain.hpp"

class DACPlugin
{
public:
    /**
     * @brief Construct a new DACPlugin object
     *
     */
    DACPlugin(){};

    /**
     * @brief Construct a new DACPlugin object
     *
     * @param num_bins Number of bins to partition the fill ratio into
     * @param num_robots Number of robots in the simulation
     * @param swarm_density Robot swarm density
     * @param arena_area Constrained arena area (that was used to calculate swarm density)
     * @param comms_range Robot communication range
     * @param robot_speed Robot movement speed
     * @param csv_path CSV file path to read and write to
     */
    DACPlugin(const unsigned int &num_bins,
              const unsigned int &num_robots,
              const float &swarm_density,
              const float &arena_area,
              const float &comms_range,
              const float &robot_speed,
              const std::string &csv_path);

    /**
     * @brief Update values based on the current experiment parameters
     *
     * @param tfr Environment target fill ratio for the current experiment
     * @param sp Robot sensor probability for the current experiment
     */
    void UpdateCurrentExperimentParams(const float &tfr, const float &sp);

    /**
     * @brief Compute the fraction of correct decisions by the swarm
     *
     * @param ptr Shared pointer to the unordered map of robot ID and Brain objects
     */
    void ComputeFractionOfCorrectDecisions(const std::shared_ptr<std::unordered_map<std::string, Brain>> &ptr);

    /**
     * @brief Write current trial statistics to the CSV file
     *
     * @param current_time_str Current datetime in string
     * @param initialize Flag to indicate whether to write initial trial statistics
     * @param sim_time_sec Current simulation time in seconds
     */
    void WriteCurrentTrialStats(const std::string &current_time_str, const bool &initialize, const unsigned int &sim_time_sec);

    /**
     * @brief Write current experiment statistics to the CSV file
     *
     * @param current_time_str Current datetime in string
     * @param finalize Flag to indicate whether to write final experiment statistics
     */
    void WriteCurrentExperimentStats(const std::string &current_time_str, const bool &finalize);

private:
    /**
     * @brief Convert estimates to bin selections
     * 
     * @param est Informed estimate value
     * @return unsigned int Bin number that corresponds to a decision
     */
    unsigned int ConvertInformedEstimateToDecision(const float &est);

    /**
     * @brief Identify the correct decision based on the environment fill ratio
     * 
     * @return unsigned int Bin number that corresponds to the correct decision
     */
    inline unsigned int IdentifyCorrectDecision() { return ConvertInformedEstimateToDecision(current_target_fill_ratio_); }

    /**
     * @brief Write string to CSV file
     * 
     * @param str String to write
     */
    void WriteToCSV(const std::string &str);

    unsigned int num_bins_;

    unsigned int current_active_robots_;

    unsigned int current_disabled_robots_;

    unsigned int current_trial_number_;

    unsigned int scaling_factor_ = 1e3;

    float swarm_density_;

    float arena_area_;

    float robot_comms_range_;

    float robot_speed_;

    float current_sensor_probability_;

    float current_target_fill_ratio_;

    float current_fraction_correct_decisions_;

    std::string csv_path_;

    std::string trial_string_;

    std::string experiment_string_;

    std::vector<unsigned int> decisions_;
};

#endif