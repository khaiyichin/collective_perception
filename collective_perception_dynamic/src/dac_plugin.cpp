#include "dac_plugin.hpp"

DACPlugin::DACPlugin(const unsigned int &num_bins,
                     const unsigned int &num_robots,
                     const float &density,
                     const float &arena_area,
                     const float &comms_range,
                     const float &speed,
                     const std::string &csv_path)
    : num_bins_(num_bins),
      total_num_robots_(num_robots),
      current_healthy_robots_(num_robots),
      current_disabled_robots_(0),
      swarm_density_(density),
      arena_area_(arena_area),
      robot_comms_range_(comms_range),
      robot_speed_(speed),
      csv_path_(csv_path)
{
    // Clear the CSV file contents
    std::ofstream csv_file(csv_path, std::ofstream::out | std::ofstream::trunc);
    csv_file.close();

    // Reserve space for informed estimates (to prevent recreating during runtime)
    decisions_.reserve(num_bins_);
}

void DACPlugin::UpdateCurrentExperimentParams(const float &tfr, const float &sp)
{
    current_target_fill_ratio_ = tfr;
    current_sensor_probability_ = sp;
    current_trial_number_ = 0;
}

void DACPlugin::ComputeFractionOfCorrectDecisions(const std::shared_ptr<std::unordered_map<std::string, Brain>> &ptr)
{
    // Go through each brain element in the map
    decisions_ = std::vector<unsigned int>(num_bins_, 0);
    std::unordered_map<DisabilityType, unsigned int> disabled_robot_counter = {{DisabilityType::motion, 0}, {DisabilityType::comms, 0}, {DisabilityType::sense, 0}};

    for (auto &kv : *ptr)
    {
        // Check if the brain is active (robot is not disabled)
        std::vector<DisabilityType> disability_type_vec = kv.second.GetDisabilityTypes();

        if (disability_type_vec.size() == 0) // no disabilities
        {
            // Increment the count for the ith bin
            unsigned int index = ConvertInformedEstimateToDecision(kv.second.GetInformedValuePair().x);
            ++decisions_[index];
        }
        else
        {
            // Update counter with the robot disabilities
            std::vector<DisabilityType> dt_vec = kv.second.GetDisabilityTypes();

            for (const auto &dt : dt_vec)
            {
                ++disabled_robot_counter[dt];
            }
        }
    }

    // Obtain correct decision
    unsigned int tfr_decision = IdentifyCorrectDecision();

    // Identify the active and disabled robots
    current_healthy_robots_ = std::reduce(decisions_.begin(), decisions_.end());
    current_disabled_robots_ = disabled_robot_counter;

    // Compute fraction of correct decisions
    current_fraction_correct_decisions_ = static_cast<float>(decisions_[tfr_decision]) / static_cast<float>(current_healthy_robots_);
}

unsigned int DACPlugin::ConvertInformedEstimateToDecision(const float &est)
{
    float scaled_est = est == 1.0 ? (est - 1e-3) * scaling_factor_ : est * scaling_factor_;
    float divisor = 1.0 / num_bins_; // this assumes the bins to be equally sized, which is always the case
    return static_cast<int>(scaled_est / divisor) / scaling_factor_;
}

void DACPlugin::WriteCurrentTrialStats(const std::string &current_time_str, const bool &initialize, const unsigned int &sim_time_sec)
{
    std::string line_to_write = current_time_str;

    if (initialize)
    {
        line_to_write += ",trialindex," + std::to_string(current_trial_number_++);
    }
    else
    {
        // Write regular stat lines
        line_to_write += ",trialstats";
        line_to_write += "\n,simseconds," + std::to_string(sim_time_sec);
        line_to_write += "\n,active," + std::to_string(current_healthy_robots_);
        line_to_write += "\n," + GetBuzzDisabilityKeyword(DisabilityType::motion) + "," + std::to_string(current_disabled_robots_[DisabilityType::motion]);
        line_to_write += "\n," + GetBuzzDisabilityKeyword(DisabilityType::comms) + "," + std::to_string(current_disabled_robots_[DisabilityType::comms]);
        line_to_write += "\n," + GetBuzzDisabilityKeyword(DisabilityType::sense) + "," + std::to_string(current_disabled_robots_[DisabilityType::sense]);
        line_to_write += "\n,fractioncorrectdecisions," + std::to_string(current_fraction_correct_decisions_);
    }

    WriteToCSV(line_to_write + "\n");
}

void DACPlugin::WriteCurrentExperimentStats(const std::string &current_time_str, const bool &finalize)
{
    std::string line_to_write = current_time_str;

    if (finalize)
    {
        // Write final line
        line_to_write += ",experimentcomplete";
    }
    else
    {
        line_to_write += ",experimentstats";
        line_to_write += "\n,range," + std::to_string(robot_comms_range_);
        line_to_write += "\n,speed," + std::to_string(robot_speed_);
        line_to_write += "\n,density," + std::to_string(swarm_density_) + ",area," + std::to_string(arena_area_);
        line_to_write += "\n,robots," + std::to_string(total_num_robots_);
        line_to_write += "\n,fillratio," + std::to_string(current_target_fill_ratio_);
        line_to_write += "\n,sensorprob," + std::to_string(current_sensor_probability_);
    }

    WriteToCSV(line_to_write + "\n");
}

void DACPlugin::WriteToCSV(const std::string &str)
{
    std::ofstream csv_file(csv_path_, std::ofstream::app);
    csv_file << str;
    csv_file.close();
}