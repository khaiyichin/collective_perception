#include "benchmark_valentini_2016.hpp"

void ProcessRobotOpinions::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    if (!initialized)
    {
        // Set robot sensor probabilities
        float prob;

        if (b_prob < 0.0 && w_prob < 0.0) // probabilities according to a distribution
        {
            prob = GenerateRandomSensorProbability(b_prob, generator);
            BuzzPut(t_vm, "b_prob", prob);

            prob = GenerateRandomSensorProbability(w_prob, generator);
            BuzzPut(t_vm, "w_prob", prob);
        }
        else // fixed probabilities
        {
            BuzzPut(t_vm, "b_prob", static_cast<float>(b_prob));
            BuzzPut(t_vm, "w_prob", static_cast<float>(w_prob));
        }

        // Set robot speed
        BuzzPut(t_vm, "spd", static_cast<float>(spd));

        // Set initial state as exploration state
        BuzzPut(t_vm, "current_state", static_cast<int>(RobotState::exploration));

        // Set randomized current opinion
        BuzzPut(t_vm, "current_opinion", initial_opinion_dist(generator));

        // Set exploration duration
        BuzzPut(t_vm, "exp_duration", static_cast<int>(exploration_duration_dist(generator)));

        // Set voter model flag
        BuzzPut(t_vm, "use_voter_model", voter_model);

        // Check to see if robots are all initialized
        std::pair<std::set<std::string>::iterator, bool> insert_receipt = initialized_robot_ids.insert(str_robot_id);

        if (insert_receipt.second == false)
        {
            THROW_ARGOSEXCEPTION("The same robot has been initialized!");
        }

        if (initialized_robot_ids.size() == num_robots)
        {
            initialized = true;
        }

        // Log the robot's initial values
        int state = buzzobj_getint(BuzzGet(t_vm, "current_state"));

        BuzzTableOpen(t_vm, "opinion_qualities");
        buzzobj_t buzz_table_obj = BuzzGet(t_vm, "opinion_qualities");

        double qual_white = buzzobj_getfloat(BuzzTableGet(t_vm, static_cast<int>(RobotOpinion::white)));
        double qual_black = buzzobj_getfloat(BuzzTableGet(t_vm, static_cast<int>(RobotOpinion::black)));

        BuzzTableClose(t_vm);

        int opinion = buzzobj_getint(BuzzGet(t_vm, "current_opinion"));

        (*id_data_str_map_ptr)[str_robot_id.c_str()].push_back(ConvertStateQualitiesOpinionToString(state, qual_white, qual_black, opinion));
    }
    else
    {
        // Get current state
        current_robot_data.current_state = buzzobj_getint(BuzzGet(t_vm, "current_state"));

        // Get duration tracker
        current_robot_data.duration_tracker = buzzobj_getint(BuzzGet(t_vm, "duration_tracker"));

        // Get current opinion
        current_robot_data.current_opinion = buzzobj_getint(BuzzGet(t_vm, "current_opinion"));

        // Get opinion qualities
        BuzzTableOpen(t_vm, "opinion_qualities");
        buzzobj_t buzz_table_obj = BuzzGet(t_vm, "opinion_qualities");

        current_robot_data.opinion_qualities[static_cast<int>(RobotOpinion::white)] =
            buzzobj_getfloat(BuzzTableGet(t_vm, static_cast<int>(RobotOpinion::white)));
        current_robot_data.opinion_qualities[static_cast<int>(RobotOpinion::black)] =
            buzzobj_getfloat(BuzzTableGet(t_vm, static_cast<int>(RobotOpinion::black)));

        BuzzTableClose(t_vm);

        // Check whether there are state changes
        if (current_robot_data.duration_tracker == 0)
        {
            if (static_cast<RobotState>(current_robot_data.current_state) == RobotState::dissemination)
            {
                // Update dissemination duration
                dis_mean_rate_ticks =
                    std::exponential_distribution<double>::param_type(1.0 / (current_robot_data.opinion_qualities[current_robot_data.current_opinion] *
                                                                             dis_mean_dur_fac_ticks));

                current_robot_data.current_dis_duration = static_cast<int>(
                    std::round(dissemination_duration_dist(generator, dis_mean_rate_ticks)));

                BuzzPut(t_vm,
                        "dis_duration",
                        current_robot_data.current_dis_duration == 0 ? 1 : current_robot_data.current_dis_duration); // to ensure duration is at least 1 tick
            }
            else if (static_cast<RobotState>(current_robot_data.current_state) == RobotState::exploration)
            {
                // Update exploration duration
                current_robot_data.current_exp_duration = static_cast<int>(
                    std::round(exploration_duration_dist(generator)));

                BuzzPut(t_vm,
                        "exp_duration",
                        current_robot_data.current_exp_duration == 0 ? 1 : current_robot_data.current_exp_duration);
            }
            else
            {
                THROW_ARGOSEXCEPTION("Unknown current state!");
            }
        }
        else
        {
            // do nothing, proceed as usual
        }

        // Log the robot data
        (*id_data_str_map_ptr)[str_robot_id.c_str()].push_back(
            ConvertStateQualitiesOpinionToString(current_robot_data.current_state,
                                                 current_robot_data.opinion_qualities[static_cast<int>(RobotOpinion::white)],
                                                 current_robot_data.opinion_qualities[static_cast<int>(RobotOpinion::black)],
                                                 current_robot_data.current_opinion));
    }
}

BenchmarkValentini2016::BenchmarkValentini2016(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                                               TConfigurationNode &t_tree,
                                               const std::vector<std::string> &robot_id_vec)
    : BenchmarkAlgorithmTemplate<BenchmarkDataValentini2016>(buzz_foreach_vm_func, t_tree, robot_id_vec)
{
    GetNodeAttribute(GetNode(t_tree, "exploration_mean_duration"), "value", data_.exp_mean_duration);
    GetNodeAttribute(GetNode(t_tree, "dissemination_mean_duration_factor"), "value", data_.dis_mean_duration_factor);
    GetNodeAttribute(GetNode(t_tree, "voter_model"), "bool", data_.voter_model);

    // Grab sensor probability range (inner group parameter)
    double min, max;
    int steps;

    TConfigurationNode &sensor_probability_range_node = GetNode(t_tree, VALENTINI_2016_PARAM + "_range");

    GetNodeAttribute(sensor_probability_range_node, "min", min);
    GetNodeAttribute(sensor_probability_range_node, "max", max);
    GetNodeAttribute(sensor_probability_range_node, "steps", steps);

    if (steps < 0.0)
    {
        data_.sensor_probability_range.push_back(EncodeSensorDistParams(steps, min, max));
    }
    else
    {
        data_.sensor_probability_range = GenerateLinspace(min, max, steps);
    }
}

void BenchmarkValentini2016::SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters)
{
    // Assign values to member variables
    curr_trial_ind_ = trial_ind;
    curr_paired_parameters_ = curr_paired_parameters;

    // Setup functors
    id_data_str_map_ptr_ = std::make_shared<RobotIdDataStrMap>();

    argos::TConfigurationNode config_root_node = CSimulator::GetInstance().GetConfigurationRoot();
    double ticks_per_sec;
    GetNodeAttribute(GetNode(GetNode(config_root_node, "framework"), "experiment"), "ticks_per_second", ticks_per_sec);

    process_robot_opinions_functor_ = ProcessRobotOpinions(data_.num_agents,
                                                           data_.speed,
                                                           curr_paired_parameters_.second,
                                                           curr_paired_parameters_.second,
                                                           data_.exp_mean_duration * ticks_per_sec,
                                                           data_.dis_mean_duration_factor * ticks_per_sec,
                                                           data_.voter_model,
                                                           id_data_str_map_ptr_);

    // Initialize each robot
    buzz_foreach_vm_func_(process_robot_opinions_functor_);

    // Initialize JSON file
    InitializeJson();
}

void BenchmarkValentini2016::PostStep()
{
    // Iterate through robots to process their opinions
    buzz_foreach_vm_func_(process_robot_opinions_functor_);
}

void BenchmarkValentini2016::PostExperiment(const bool &final_experiment /*=false*/)
{
    // Store new trial result
    std::string key = "data_str";

    // Store data into current json
    std::vector<std::vector<std::string>> vec;
    vec.reserve(data_.num_agents);

    for (auto itr = id_data_str_map_ptr_->begin(); itr != id_data_str_map_ptr_->end(); ++itr)
    {
        vec.push_back(itr->second); // store the values from the map, which are vectors of strings
    }

    curr_json_[key] = vec;
    data_.json_data.push_back(curr_json_);
}

void BenchmarkValentini2016::InitializeJson()
{
    curr_json_ = ordered_json{};
    curr_json_["sim_type"] = data_.simulation_type;
    curr_json_["num_agents"] = data_.num_agents;
    curr_json_["num_trials"] = data_.num_trials;
    curr_json_["num_steps"] = data_.num_steps;
    curr_json_["comms_range"] = data_.comms_range;
    curr_json_["speed"] = data_.speed;
    curr_json_["density"] = data_.density;
    curr_json_["tfr"] = curr_paired_parameters_.first;
    curr_json_[VALENTINI_2016_PARAM_ABBR] = curr_paired_parameters_.second;
    curr_json_["trial_ind"] = curr_trial_ind_;
    curr_json_["exp_mean_dur"] = data_.exp_mean_duration;
    curr_json_["dis_mean_dur_factor"] = data_.dis_mean_duration_factor;
    curr_json_["voter_model"] = data_.voter_model;
}

void BenchmarkValentini2016::SaveData(const std::string &foldername_prefix /*=""*/)
{
    // Create high-level folder
    std::string folder = foldername_prefix +
                         "tfr" +
                         Round1000DoubleToStr(data_.tfr_range.front()) +
                         "-" +
                         Round1000DoubleToStr(
                             data_.tfr_range.size() > 1 ? (data_.tfr_range.back() - data_.tfr_range.front()) /
                                                              (data_.tfr_range.size() - 1)
                                                        : 0) +
                         "-" +
                         Round1000DoubleToStr(data_.tfr_range.back()) +
                         "_" +
                         VALENTINI_2016_PARAM_ABBR +
                         Round1000DoubleToStr(data_.sensor_probability_range.front()) +
                         "-" +
                         Round1000DoubleToStr(
                             data_.sensor_probability_range.size() > 1 ? (data_.sensor_probability_range.back() - data_.sensor_probability_range.front()) /
                                                                             (data_.sensor_probability_range.size() - 1)
                                                                       : 0) +
                         "-" +
                         Round1000DoubleToStr(data_.sensor_probability_range.back()) +
                         "_" +
                         "exp" + Round1000DoubleToStr(data_.exp_mean_duration) +
                         "_" +
                         "dis" + Round1000DoubleToStr(data_.dis_mean_duration_factor) +
                         "_" +
                         "vot" + std::to_string(data_.voter_model);

    std::filesystem::create_directory(folder);

    // Write JSON files into folder
    std::string filepath_prefix = folder + "/";

    // Strip extension from filename
    std::pair<std::string, std::string> name_ext_pair_output;
    std::stringstream stream_output(data_.output_filename);

    getline(stream_output, name_ext_pair_output.first, '.');  // filename
    getline(stream_output, name_ext_pair_output.second, '.'); // extension

    if (name_ext_pair_output.second.empty()) // in case no extension was provided
    {
        name_ext_pair_output.second = "json";
    }

    // Create individual JSON files
    for (auto itr = data_.json_data.begin(); itr != data_.json_data.end(); ++itr)
    {
        std::string filename =
            "tfr" + Round1000DoubleToStr((*itr)["tfr"].get<double>()) + "_" +
            VALENTINI_2016_PARAM_ABBR + Round1000DoubleToStr((*itr)[VALENTINI_2016_PARAM_ABBR]) + "_" +
            "t" + std::to_string((*itr)["trial_ind"].get<int>());

        filename = filepath_prefix + name_ext_pair_output.first + "_" + filename + "." + name_ext_pair_output.second;

        // Export to single JSON file
        std::ofstream outfile(filename);

        outfile << std::setw(4) << (*itr) << std::endl; // write pretty JSON
    }
}
