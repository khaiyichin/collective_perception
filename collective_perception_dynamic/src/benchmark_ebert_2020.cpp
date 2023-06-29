#include "benchmark_ebert_2020.hpp"

void ProcessRobotPosteriorProb::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
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

        // Set robot prior parameters
        BuzzPut(t_vm, "alpha", prior_param);
        BuzzPut(t_vm, "beta", prior_param);

        // Set switch to use positive_feedback
        BuzzPut(t_vm, "positive_feedback", static_cast<int>(positive_feedback));

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
    }
    else
    {
        // Read the robot's Beta posterior parameters
        int a = buzzobj_getint(BuzzGet(t_vm, "alpha"));
        int b = buzzobj_getint(BuzzGet(t_vm, "beta"));

        // Evaluate if the robot have enough information to come to a decision
        double p = EvaluateBetaCDF(static_cast<double>(a),
                                   static_cast<double>(b)); // calculate P(X < 0.5), X ~ Beta(a, b)

        // Obtain robot decision state
        int decision_state_int = buzzobj_getint(BuzzGet(t_vm, "decision_state"));

        // Check to see if the current robot needs to make a decision
        if (static_cast<RobotDecision>(decision_state_int) == RobotDecision::undecided)
        {
            if (p > credible_threshold) // majority white
            {
                decision_state_int = static_cast<int>(RobotDecision::white);

                BuzzPut(t_vm, "decision_state", decision_state_int);

                ++num_decided_robots;
            }
            else if (1 - p > credible_threshold) // majority black
            {
                decision_state_int = static_cast<int>(RobotDecision::black);

                BuzzPut(t_vm, "decision_state", decision_state_int);

                ++num_decided_robots;
            }
        }

        // Log the robot's posterior and decision
        (*id_data_str_map_ptr)[str_robot_id.c_str()].push_back(ConvertParamPosteriorDecisionToString(a, b, p, decision_state_int));

        // Flag for termination if all the robots have decided
        if (num_decided_robots == num_robots)
        {
            decided = true;
        }
    }
}

BenchmarkEbert2020::BenchmarkEbert2020(const BuzzForeachVMFunc &buzz_foreach_vm_func,
                                       TConfigurationNode &t_tree,
                                       const std::vector<std::string> &robot_id_vec)
    : BenchmarkAlgorithmTemplate<BenchmarkDataEbert2020>(buzz_foreach_vm_func, t_tree, robot_id_vec)
{
    // Grab benchmark algorithm specific parameters (outer group parameters)
    GetNodeAttribute(GetNode(t_tree, "positive_feedback"), "bool", data_.positive_feedback);
    GetNodeAttribute(GetNode(t_tree, "prior"), "value", data_.prior_param);
    GetNodeAttribute(GetNode(t_tree, "credible_threshold"), "value", data_.credible_threshold);

    // Grab sensor probability range (inner group parameter)
    double min, max;
    int steps;

    TConfigurationNode &sensor_probability_range_node = GetNode(t_tree, EBERT_2020_PARAM + "_range");

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

void BenchmarkEbert2020::SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters)
{
    // Assign values to member variables
    curr_trial_ind_ = trial_ind;
    curr_paired_parameters_ = curr_paired_parameters;

    // Setup functors
    id_data_str_map_ptr_ = std::make_shared<RobotIdDataStrMap>();

    process_robot_posterior_prob_functor_ = ProcessRobotPosteriorProb(data_.num_agents,
                                                                      data_.speed,
                                                                      curr_paired_parameters_.second,
                                                                      curr_paired_parameters_.second,
                                                                      data_.prior_param,
                                                                      data_.credible_threshold,
                                                                      data_.positive_feedback,
                                                                      id_data_str_map_ptr_);

    // Initialize each robot
    buzz_foreach_vm_func_(process_robot_posterior_prob_functor_);

    // Initialize JSON file
    InitializeJson();
}

void BenchmarkEbert2020::PostStep()
{
    // Iterate through robots to process their posterior and decision
    buzz_foreach_vm_func_(process_robot_posterior_prob_functor_);

    // Terminate early if all the robots have decided
    if (process_robot_posterior_prob_functor_.decided)
    {
        curr_json_["collectively_decided"] = true;
        CSimulator::GetInstance().Terminate();
    }
}

void BenchmarkEbert2020::PostExperiment(const bool &final_experiment /*=false*/)
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

void BenchmarkEbert2020::InitializeJson()
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
    curr_json_[EBERT_2020_PARAM_ABBR] = curr_paired_parameters_.second;
    curr_json_["trial_ind"] = curr_trial_ind_;
    curr_json_["prior_param"] = data_.prior_param;
    curr_json_["credible_threshold"] = data_.credible_threshold;
    curr_json_["positive_feedback"] = data_.positive_feedback;
    curr_json_["collectively_decided"] = false;
}

void BenchmarkEbert2020::SaveData(const std::string &foldername_prefix /*=""*/)
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
                         EBERT_2020_PARAM_ABBR +
                         Round1000DoubleToStr(data_.sensor_probability_range.front()) +
                         "-" +
                         Round1000DoubleToStr(
                             data_.sensor_probability_range.size() > 1 ? (data_.sensor_probability_range.back() - data_.sensor_probability_range.front()) /
                                                                             (data_.sensor_probability_range.size() - 1)
                                                                       : 0) +
                         "-" +
                         Round1000DoubleToStr(data_.sensor_probability_range.back()) +
                         "_" +
                         "prior" + std::to_string(data_.prior_param) +
                         "_" +
                         "thrsh" + Round1000DoubleToStr(data_.credible_threshold) +
                         "_" +
                         "posfb" + std::to_string(data_.positive_feedback);

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
            EBERT_2020_PARAM_ABBR + Round1000DoubleToStr((*itr)[EBERT_2020_PARAM_ABBR]) + "_" +
            "t" + std::to_string((*itr)["trial_ind"].get<int>());

        filename = filepath_prefix + name_ext_pair_output.first + "_" + filename + "." + name_ext_pair_output.second;

        // Export to single JSON file
        std::ofstream outfile(filename);

        outfile << std::setw(4) << (*itr) << std::endl; // write pretty JSON
    }
}