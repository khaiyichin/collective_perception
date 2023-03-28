#include "benchmark_crosscombe_2017.hpp"

void ProcessRobotBelief::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    if (!initialized)
    {
        // Set robot speed
        BuzzPut(t_vm, "spd", spd);

        // Randomize initial robot belief
        std::vector<int> init_belief_vec(num_options);
        init_belief_vec = UpdateSelfBeliefVec(t_vm, std::make_pair(init_belief_vec, std::vector<int>()), true);

        // Randomize the robot state
        std::vector<int> states, selected_state(1);
        states = {static_cast<int>(RobotState::signalling), static_cast<int>(RobotState::updating)};

        std::sample(states.begin(),
                    states.end(),
                    selected_state.begin(),
                    1,
                    generator);
        BuzzPut(t_vm, "state", *selected_state.begin());

        std::pair<std::set<std::string>::iterator, bool> insert_receipt = initialized_robot_ids.insert(str_robot_id);

        if (insert_receipt.second == false)
        {
            THROW_ARGOSEXCEPTION("The same robot has been initialized!");
        }

        if (initialized_robot_ids.size() == num_robots)
        {
            initialized = true;
        }

        if (!IsFlawed(str_robot_id))
        {
            (*id_belief_map_ptr)[str_robot_id.c_str()] = std::vector<std::string>{};
            (*id_belief_map_ptr)[str_robot_id.c_str()].push_back(ConvertBeliefVecToString(init_belief_vec));
        }
    }
    else
    {
        // Only update beliefs if in updating state
        int state = buzzobj_getint(BuzzGet(t_vm, "state"));
        std::vector<int> curr_self_belief_vec, updated_belief_vec;

        if (state == static_cast<int>(RobotState::updating))
        {
            // Prevent belief updates if robot is flawed
            if (IsFlawed(str_robot_id))
            {
                // Pick random option
                curr_self_belief_vec = std::vector<int>(num_options, static_cast<int>(BeliefState::indeterminate));

                updated_belief_vec = UpdateSelfBeliefVec(t_vm,
                                                         std::make_pair(curr_self_belief_vec, std::vector<int>()),
                                                         true);
            }
            else
            {
                /*
                    Currently I don't have a good way to perform a pairwise binary operation on two tables
                    simultaneously. Specifically, I don't know if it's possible to have two opened tables using
                    BuzzTableOpen.

                    Without that, I can't operate on the two table elements in pairwise fashion in a for loop. There
                    are 2 alternatives:

                        1. Load the two tables into 2 vectors, then perform the operation. This involves sequential
                        table opening and closings in 2 separate for loops, so shouldn't be a problem.

                        2. Within a single iteration in a for loop, open the first table and get the element,
                        then close it. Do this for the second table. Repeat for the remaining iterations.

                    The issue with the 2 alternatives is that:

                        1. The for loop is run at least 3 times. Once for populating the first vector, once for
                        populating the second vector, and once for updating the table with the result. There are
                        potentially more loops involved (but might be optimized internally based on the used libraries)
                        depending on the pairwise operation desired.

                        2. The cost of repeated openings and closings of a table is unknown. This is performed for
                        every robot per time step, which means it could be expensive if the cost is non-negligible.

                    For now, I'll use to alternative 1, because alternative 2 is unavailable; the second table is
                    actually an element in a parent table. Thus, I can't access this table without closing the parent
                    table, but that means extracting all the values in the second table into a vector (i.e.,
                    alternative 1).

                */

                // Extract self belief
                curr_self_belief_vec = ExtractSelfBeliefVecFromVM(t_vm);

                // Pick one of the neighbors to use their beliefs
                std::vector<int> signalled_belief_vec;
                signalled_belief_vec.reserve(num_options);

                BuzzTableOpen(t_vm, "past_signalled_beliefs");
                buzzobj_t tSignalledBeliefs = BuzzGet(t_vm, "past_signalled_beliefs");

                // Ensure the type is correct (a table)
                if (!buzzobj_istable(tSignalledBeliefs))
                {
                    LOGERR << str_robot_id << ": variable \"past_signalled_beliefs\" has wrong type " << buzztype_desc[tSignalledBeliefs->o.type] << std::endl;
                    return;
                }

                // Check whether signalled beliefs have been received (i.e., if there are neighbors)
                size_t tSignalledBeliefsSize = tSignalledBeliefs->t.value->size; // ->t represents the buzzvm_u union as a table, which is a struct that contains the attribute `value` which is a buzzdict_s type

                if (tSignalledBeliefsSize > 0)
                {
                    // Randomly draw one value from the opened `past_signalled_beliefs` table
                    // Pick a random neighbor's belief
                    std::uniform_int_distribution<int> dist(0, tSignalledBeliefsSize - 1);

                    int random_neighbor_index = dist(generator);

                    BuzzTableOpenNested(t_vm, random_neighbor_index);

                    for (int i = 0; i < num_options; ++i)
                    {
                        signalled_belief_vec.push_back(buzzobj_getint(BuzzTableGet(t_vm, i)));
                    }

                    BuzzTableCloseNested(t_vm);
                }

                // Update self belief
                updated_belief_vec = UpdateSelfBeliefVec(t_vm,
                                                         std::make_pair(curr_self_belief_vec, signalled_belief_vec),
                                                         false);

                BuzzTableClose(t_vm);
            }

            // Update the robot state
            BuzzPut(t_vm, "state", static_cast<int>(RobotState::signalling));
        }
        else if (!IsFlawed(str_robot_id) && state == static_cast<int>(RobotState::signalling)) // non-flawed and signalling
        {
            // Extract and store self belief
            updated_belief_vec = ExtractSelfBeliefVecFromVM(t_vm);
        }

        // Log the current belief (to be exported) for analysis
        if (!IsFlawed(str_robot_id))
        {
            (*id_belief_map_ptr)[str_robot_id.c_str()].push_back(ConvertBeliefVecToString(updated_belief_vec));
        }
    }
}

std::vector<int> ProcessRobotBelief::UpdateSelfBeliefVec(buzzvm_t t_vm,
                                                         const std::pair<std::vector<int>, std::vector<int>> &self_and_signalled_beliefs,
                                                         const bool &randomize_belief /*=false*/)
{
    std::vector<int> self_belief_vec = self_and_signalled_beliefs.first;
    std::vector<int> signalled_belief_vec = self_and_signalled_beliefs.second;
    std::vector<int> updated_belief_vec;
    updated_belief_vec.reserve(self_belief_vec.size());

    // Check whether belief should be completely randomized (useful for flawed robots and for initial beliefs)
    if (!randomize_belief)
    {
        // Check whether a neighbor's signalled belief is available
        if (signalled_belief_vec.empty())
        {
            updated_belief_vec = self_belief_vec;
        }
        else
        {
            // Define a lambda function to capture a reference to this object instance to call the member function
            auto truth_func = [&, this](const int &a, const int &b)
            { return GetTruthValue(a, b); };

            // Fuse the signalled belief into the self belief
            std::transform(self_belief_vec.begin(),
                           self_belief_vec.end(),
                           signalled_belief_vec.begin(),
                           std::back_inserter(updated_belief_vec),
                           truth_func);
        }
    }
    else
    {
        // Pick random belief states
        for (int i = 0; i < num_options; ++i)
        {
            updated_belief_vec.push_back(flawed_belief_dist(generator));
        }
    }

    // Populate the belief state and the duration to the VM (this takes care of the normalization)
    return NormalizeAndPopulateSelfBeliefVec(t_vm, updated_belief_vec);
}

std::vector<int> ProcessRobotBelief::NormalizeAndPopulateSelfBeliefVec(buzzvm_t t_vm, const std::vector<int> &self_belief_vec)
{
    int duration;
    std::vector<int> normalized_belief_vec = self_belief_vec;

    // Count the number of positive beliefs
    int num_pos_beliefs = std::count(normalized_belief_vec.begin(),
                                     normalized_belief_vec.end(),
                                     static_cast<int>(BeliefState::positive));

    // Ensure that the beliefs are properly normalized and find the correct broadcast duration
    if (num_pos_beliefs == 0) // no positive beliefs
    {
        // Check to see which belief states are indeterminate
        std::vector<int> int_belief_indices = GetIndeterminateBeliefIndices(normalized_belief_vec);
        int num_int_beliefs = int_belief_indices.size();

        if (num_int_beliefs == 1) // only one of the beliefs is indeterminate (deterministic option)
        {
            // Normalize the belief vector by replacing the indeterminate belief with a positive one
            std::replace(normalized_belief_vec.begin(),
                         normalized_belief_vec.end(),
                         static_cast<int>(BeliefState::indeterminate),
                         static_cast<int>(BeliefState::positive));

            // Calculate the broadcast duration for the option corresponding to the positive belief
            duration = GetBroadcastDuration(GetPositiveBeliefIndex(normalized_belief_vec));
        }
        else // either >1 or 0 indeterminate beliefs (probabilistic outcome)
        {
            std::vector<int> sample_space;

            if (num_int_beliefs == 0) // all negative beliefs
            {
                // Normalize the belief vector by replacing all negative beliefs with indeterminate beliefs
                std::replace(normalized_belief_vec.begin(),
                             normalized_belief_vec.end(),
                             static_cast<int>(BeliefState::negative),
                             static_cast<int>(BeliefState::indeterminate));

                // Define the sample space as all choices
                sample_space = std::vector<int>(num_options);

                std::iota(sample_space.begin(), sample_space.end(), 0);
            }
            else // multiple indeterminate beliefs
            {
                // Define the sample space as only choices with indeterminate beliefs
                sample_space = int_belief_indices;
            }

            // Randomly select one option
            std::vector<int> selection(1);

            std::sample(sample_space.begin(),
                        sample_space.end(),
                        selection.begin(),
                        1,
                        generator);

            // Calculate the broadcast duration for the selection option
            duration = GetBroadcastDuration(*selection.begin());
        }
    }
    else if (num_pos_beliefs == 1) // exactly 1 positive belief (and all other non-positive beliefs)
    {
        // Normalize the belief vector by replacing all indeterminate beliefs with negative beliefs
        std::replace(normalized_belief_vec.begin(),
                     normalized_belief_vec.end(),
                     static_cast<int>(BeliefState::indeterminate),
                     static_cast<int>(BeliefState::negative));

        // Calculate the broadcast duration for the option corresponding to the positive belief
        duration = GetBroadcastDuration(GetPositiveBeliefIndex(normalized_belief_vec));
    }
    else if (num_pos_beliefs > 1) // more than 1 positive belief (possibly containing indeterminate beliefs too)
    {
        /*
            Since there is > 1 positive belief, the robot becomes uncertain only between the positive beliefs;
            every other non-positive beliefs is ignored.
        */

        // Replace indeterminate values so that they become negative
        std::replace(normalized_belief_vec.begin(),
                     normalized_belief_vec.end(),
                     static_cast<int>(BeliefState::indeterminate),
                     static_cast<int>(BeliefState::negative));

        // Replace positive values so that they become indeterminate
        std::replace(normalized_belief_vec.begin(),
                     normalized_belief_vec.end(),
                     static_cast<int>(BeliefState::positive),
                     static_cast<int>(BeliefState::indeterminate));

        // Find the indices to the indeterminate belief states
        std::vector<int> int_belief_indices = GetIndeterminateBeliefIndices(normalized_belief_vec);

        // Randomly select one option
        std::vector<int> selection(1);

        std::sample(int_belief_indices.begin(),
                    int_belief_indices.end(),
                    selection.begin(),
                    1,
                    generator);

        // Calculate the broadcast duration for the selection option
        duration = GetBroadcastDuration(*selection.begin());
    }

    // Populate the self belief and broadcast duration in the VM
    PopulateSelfBeliefVecInVM(t_vm, normalized_belief_vec);
    PopulateBroadcastDurationInVM(t_vm, duration);

    return normalized_belief_vec;
}

std::vector<int> ProcessRobotBelief::GetIndeterminateBeliefIndices(const std::vector<int> &self_belief_vec)
{
    std::vector<int> int_belief_indices;

    for (auto itr = self_belief_vec.begin(); itr != self_belief_vec.end(); ++itr)
    {
        if (*itr == static_cast<int>(BeliefState::indeterminate))
        {
            int_belief_indices.push_back(itr - self_belief_vec.begin());
        }
    }

    return int_belief_indices;
}

int ProcessRobotBelief::GetTruthValue(const int &self_belief,
                                      const int &signalled_belief)
{
    if (self_belief == static_cast<int>(BeliefState::indeterminate))
    {
        return signalled_belief;
    }
    else if (signalled_belief == static_cast<int>(BeliefState::indeterminate))
    {
        return self_belief;
    }
    else // either both beliefs are the same or the opposite
    {
        return static_cast<int>((self_belief + signalled_belief) / 2);
    }
}

BenchmarkCrosscombe2017::BenchmarkCrosscombe2017(const BuzzForeachVMFunc &buzz_foreach_vm_func, TConfigurationNode &t_tree)
    : BenchmarkAlgorithmTemplate<BenchmarkDataCrosscombe2017>(buzz_foreach_vm_func)
{
    // Grab number of possible options
    GetNodeAttribute(GetNode(t_tree, "num_possible_options"), "int", data_.num_possible_options);

    // Grab flawed robot ratio range
    double min, max;
    int steps;

    TConfigurationNode &flawed_ratio_range_node = GetNode(t_tree, CROSSCOMBE_2017_PARAM + "_range");

    GetNodeAttribute(flawed_ratio_range_node, "min", min);
    GetNodeAttribute(flawed_ratio_range_node, "max", max);
    GetNodeAttribute(flawed_ratio_range_node, "steps", steps);

    data_.flawed_ratio_range = GenerateLinspace(min, max, steps);
}

void BenchmarkCrosscombe2017::SetupExperiment(const int &trial_ind, const std::pair<double, double> &curr_paired_parameters)
{
    // Assign values to member variables
    curr_trial_ind_ = trial_ind;
    curr_paired_parameters_ = curr_paired_parameters;

    // Compute current option qualities
    ComputeOptionQualities(curr_paired_parameters.first);

    // Sample robots to make unreliable (flawed robots)
    double num_flawed_robots_decimal = data_.num_agents * curr_paired_parameters.second;
    curr_num_flawed_robots_ = static_cast<int>(std::round(num_flawed_robots_decimal)); // the actual number of flawed robots for the current setup

    if (std::fmod(num_flawed_robots_decimal, 1.0) != 0)
    {
        LOG << "[INFO] The desired number of flawed robots = "
            << num_flawed_robots_decimal << " is not an integer. This will be rounded to the next closest integer = "
            << curr_num_flawed_robots_ << std::endl;
    }

    std::vector<int> flawed_robot_ids = SampleRobotIdsWithoutReplacement(curr_num_flawed_robots_, data_.id_base_num);

    // Setup functors
    id_belief_map_ptr_ = std::make_shared<RobotIdBeliefStrMap>();

    process_robot_belief_functor_ = ProcessRobotBelief(data_.id_prefix,
                                                       data_.id_base_num,
                                                       data_.num_possible_options,
                                                       data_.num_agents,
                                                       data_.speed,
                                                       flawed_robot_ids,
                                                       curr_option_qualities_,
                                                       id_belief_map_ptr_);

    // Initialize each robot
    buzz_foreach_vm_func_(process_robot_belief_functor_);

    // Initialize JSON file
    InitializeJson();

    return;
}

void BenchmarkCrosscombe2017::PostStep()
{
    // Iterate through robots to process their beliefs
    buzz_foreach_vm_func_(process_robot_belief_functor_);
}

void BenchmarkCrosscombe2017::PostExperiment(const bool &final_experiment /*=false*/)
{
    // Store new trial result
    std::string key = "beliefs";

    // Store data into current json
    std::vector<std::vector<std::string>> vec;
    vec.reserve(data_.num_agents);

    for (auto itr = id_belief_map_ptr_->begin(); itr != id_belief_map_ptr_->end(); ++itr)
    {
        vec.push_back(itr->second); // store the values from the map, which are vectors of strings
    }

    curr_json_[key] = vec;
    data_.json_data.push_back(curr_json_);
}

void BenchmarkCrosscombe2017::InitializeJson()
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
    curr_json_[CROSSCOMBE_2017_PARAM_ABBR] = curr_paired_parameters_.second;
    curr_json_["num_flawed_robots"] = curr_num_flawed_robots_;
    curr_json_["option_qualities"] = curr_option_qualities_;
    curr_json_["trial_ind"] = curr_trial_ind_;
}

void BenchmarkCrosscombe2017::SaveData(const std::string &foldername_prefix /*=""*/)
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
                         Round1000DoubleToStr(data_.tfr_range.back()) + "_" +
                         CROSSCOMBE_2017_PARAM_ABBR +
                         Round1000DoubleToStr(data_.flawed_ratio_range.front()) +
                         "-" +
                         Round1000DoubleToStr(
                             data_.flawed_ratio_range.size() > 1 ? (data_.flawed_ratio_range.back() - data_.flawed_ratio_range.front()) /
                                                                       (data_.flawed_ratio_range.size() - 1)
                                                                 : 0) +
                         "-" +
                         Round1000DoubleToStr(data_.flawed_ratio_range.back()) + "_" +
                         "opt" + std::to_string(data_.num_possible_options);

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
            CROSSCOMBE_2017_PARAM_ABBR + Round1000DoubleToStr((*itr)[CROSSCOMBE_2017_PARAM_ABBR]) + "_" +
            "t" + std::to_string((*itr)["trial_ind"].get<int>());

        filename = filepath_prefix + name_ext_pair_output.first + "_" + filename + "." + name_ext_pair_output.second;

        // Export to single JSON file
        std::ofstream outfile(filename);

        outfile << std::setw(4) << (*itr) << std::endl; // write pretty JSON
    }
}

void BenchmarkCrosscombe2017::ComputeOptionQualities(const double &curr_tfr)
{
    curr_option_qualities_.clear();

    // Compute weights
    double total_weight = 0;
    std::vector<double> weights;

    for (int i = 0; i < data_.num_possible_options; ++i)
    {
        double weight = 1 - std::abs(curr_tfr - (2.0 * i + 1) / (2.0 * data_.num_possible_options));

        weights.push_back(weight);
        total_weight += weight;
    }

    // Normalize weights into qualities
    std::transform(weights.begin(), weights.end(), std::back_inserter(curr_option_qualities_), [&total_weight](double w)
                   { return static_cast<unsigned int>(std::round(100 * w / total_weight)); });
}
