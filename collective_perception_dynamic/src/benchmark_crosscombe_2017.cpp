#include "benchmark_crosscombe_2017.hpp"

void ProcessRobotBelief::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    if (!initialized)
    {
        // Initialize self belief states to be indeterminate
        std::vector<int> init_belief(num_options, static_cast<int>(BeliefState::indeterminate));

        PopulateSelfBeliefVecInVM(t_vm, init_belief);

        // Update the robot state
        BuzzPut(t_vm, "state", static_cast<int>(RobotState::signalling));

        std::pair<std::set<std::string>::iterator, bool> insert_receipt = initialized_robot_ids.insert(str_robot_id);

        if (insert_receipt.second == false)
        {
            THROW_ARGOSEXCEPTION("The same robot has been initialized!");
        }

        if (initialized_robot_ids.size() == num_robots)
        {
            initialized = true;
        }

        (*id_belief_map_ptr)[str_robot_id.c_str()] = std::vector<std::string>{};
    }
    else
    {
        // Only update beliefs if in updating state
        int state = buzzobj_getint(BuzzGet(t_vm, "state"));

        if (state == static_cast<int>(RobotState::updating))
        {
            // Prevent belief updates if robot is malfunctioning
            if (IsMalfunctioning(str_robot_id))
            {
                // Pick random option
                std::vector<int> all_options_indices(num_options);
                std::iota(all_options_indices.begin(), all_options_indices.end(), 0);

                std::vector<int> self_belief = GenerateBeliefVecFromRandomOption(all_options_indices);

                PopulateSelfBeliefVecInVM(t_vm, self_belief);
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
                std::vector<int> self_belief = ExtractSelfBeliefVecFromVM(t_vm);

                // Store self belief
                (*id_belief_map_ptr)[str_robot_id.c_str()].push_back(ConvertBeliefVecToString(self_belief));

                // Pick one of the neighbors to use their beliefs
                std::vector<int> signalled_belief;
                signalled_belief.reserve(num_options);

                BuzzTableOpen(t_vm, "past_signalled_beliefs");
                buzzobj_t tSignalledBeliefs = BuzzGet(t_vm, "past_signalled_beliefs");

                // Ensure the type is correct (a table)
                if (!buzzobj_istable(tSignalledBeliefs))
                {
                    LOGERR << str_robot_id << ": variable \"past_signalled_beliefs\" has wrong type " << buzztype_desc[tSignalledBeliefs->o.type] << std::endl;
                    return;
                }
                else
                {
                    // Randomly draw one value from the opened `past_signalled_beliefs` table
                    size_t tSignalledBeliefsSize = tSignalledBeliefs->t.value->size; // ->t represents the buzzvm_u union as a table, which is a struct that contains the attribute `value` which is a buzzdict_s type

                    std::uniform_int_distribution<int> dist(0, tSignalledBeliefsSize);

                    int random_neighbor_index = dist(generator);

                    // Extract randomly picked neighbor's belief
                    BuzzTableOpenNested(t_vm, std::to_string(random_neighbor_index));

                    for (int i = 0; i < num_options; ++i)
                    {
                        signalled_belief.push_back(buzzobj_getint(BuzzTableGet(t_vm, std::to_string(i))));
                    }

                    BuzzTableCloseNested(t_vm);
                }

                BuzzTableClose(t_vm);

                // Update self belief
                self_belief = UpdateSelfBeliefVec(self_belief, signalled_belief);

                // Obtain broadcast duration based on self belief
                size_t option = std::find(self_belief.begin(),
                                          self_belief.end(),
                                          static_cast<int>(BeliefState::positive)) -
                                self_belief.begin();

                // Update broadcast duration based on the selected option
                int duration = GetBroadcastDuration(option);

                BuzzPut(t_vm, "broadcast_duration", duration);

                // Populate self belief into the Buzz VM
                PopulateSelfBeliefVecInVM(t_vm, self_belief);
            }
        }
        else if (!IsMalfunctioning(str_robot_id) && state == static_cast<int>(RobotState::signalling)) // non-malfunctioning and signalling
        {
            // Extract and store self belief
            std::vector<int> self_belief = ExtractSelfBeliefVecFromVM(t_vm);

            (*id_belief_map_ptr)[str_robot_id.c_str()].push_back(ConvertBeliefVecToString(self_belief));
        }
    }
}

// std::string ProcessRobotBelief::ConvertBeliefVecToString(const std::vector<int> &belief_vec)

std::vector<int> ProcessRobotBelief::UpdateSelfBeliefVec(const std::vector<int> &self_belief_vec,
                                                         const std::vector<int> &signalled_belief_vec)
{
    /*
        The updated belief vector returned will only contain one positive belief and n-1 negative beliefs.
        This is because in the case of indeterminate beliefs, we go ahead and randomly choose one of the
        indeterminate options to become the positive belief.
    */

    // Define a lambda function to capture a reference to this object instance to call the member function
    auto truth_func = [this](const int &a, const int &b)
    { return GetTruthValue(a, b); };

    // Update self belief
    std::vector<int> updated_belief;

    std::transform(self_belief_vec.begin(), self_belief_vec.end(), signalled_belief_vec.begin(), updated_belief.begin(), truth_func);

    // Normalize the updated belief so that it consists of only one strong belief or only indeterminate & negative beliefs
    int num_pos_beliefs = std::count(updated_belief.begin(),
                                     updated_belief.end(),
                                     static_cast<int>(BeliefState::positive));

    if (num_pos_beliefs <= 1) // number of positive beliefs is within reason
    {
        if (num_pos_beliefs == 0) // no positive beliefs
        {
            // Check the number of indeterminate belief states
            int num_int_beliefs = 0;
            std::vector<int> int_belief_indices, randomly_selected_index(1);

            for (auto itr = updated_belief.begin(); itr != updated_belief.end(); ++itr)
            {
                if (*itr == static_cast<int>(BeliefState::indeterminate))
                {
                    int_belief_indices.push_back(itr - updated_belief.begin());
                    ++num_int_beliefs;
                }
            }

            if (num_int_beliefs == 1) // only one of the beliefs is indeterminate
            {
                std::replace(updated_belief.begin(),
                             updated_belief.end(),
                             static_cast<int>(BeliefState::indeterminate),
                             static_cast<int>(BeliefState::positive));
            }
            else // multiple indeterminate beliefs
            {
                updated_belief = GenerateBeliefVecFromRandomOption(int_belief_indices);
            }
        }
    }
    else
    {
        THROW_ARGOSEXCEPTION("Number of positive beliefs should only be one in belief state vector!");
    }

    return updated_belief;
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

    TConfigurationNode &flawed_ratio_range_node = GetNode(t_tree, "flawed_ratio_range");

    GetNodeAttribute(flawed_ratio_range_node, "min", min);
    GetNodeAttribute(flawed_ratio_range_node, "max", max);
    GetNodeAttribute(flawed_ratio_range_node, "steps", steps);

    data_.flawed_ratio_range = GenerateLinspace(min, max, steps);
}

void BenchmarkCrosscombe2017::SetupExperiment(const std::pair<double, double> &curr_paired_parameters)
{
    // Compute current option qualities
    ComputeOptionQualities(curr_paired_parameters.first);

    // Sample robots to make unreliable (malfunctioning robots)
    int num_malf_robots = static_cast<int>(std::round(data_.num_agents * curr_paired_parameters.second));

    std::vector<int> malf_robot_ids = SampleRobotIdsWithoutReplacement(num_malf_robots, data_.id_base_num);

    // Setup functors
    process_robot_belief_functor_ = ProcessRobotBelief(data_.id_prefix,
                                                       data_.id_base_num,
                                                       data_.num_possible_options,
                                                       data_.num_agents,
                                                       malf_robot_ids,
                                                       curr_option_qualities_,
                                                       id_belief_map_ptr_);

    // Initialize each robot
    // (*buzz_foreach_vm_func_ptr_)(process_robot_belief_functor_);
    buzz_foreach_vm_func_(process_robot_belief_functor_);

    // Compute the pre-experiment statistics
    // ComputeStats();

    return;
}

void BenchmarkCrosscombe2017::ComputeStats()
{
    // Store
    // curr_json_data_itr_;
}

void BenchmarkCrosscombe2017::PostStep()
{
    // Iterate through robots to process their beliefs
    buzz_foreach_vm_func_(process_robot_belief_functor_);

    // ComputeStats(); ?
}

void BenchmarkCrosscombe2017::PostExperiment()
{
}

void BenchmarkCrosscombe2017::InitializeJson(const std::pair<double, double> &curr_paired_parameters)
{
    (*curr_json_data_itr_)["sim_type"] = data_.simulation_type;
    (*curr_json_data_itr_)["num_agents"] = data_.num_agents;
    (*curr_json_data_itr_)["num_trials"] = data_.num_trials;
    (*curr_json_data_itr_)["num_steps"] = data_.num_steps;
    (*curr_json_data_itr_)["comms_range"] = data_.comms_range;
    (*curr_json_data_itr_)["density"] = data_.density;
    (*curr_json_data_itr_)["tfr"] = curr_paired_parameters.first;
    (*curr_json_data_itr_)[data_.parameter_keyword] = curr_paired_parameters.second;
    (*curr_json_data_itr_)["option_qualities"] = curr_option_qualities_;
}

void BenchmarkCrosscombe2017::WriteToJson()
{
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
    std::transform(weights.begin(), weights.end(), curr_option_qualities_.begin(), [&total_weight](double w)
                   { return static_cast<unsigned int>(std::round(100 * w / total_weight)); });
}
