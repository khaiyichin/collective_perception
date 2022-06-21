#include "collective_perception_loop_functions.hpp"

void InitializeRobot::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    // Set robot sensor probabilities
    BuzzPut(t_vm, "b_prob", b_prob);
    BuzzPut(t_vm, "w_prob", w_prob);

    // Set robot speed
    BuzzPut(t_vm, "spd", spd);

    // Initialize RobotIDBrainMap
    (*id_brain_map_ptr)[str_robot_id.c_str()] = Brain(str_robot_id, b_prob, w_prob);
    auto &robot_brain = (*id_brain_map_ptr)[str_robot_id.c_str()];

    // Open the local_vals table
    BuzzTableOpen(t_vm, "local_vals");
    buzzobj_t tLocalVals = BuzzGet(t_vm, "local_vals");

    // Ensure the type is correct (a table)
    if (!buzzobj_istable(tLocalVals))
    {
        LOGERR << str_robot_id << ": variable \"local_vals\" has wrong type " << buzztype_desc[tLocalVals->o.type] << std::endl;
        return;
    }

    // Extract values from the opened "local_vals" table (now a global variable)
    Brain::ValuePair v(
        buzzobj_getfloat(BuzzTableGet(t_vm, "x")),
        buzzobj_getfloat(BuzzTableGet(t_vm, "confidence")));

    ///< @todo: do i need to close the table here?

    // Store the local values
    ///< @todo: this should probably be done without the body; the brain only cares about the observations and neighbor values, not self
    robot_brain.StoreLocalValuePair(v);
}

void ProcessRobotThought::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    // Collect debugging values for AgentData objects
    unsigned int encounter, observation;

    BuzzTableOpen(t_vm, "debug_data");
    buzzobj_t tDebugData = BuzzGet(t_vm, "debug_data");

    if (!buzzobj_istable(tDebugData))
    {
        LOGERR << str_robot_id << ": variable \"debug_data\" has wrong type " << buzztype_desc[tDebugData->o.type] << std::endl;
        return;
    }
    else
    {
        // Extract the integer component from the ID
        std::string dup_str_robot_id = str_robot_id;
        int index = std::stoi(dup_str_robot_id.erase(0, prefix.length())) - base_num;

        // Get data
        auto &agent_data = (*agt_data_vec_ptr)[index];
        agent_data.tile_occurrences.push_back(buzzobj_getint(BuzzTableGet(t_vm, "encounter")));
        agent_data.tile_occurrences.push_back(buzzobj_getint(BuzzTableGet(t_vm, "observation")));

        BuzzTableClose(t_vm);
    }

    // Get reference to the robot brain
    auto &robot_brain = (*id_brain_map_ptr)[str_robot_id.c_str()];

    // Collect observations
    buzzobj_t tTotalBlackObs = BuzzGet(t_vm, "total_b_tiles_obs");
    buzzobj_t tTotalObs = BuzzGet(t_vm, "total_obs");

    int total_black_obs, total_obs;

    // Verify that the types are correct
    if (!buzzobj_isint(tTotalBlackObs))
    {
        // Temporary hack
        if (buzzobj_isfloat(tTotalBlackObs))
        {
            total_black_obs = static_cast<int>(buzzobj_getfloat(tTotalBlackObs));
        }
        else
        {
            LOGERR << str_robot_id << ": variable \"total_b_tiles_obs\" has wrong type " << buzztype_desc[tTotalBlackObs->o.type] << std::endl;
            return;
        }
    }
    else
    {
        total_black_obs = buzzobj_getint(tTotalBlackObs);
    }

    if (!buzzobj_isint(tTotalObs))
    {
        LOGERR << str_robot_id << ": variable \"total_obs\" has wrong type " << buzztype_desc[tTotalObs->o.type] << std::endl;
        return;
    }
    else
    {
        total_obs = buzzobj_getint(tTotalObs);
    }

    // Store the observation into the brain
    robot_brain.StoreObservations(total_black_obs, total_obs);

    // Collect neighbors' values
    BuzzTableOpen(t_vm, "past_neighbor_vals");
    buzzobj_t tNeighborVals = BuzzGet(t_vm, "past_neighbor_vals");

    // Ensure the type is correct (a table)
    if (!buzzobj_istable(tNeighborVals))
    {
        LOGERR << str_robot_id << ": variable \"neighbor_vals\" has wrong type " << buzztype_desc[tNeighborVals->o.type] << std::endl;
        return;
    }

    // Extract values from the opened "neighbor_vals" table
    size_t tNeighborValsSize = tNeighborVals->t.value->size; // ->t represents the buzzvm_u union as a table, which is a struct that contains the attribute `value` which is a buzzdict_s type

    std::vector<Brain::ValuePair> value_pair_vec; // vecValuePair

    for (int i = 0; i < tNeighborValsSize; ++i)
    {
        // Open the nested table to get the neighbor's values
        BuzzTableOpenNested(t_vm, i);

        Brain::ValuePair v(
            buzzobj_getfloat(BuzzTableGet(t_vm, "x")),
            buzzobj_getfloat(BuzzTableGet(t_vm, "conf")));

        BuzzTableCloseNested(t_vm);

        value_pair_vec.push_back(v);
    }

    BuzzTableClose(t_vm); // close the "neighbor_vals" table

    robot_brain.StoreNeighborValuePairs(value_pair_vec);

    // Solve for values
    robot_brain.Solve();

    // Provide updated local values back to the robot body
    BuzzTableOpen(t_vm, "local_vals");

    BuzzTablePut(t_vm, "x", robot_brain.GetLocalValuePair().x);
    BuzzTablePut(t_vm, "conf", robot_brain.GetLocalValuePair().confidence);
}

void CollectivePerceptionLoopFunctions::Init(TConfigurationNode &t_tree)
{
    // Call parent's Init
    CBuzzLoopFunctions::Init(t_tree);

    // Extract XML information
    try
    {
        // Grab the reference to the XML node with the tag "collective_perception"
        TConfigurationNode &col_per_root_node = GetNode(t_tree, "collective_perception");

        // Grab arena information
        TConfigurationNode &arena_tiles_node = GetNode(col_per_root_node, "arena_tiles");

        // Grab verbosity level
        GetNodeAttribute(GetNode(col_per_root_node, "verbosity"), "level", verbose_level_);

        // Get a pointer to the ARGoS floor entity (method provided by superclass)
        CSpace &space_entity = GetSpace();
        floor_entity_ptr_ = &space_entity.GetFloorEntity();

        // Get the size of the arena (in units of tiles)
        uint32_t arena_x, arena_y;

        GetNodeAttribute(arena_tiles_node, "tile_count_x", arena_x);
        GetNodeAttribute(arena_tiles_node, "tile_count_y", arena_y);

        arena_tile_count_ = std::make_pair(arena_x, arena_y);

        // Get the limits of arena
        CRange<CVector3> lims = space_entity.GetArenaLimits();
        arena_lower_lim_ = std::make_pair(static_cast<float>(lims.GetMin().GetX()),
                                          static_cast<float>(lims.GetMin().GetY()));

        // Compute tile size
        CVector3 arena_size = space_entity.GetArenaSize();

        float length_x = arena_size.GetX() / arena_x; // tile size in the x-direction
        float length_y = arena_size.GetY() / arena_y; // tile size in the y-direction

        assert(length_x == length_y); // only square tiles allowed
        arena_tile_size_ = length_x;

        // Grab fill ratio ranges
        TConfigurationNode &fill_ratio_node = GetNode(col_per_root_node, "fill_ratio_range");

        float min, max, steps;

        GetNodeAttribute(fill_ratio_node, "min", min);
        GetNodeAttribute(fill_ratio_node, "max", max);
        GetNodeAttribute(fill_ratio_node, "steps", steps);

        simulation_parameters_.tfr_range_ = GenerateLinspace(min, max, steps);

        // Grab sensor probability ranges
        TConfigurationNode &sensor_probability_node = GetNode(col_per_root_node, "sensor_probability_range");

        GetNodeAttribute(sensor_probability_node, "min", min);
        GetNodeAttribute(sensor_probability_node, "max", max);
        GetNodeAttribute(sensor_probability_node, "steps", steps);

        simulation_parameters_.sp_range_ = GenerateLinspace(min, max, steps);

        // Create pairings for target fill ratios and sensor probabilities
        for (const float &tfr : simulation_parameters_.tfr_range_)
        {
            for (const float &sp : simulation_parameters_.sp_range_)
            {
                tfr_sp_ranges_.push_back(std::pair<float, float>(tfr, sp));
            }
        }

        curr_tfr_sp_range_itr_ = tfr_sp_ranges_.begin();

        // Grab robot speeds
        GetNodeAttribute(GetNode(col_per_root_node, "speed"), "value", simulation_parameters_.speed_);

        // Grab number of agents and communications range
        auto &rab_map = space_entity.GetEntitiesByType("rab");
        CRABEquippedEntity &random_rab = *any_cast<CRABEquippedEntity *>(rab_map.begin()->second);

        simulation_parameters_.num_agents_ = rab_map.size();         // the number of range and bearing sensors is the same as the number of robots
        simulation_parameters_.comms_range_ = random_rab.GetRange(); // all the range and bearing sensors have the same range
        simulation_parameters_.density_ = simulation_parameters_.num_agents_ * M_PI * std::pow(simulation_parameters_.comms_range_, 2) /
                                          (arena_size.GetX() * arena_size.GetY()); // the density is the ratio of swarm communication area to total walkable area

        // Grab number of trials
        GetNodeAttribute(GetNode(col_per_root_node, "num_trials"), "value", simulation_parameters_.num_trials_);

        // Grab number of steps
        simulation_parameters_.num_steps_ = GetSimulator().GetMaxSimulationClock();

        // Grab robot ID prefix and base number
        TConfigurationNode &robot_id_node = GetNode(col_per_root_node, "robot_id");

        GetNodeAttribute(robot_id_node, "prefix", id_prefix_);
        GetNodeAttribute(robot_id_node, "base_num", id_base_num_);

        // Grab probotuf file save path
        TConfigurationNode &path_node = GetNode(col_per_root_node, "path");
        GetNodeAttribute(path_node, "stats", sim_stats_set_.proto_filepath_);
        GetNodeAttribute(path_node, "agent_data", sim_agent_data_set_.proto_filepath_);
        GetNodeAttribute(path_node, "include_datetime", proto_datetime_);

        // Populate simulation parameters for SimulationStatsSet and SimulationAgentDataSet objects
        sim_stats_set_.PopulateSimulationSetParams(simulation_parameters_);
        sim_agent_data_set_.PopulateSimulationSetParams(simulation_parameters_);

        if (verbose_level_ == "full" || verbose_level_ == "reduced")
        {
            LOG << "[INFO] Collective perception loop functions verbose level = \"" << verbose_level_ << "\"" << std::endl;
            LOG << "[INFO] Specifying number of arena tiles = " << arena_x << "*" << arena_y << std::endl;
            LOG << "[INFO] Specifying robot speed = " << simulation_parameters_.speed_ << " cm/s" << std::endl;
            LOG << "[INFO] Specifying number of trials = " << simulation_parameters_.num_trials_ << std::endl;
            LOG << "[INFO] Specifying output statistics filepath (" << ((proto_datetime_) ? "with" : "without") << " datetime) = \"" << sim_stats_set_.proto_filepath_ << "\"" << std::endl;
            LOG << "[INFO] Specifying output agent data filepath (" << ((proto_datetime_) ? "with" : "without") << " datetime) = \"" << sim_agent_data_set_.proto_filepath_ << "\"" << std::endl;

            LOG << "[INFO] Computed swarm density = " << simulation_parameters_.density_ << std::endl;
            LOG << "[INFO] Generated tile size = " << arena_tile_size_ << " m" << std::endl;

            LOG << "[INFO] Running trial 1 with new parameters:"
                << " target fill ratio = " << curr_tfr_sp_range_itr_->first
                << " & sensor probability = " << curr_tfr_sp_range_itr_->second
                << std::endl;
        }
    }
    catch (CARGoSException &ex)
    {
        THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!", ex);
    }

    // Create Packet to store data
    CreateNewPacket();

    // Setup experiment
    SetupExperiment();
}

void CollectivePerceptionLoopFunctions::CreateNewPacket()
{
    curr_stats_packet_ = StatsPacket(simulation_parameters_.num_trials_);
    curr_stats_packet_.comms_range = simulation_parameters_.comms_range_;
    curr_stats_packet_.target_fill_ratio = curr_tfr_sp_range_itr_->first;
    curr_stats_packet_.b_prob = curr_tfr_sp_range_itr_->second;
    curr_stats_packet_.w_prob = curr_tfr_sp_range_itr_->second;
    curr_stats_packet_.num_agents = simulation_parameters_.num_agents_;
    curr_stats_packet_.num_trials = simulation_parameters_.num_trials_;
    curr_stats_packet_.num_steps = simulation_parameters_.num_steps_;
    curr_stats_packet_.density = simulation_parameters_.density_;

    curr_agent_data_packet_ = AgentDataPacket(simulation_parameters_.num_trials_, simulation_parameters_.num_agents_);
    curr_agent_data_packet_.comms_range = simulation_parameters_.comms_range_;
    curr_agent_data_packet_.target_fill_ratio = curr_tfr_sp_range_itr_->first;
    curr_agent_data_packet_.b_prob = curr_tfr_sp_range_itr_->second;
    curr_agent_data_packet_.w_prob = curr_tfr_sp_range_itr_->second;
    curr_agent_data_packet_.num_agents = simulation_parameters_.num_agents_;
    curr_agent_data_packet_.num_trials = simulation_parameters_.num_trials_;
    curr_agent_data_packet_.num_steps = simulation_parameters_.num_steps_;
    curr_agent_data_packet_.density = simulation_parameters_.density_;
}

void CollectivePerceptionLoopFunctions::SetupExperiment()
{
    // Create new Arena object
    arena_ = Arena(arena_tile_count_, arena_lower_lim_, arena_tile_size_, curr_tfr_sp_range_itr_->first);

    if (verbose_level_ == "full")
    {
        LOG << "[INFO] Arena tile fill ratio = " << arena_.GetTrueTileDistribution() << " with " << arena_.GetTotalNumTiles() << " tiles." << std::endl;
    }

    // Setup functors
    std::vector<AgentData> *curr_agent_data_vec_ptr = &curr_agent_data_packet_.repeated_agent_data_vec[trial_counter_];

    initialization_functor_ = InitializeRobot(id_brain_map_ptr_, curr_tfr_sp_range_itr_->second, simulation_parameters_.speed_);
    process_thought_functor_ = ProcessRobotThought(id_brain_map_ptr_,
                                                   curr_agent_data_vec_ptr,
                                                   id_prefix_,
                                                   id_base_num_);

    // Re-initialize each robot
    BuzzForeachVM(initialization_functor_);

    // Compute the pre-experiment statistics
    ComputeStats();
}

void CollectivePerceptionLoopFunctions::PostStep()
{
    // Iterate through each brain to process 'thought'
    BuzzForeachVM(process_thought_functor_);

    // Compute statistics
    ComputeStats();
}

void CollectivePerceptionLoopFunctions::ComputeStats()
{
    // Get all agent values
    auto solver_vals = GetAllSolverValues();
    std::vector<Brain::ValuePair> local = solver_vals[0];
    std::vector<Brain::ValuePair> social = solver_vals[1];
    std::vector<Brain::ValuePair> informed = solver_vals[2];

    // Compute mean and std dev across all values
    std::pair<Brain::ValuePair, Brain::ValuePair> local_vals = ComputeValuePairsSampleMeanAndStdDev(local);
    std::pair<Brain::ValuePair, Brain::ValuePair> social_vals = ComputeValuePairsSampleMeanAndStdDev(social);
    std::pair<Brain::ValuePair, Brain::ValuePair> informed_vals = ComputeValuePairsSampleMeanAndStdDev(informed);

    // Store mean values
    curr_stats_packet_.repeated_local_values[trial_counter_].x_sample_mean.push_back(local_vals.first.x);
    curr_stats_packet_.repeated_local_values[trial_counter_].confidence_sample_mean.push_back(local_vals.first.confidence);

    curr_stats_packet_.repeated_social_values[trial_counter_].x_sample_mean.push_back(social_vals.first.x);
    curr_stats_packet_.repeated_social_values[trial_counter_].confidence_sample_mean.push_back(social_vals.first.confidence);

    curr_stats_packet_.repeated_informed_values[trial_counter_].x_sample_mean.push_back(informed_vals.first.x);
    curr_stats_packet_.repeated_informed_values[trial_counter_].confidence_sample_mean.push_back(informed_vals.first.confidence);

    // Store std dev values
    curr_stats_packet_.repeated_local_values[trial_counter_].x_sample_std.push_back(local_vals.second.x);
    curr_stats_packet_.repeated_local_values[trial_counter_].confidence_sample_std.push_back(local_vals.second.confidence);

    curr_stats_packet_.repeated_social_values[trial_counter_].x_sample_std.push_back(social_vals.second.x);
    curr_stats_packet_.repeated_social_values[trial_counter_].confidence_sample_std.push_back(social_vals.second.confidence);

    curr_stats_packet_.repeated_informed_values[trial_counter_].x_sample_std.push_back(informed_vals.second.x);
    curr_stats_packet_.repeated_informed_values[trial_counter_].confidence_sample_std.push_back(informed_vals.second.confidence);
}

std::array<std::vector<Brain::ValuePair>, 3> CollectivePerceptionLoopFunctions::GetAllSolverValues()
{
    std::vector<Brain::ValuePair> local;
    std::vector<Brain::ValuePair> social;
    std::vector<Brain::ValuePair> informed;

    local.reserve(sim_stats_set_.num_agents_);
    social.reserve(sim_stats_set_.num_agents_);
    informed.reserve(sim_stats_set_.num_agents_);

    for (auto &kv : *id_brain_map_ptr_)
    {
        local.push_back(kv.second.GetLocalValuePair());
        social.push_back(kv.second.GetSocialValuePair());
        informed.push_back(kv.second.GetInformedValuePair());
    }

    return {local, social, informed};
}

std::pair<Brain::ValuePair, Brain::ValuePair> CollectivePerceptionLoopFunctions::ComputeValuePairsSampleMeanAndStdDev(const std::vector<Brain::ValuePair> &input)
{
    // Compute sample mean
    auto sum_func = [](const Brain::ValuePair &a, const Brain::ValuePair &b)
    {
        return Brain::ValuePair(a.x + b.x, a.confidence + b.confidence);
    };

    Brain::ValuePair sample_mean = std::reduce(input.begin(), input.end(), Brain::ValuePair(0.0, 0.0), sum_func) / input.size();

    // Compute sample std dev
    auto sq_err_func = [sample_mean](const Brain::ValuePair &v)
    {
        return Brain::ValuePair(std::pow(v.x - sample_mean.x, 2),
                                std::pow(v.confidence - sample_mean.confidence, 2));
    };

    std::vector<Brain::ValuePair> squared_errors(input.size());
    std::transform(input.begin(), input.end(), squared_errors.begin(), sq_err_func);
    Brain::ValuePair mean_squared_errors = std::reduce(squared_errors.begin(),
                                                       squared_errors.end(),
                                                       Brain::ValuePair(0.0, 0.0), sum_func) /
                                           (input.size() - 1);
    Brain::ValuePair sample_std_dev(std::sqrt(mean_squared_errors.x), std::sqrt(mean_squared_errors.confidence));

    return std::make_pair(sample_mean, sample_std_dev);
}

void CollectivePerceptionLoopFunctions::PostExperiment()
{
    // Check to see if all trials are complete
    if (++trial_counter_ % simulation_parameters_.num_trials_ == 0) // all trials for current param set is done
    {
        trial_counter_ = 0; // reset counter

        ++curr_tfr_sp_range_itr_; // use next parameter set

        // Store Packet
        sim_stats_set_.InsertStatsPacket(curr_stats_packet_);
        sim_agent_data_set_.InsertAgentDataPacket(curr_agent_data_packet_);

        // Check parameter sets
        if (curr_tfr_sp_range_itr_ != tfr_sp_ranges_.end()) // more parameter sets are left
        {
            if (verbose_level_ == "full" || verbose_level_ == "reduced")
            {
                LOG << "[INFO] Running trial 1 with new parameters:"
                    << " target fill ratio = " << curr_tfr_sp_range_itr_->first
                    << " & sensor probability = " << curr_tfr_sp_range_itr_->second
                    << std::endl;
            }

            // Create Packet to store data
            CreateNewPacket();
        }
        else // no more parameter sets
        {
            if (verbose_level_ == "full" || verbose_level_ == "reduced")
            {
                LOG << "[INFO] All simulation parameters executed." << std::endl;
            }

            // Export SimulationStatsSet object
            collective_perception_cpp::proto::SimulationStatsSet sim_stats_set_proto_msg;
            collective_perception_cpp::proto::SimulationAgentDataSet sim_agent_data_set_proto_msg;

            sim_stats_set_.Serialize(sim_stats_set_proto_msg);
            sim_agent_data_set_.Serialize(sim_agent_data_set_proto_msg);

            // Create output filename
            std::string output_file_stats, output_file_agent_data;

            if (proto_datetime_)
            {
                // Get current time in string form
                time_t curr_time;
                time(&curr_time);
                tm *curr_tm = localtime(&curr_time);

                std::string datetime_str;
                datetime_str.resize(100);

                strftime(&(datetime_str[0]), datetime_str.size(), "%m%d%y_%H%M%S", curr_tm);

                // Strip extension from filename
                std::pair<std::string, std::string> name_ext_pair_stats, name_ext_pair_agent_data;
                std::stringstream stream_stats(sim_stats_set_.proto_filepath_);
                std::stringstream stream_agent_data(sim_agent_data_set_.proto_filepath_);

                getline(stream_stats, name_ext_pair_stats.first, '.');
                getline(stream_stats, name_ext_pair_stats.second, '.');
                getline(stream_agent_data, name_ext_pair_agent_data.first, '.');
                getline(stream_agent_data, name_ext_pair_agent_data.second, '.');

                // Generate updated filename
                output_file_stats = name_ext_pair_stats.first + "_" + datetime_str.c_str() + "." + name_ext_pair_stats.second;
                output_file_agent_data = name_ext_pair_agent_data.first + "_" + datetime_str.c_str() + "." + name_ext_pair_agent_data.second;
            }
            else
            {
                output_file_stats = sim_stats_set_.proto_filepath_;
                output_file_agent_data = sim_agent_data_set_.proto_filepath_;
            }

            WriteProtoToDisk(sim_stats_set_proto_msg, output_file_stats);
            WriteProtoToDisk(sim_agent_data_set_proto_msg, output_file_agent_data);

            google::protobuf::ShutdownProtobufLibrary();

            finished_ = true;
        }
    }
    else // more trials required for the current param set
    {

        // Repeat trial
        if (verbose_level_ == "full")
        {
            LOG << "[INFO] Running trial " << trial_counter_ + 1 << " with same parameters." << std::endl;
        }
    }
}

CColor CollectivePerceptionLoopFunctions::GetFloorColor(const CVector2 &c_position_on_plane)
{
    uint32_t color_int = arena_.GetColor(c_position_on_plane.GetX(), c_position_on_plane.GetY());

    return color_int == 1 ? CColor::BLACK : CColor::WHITE;
}

template <typename T>
std::vector<T> CollectivePerceptionLoopFunctions::GenerateLinspace(const T &min, const T &max, const size_t &steps)
{
    // Compute increment
    T inc = (max - min) / static_cast<T>(steps - 1);

    // Populate vector
    std::vector<T> output(steps);
    T val;

    for (auto itr = output.begin(), val = min; itr != output.end(); ++itr, val += inc)
    {
        *itr = val;
    }

    return output;
}

REGISTER_LOOP_FUNCTIONS(CollectivePerceptionLoopFunctions, "collective_perception_loop_functions")