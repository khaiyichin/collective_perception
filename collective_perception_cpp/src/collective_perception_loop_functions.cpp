#include "collective_perception_loop_functions.hpp"

void InitializeRobot::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    BuzzPut(t_vm, "b_prob", b_prob);
    BuzzPut(t_vm, "w_prob", w_prob);

    // Use robot ID and sensor accuracy to instantiate Brain class
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
        TConfigurationNode &arena_node = GetNode(col_per_root_node, "arena");

        // Get a pointer to the ARGoS floor entity (method provided by superclass)
        CSpace &space_entity = GetSpace();
        floor_entity_ptr_ = &space_entity.GetFloorEntity();

        // Get the size of the arena (in units of tiles)
        uint32_t arena_x, arena_y;

        GetNodeAttribute(arena_node, "tile_count_x", arena_x);
        GetNodeAttribute(arena_node, "tile_count_y", arena_y);

        arena_tile_count_ = std::make_pair(arena_x, arena_y);

        // Get the limits of arena
        CRange<CVector3> lims = space_entity.GetArenaLimits();
        arena_lower_lim_ = std::make_pair(static_cast<float>(lims.GetMin().GetX()),
                                          static_cast<float>(lims.GetMin().GetY()));

        // Compute tile size
        CVector3 arena_size = space_entity.GetArenaSize();

        float length_x = arena_size.GetX() / arena_x;
        float length_y = arena_size.GetY() / arena_y;

        assert(length_x == length_y); // only square tiles allowed
        arena_tile_size_ = length_x;

        LOG << "[INFO] Generated tile size of " << arena_tile_size_ << "m" << std::endl;

        // Grab fill ratio ranges
        TConfigurationNode &fill_ratio_node = GetNode(col_per_root_node, "fill_ratio_range");

        float min, max, steps;

        GetNodeAttribute(fill_ratio_node, "min", min);
        GetNodeAttribute(fill_ratio_node, "max", max);
        GetNodeAttribute(fill_ratio_node, "steps", steps);

        sim_data_set_.tfr_range_ = GenerateLinspace(min, max, steps);

        // Grab sensor probability ranges
        TConfigurationNode &sensor_probability_node = GetNode(col_per_root_node, "sensor_probability_range");

        GetNodeAttribute(sensor_probability_node, "min", min);
        GetNodeAttribute(sensor_probability_node, "max", max);
        GetNodeAttribute(sensor_probability_node, "steps", steps);

        sim_data_set_.sp_range_ = GenerateLinspace(min, max, steps);

        // Create pairings for target fill ratios and sensor probabilities
        for (const float &tfr : sim_data_set_.tfr_range_)
        {
            for (const float &sp : sim_data_set_.sp_range_)
            {
                tfr_sp_ranges_.push_back(std::pair<float, float>(tfr, sp));
            }
        }

        curr_tfr_sp_range_itr_ = tfr_sp_ranges_.begin();

        // Grab number of agents and communications range
        auto &rab_map = space_entity.GetEntitiesByType("rab");
        CRABEquippedEntity &random_rab = *any_cast<CRABEquippedEntity *>(rab_map.begin()->second);

        sim_data_set_.num_agents_ = rab_map.size();         // the number of range and bearing sensors is the same as the number of robots
        sim_data_set_.comms_range_ = random_rab.GetRange(); // all the range and bearing sensors have the same range

        // Grab number of trials
        GetNodeAttribute(GetNode(col_per_root_node, "num_trials"), "value", sim_data_set_.num_trials_);

        // Grab number of steps
        sim_data_set_.num_steps_ = GetSimulator().GetMaxSimulationClock();
    }
    catch (CARGoSException &ex)
    {
        THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!", ex);
    }

    // Initialize each robot
    initialization_functor_ = InitializeRobot(id_brain_map_ptr_, curr_tfr_sp_range_itr_->second);

    LOG << "[INFO] Running trial " << trial_counter_ + 1 << " with new parameters." << std::endl;
    LOG << "[INFO] Target fill ratio = " << curr_tfr_sp_range_itr_->first << std::endl;
    LOG << "[INFO] Sensor probability = " << curr_tfr_sp_range_itr_->second << std::endl;

    SetupExperiment();
}

void CollectivePerceptionLoopFunctions::SetupExperiment()
{
    // Create new Arena object
    arena_ = Arena(arena_tile_count_, arena_lower_lim_, arena_tile_size_, curr_tfr_sp_range_itr_->first);

    LOG << "[INFO] Arena tile fill ratio = " << arena_.GetTrueTileDistribution() << " with " << arena_.GetTotalNumTiles() << " tiles." << std::endl;

    // Re-initialize each robot
    BuzzForeachVM(initialization_functor_);
}

void CollectivePerceptionLoopFunctions::PostStep()
{
    // Iterate through each brain to process 'thought'
    BuzzForeachVM(ProcessRobotThought(id_brain_map_ptr_));

    // Log brain information
}

void CollectivePerceptionLoopFunctions::PostExperiment()
{
    // Check to see if all trials are complete
    if (++trial_counter_ % sim_data_set_.num_trials_ == 0)
    {
        trial_counter_ = 0; // reset counter

        ++curr_tfr_sp_range_itr_; // use next parameter set

        // Check to see if there are any parameter sets left
        if (curr_tfr_sp_range_itr_ != tfr_sp_ranges_.end())
        {
            initialization_functor_.UpdateSensorProbability(curr_tfr_sp_range_itr_->second);

            LOG << "[INFO] Running trial " << trial_counter_ + 1 << " with new parameters." << std::endl;
            LOG << "[INFO] Target fill ratio = " << curr_tfr_sp_range_itr_->first << std::endl;
            LOG << "[INFO] Sensor probability = " << curr_tfr_sp_range_itr_->second << std::endl;
        }
        else
        {
            finished_ = true;
        }
    }
    else
    {
        // Repeat trial
        LOG << "[INFO] Running trial " << trial_counter_ + 1 << " with same parameters." << std::endl;
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