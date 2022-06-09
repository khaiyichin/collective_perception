#include "collective_perception_loop_functions.hpp"

void CollectivePerceptionLoopFunctions::Init(TConfigurationNode &t_tree)
{
    // Call parent's Init
    CBuzzLoopFunctions::Init(t_tree);

    // Extract experiment information from the loop_functions tag in the ARGoS XML file
    // like what?

    // Extract floor information
    try
    {
        // Grab the reference to the XML node with the tag "floor_loop_functions"
        TConfigurationNode &tFloor = GetNode(t_tree, "collective_perception");

        // Get a pointer to the ARGoS floor entity (method provided by superclass)
        CSpace &space_entity = GetSpace();

        floor_entity_ptr_ = &space_entity.GetFloorEntity();

        float fill_ratio;
        uint32_t arena_x, arena_y;

        // Get the size of the arena (in units of tiles)
        GetNodeAttribute(tFloor, "arena_tile_count_x", arena_x);
        GetNodeAttribute(tFloor, "arena_tile_count_y", arena_y);
        std::vector<uint32_t> tile_count{arena_x, arena_y};

        // Get the limits of arena
        CRange<CVector3> lims = space_entity.GetArenaLimits();
        std::vector<float> lower_lim{static_cast<float>(lims.GetMin().GetX()), static_cast<float>(lims.GetMin().GetY())};

        // Get the fill ratio
        GetNodeAttribute(tFloor, "fill_ratio", fill_ratio);

        // Compute tile size
        CVector3 arena_size = space_entity.GetArenaSize();
        float length_x = arena_size.GetX() / arena_x;
        float length_y = arena_size.GetY() / arena_y;

        assert(length_x == length_y); // only square tiles allowed

        LOG << "[INFO] Generated tile size of " << length_x << "m" << std::endl;

        // Create Arena object
        arena_ = Arena(tile_count, lower_lim, length_x, fill_ratio);

        LOG << "[INFO] Arena tile fill ratio = " << arena_.GetTrueTileDistribution() << " with " << arena_.GetTotalNumTiles() << " tiles." << std::endl;
    }
    catch (CARGoSException &ex)
    {
        THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!", ex);
    }

    // Initialize each robot
    BuzzForeachVM(InitializeRobot(id_brain_map_));
}

void CollectivePerceptionLoopFunctions::Reset()
{
    // Reset tiles
    arena_.GenerateTileArrangement();

    LOG << "[INFO] Arena tile fill ratio = " << arena_.GetTrueTileDistribution() << " with " << arena_.GetTotalNumTiles() << " tiles." << std::endl;
}

void CollectivePerceptionLoopFunctions::PostStep()
{
    // Iterate through each brain to process 'thought'
    BuzzForeachVM(ProcessRobotThought(id_brain_map_));

    // Log brain information
}

CColor CollectivePerceptionLoopFunctions::GetFloorColor(const CVector2 &c_position_on_plane)
{
    uint32_t color_int = arena_.GetColor(c_position_on_plane.GetX(), c_position_on_plane.GetY());

    return color_int == 1 ? CColor::BLACK : CColor::WHITE;
}

void InitializeRobot::operator()(const std::string &str_robot_id, buzzvm_t t_vm)
{
    // Assign sensor accuracy
    float b_acc = 0.95; // should be obtained from ARGoS XML file
    float w_acc = 0.95;

    BuzzPut(t_vm, "b_prob", b_acc);
    BuzzPut(t_vm, "w_prob", w_acc);

    // Use robot ID and sensor accuracy to instantiate Brain class
    id_brain_map[str_robot_id.c_str()] = Brain(str_robot_id, b_acc, w_acc);
    auto &robot_brain = id_brain_map[str_robot_id.c_str()];

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
    auto &robot_brain = id_brain_map[str_robot_id.c_str()];

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

REGISTER_LOOP_FUNCTIONS(CollectivePerceptionLoopFunctions, "collective_perception_loop_functions")