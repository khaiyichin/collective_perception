#include "collective_perception_loop_functions.hpp"

CCollectivePerceptionLoopFunctions::CCollectivePerceptionLoopFunctions() : m_pcFloor_(NULL) {}

void CCollectivePerceptionLoopFunctions::Init(TConfigurationNode &t_tree)
{
    try
    {
        // Grab the reference to the XML node with the tag "floor_loop_functions"
        TConfigurationNode &tFloor = GetNode(t_tree, "collective_perception");

        // Get a pointer to the ARGoS floor entity (method provided by superclass)
        CSpace &space_entity = GetSpace();

        m_pcFloor_ = &space_entity.GetFloorEntity();

        float fill_ratio;
        uint32_t arena_x, arena_y;

        // Get the size of the arena (in units of tiles)
        GetNodeAttribute(tFloor, "arena_tile_count_x", arena_x);
        GetNodeAttribute(tFloor, "arena_tile_count_y", arena_y);
        std::vector<uint32_t> tile_count{arena_x, arena_y};

        // Get the limits of arena
        CRange<CVector3> lims = space_entity.GetArenaLimits();
        std::vector<float> lower_lim{lims.GetMin().GetX(), lims.GetMin().GetY()};

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
    }
    catch (CARGoSException &ex)
    {
        THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!", ex);
    }
}

CColor CCollectivePerceptionLoopFunctions::GetFloorColor(const CVector2 &c_position_on_plane)
{
    uint32_t color_int = arena_.GetColor( static_cast<float>( c_position_on_plane.GetX() ), static_cast<float>( c_position_on_plane.GetY() ) );

    return color_int == 1 ? CColor::BLACK : CColor::WHITE;
}

REGISTER_LOOP_FUNCTIONS(CCollectivePerceptionLoopFunctions, "collective_perception_loop_functions")