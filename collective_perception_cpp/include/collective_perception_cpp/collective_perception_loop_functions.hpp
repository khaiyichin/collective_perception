#ifndef COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP
#define COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP

#include <vector>
#include <unordered_map>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <buzz/buzzvm.h>

// Local headers
#include "arena.hpp"
#include "brain.hpp"

using namespace argos;

/**
 * @brief Class to implement loop functions for the collective perception simulation
 *
 */
class CollectivePerceptionLoopFunctions : public CBuzzLoopFunctions
{
public:
    /**
     * @brief Construct a new CollectivePerceptionLoopFunctions object
     *
     */
    CollectivePerceptionLoopFunctions() {}

    /**
     * @brief Destroy the CollectivePerceptionLoopFunctions object
     *
     */
    virtual ~CollectivePerceptionLoopFunctions() {}

    /**
     * @brief Initialize loop functions
     *
     * @param t_tree Pointer to the XML config node
     */
    virtual void Init(TConfigurationNode &t_tree);

    /**
     * @brief Reset loop functions (triggered by simulation reset)
     *
     */
    virtual void Reset();

    /**
     * @brief Get the floor color
     *
     * @param c_position_on_plane Coordinates of the floor
     * @return CColor Color at the specified coordinates
     */
    virtual CColor GetFloorColor(const CVector2 &c_position_on_plane);

    /**
     * @brief Execute post step activities
     *
     */
    virtual void PostStep();

private:
    Arena arena_; ///< Arena object

    std::unordered_map<std::string, Brain> id_brain_map_; ///< Unoredered map for robot ID and brain instance

    CFloorEntity *floor_entity_ptr_ = NULL; ///< Pointer to the floor entity class
};

/**
 * @brief Functor to initialize robots
 *
 */
struct InitializeRobot : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new InitializeRobot struct
     *
     * @param id_brain_map_ref An empty unordered map to be populated with robot IDs and Brain instances
     */
    InitializeRobot(std::unordered_map<std::string, Brain> &id_brain_map_ref) : id_brain_map(id_brain_map_ref) {}

    /**
     * @brief Overload the () operator (used to initialize each robot using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    std::unordered_map<std::string, Brain> &id_brain_map; ///< Unordered map containing robot IDs and corresponding Brain instances
};

/**
 * @brief Functor to process robot's estimates and confidences
 *
 */
struct ProcessRobotThought : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new ProcessRobotThought struct
     *
     * @param id_brain_map_ref Unordered map containing robot IDs and corresponding Brain instances
     */
    ProcessRobotThought(std::unordered_map<std::string, Brain> &id_brain_map_ref) : id_brain_map(id_brain_map_ref) {}

    /**
     * @brief Overload the () operator (used to process each robot's values using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    std::unordered_map<std::string, Brain> &id_brain_map; ///< Unordered map containing robot IDs and corresponding Brain instances
};

#endif