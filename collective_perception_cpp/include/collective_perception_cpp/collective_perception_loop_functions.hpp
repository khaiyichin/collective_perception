#ifndef COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP
#define COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP

#include <vector>
#include <unordered_map>
#include <ctime>
#include <cmath>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <buzz/buzzvm.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/plugins/simulator/entities/rab_equipped_entity.h>

// Local headers
#include "arena.hpp"
#include "brain.hpp"
#include "simulation_set.hpp"
#include "simulation_stats_set.hpp"
#include "simulation_agent_data_set.hpp"

using namespace argos;

using RobotIdBrainMap = std::unordered_map<std::string, Brain>;

/**
 * @brief Functor to initialize robots
 *
 */
struct InitializeRobot : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new InitializeRobot struct
     *
     */
    InitializeRobot() : id_brain_map_ptr(nullptr) {}

    /**
     * @brief Construct a new InitializeRobot struct
     *
     * @param id_brain_ptr Pointer to the unordered map to be populated with robot IDs and Brain instances
     * @param sensor_prob Initial sensor probability to be assigned to robots
     * @param spd Speed to be assigned to robots
     */
    InitializeRobot(const std::shared_ptr<RobotIdBrainMap> &id_brain_ptr, const float &sensor_prob, const float &spd)
        : id_brain_map_ptr(id_brain_ptr), b_prob(sensor_prob), w_prob(sensor_prob), spd(spd) {}

    /**
     * @brief Overload the () operator (used to initialize each robot using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    float b_prob;

    float w_prob;

    float spd;

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances
};

/**
 * @brief Functor to process robot's estimates and confidences
 *
 */
struct ProcessRobotThought : public CBuzzLoopFunctions::COperation
{
    ProcessRobotThought() : id_brain_map_ptr(nullptr), agt_data_vec_ptr(nullptr) {}

    /**
     * @brief Construct a new Process Robot Thought object
     *
     * @param id_brain_ptr Pointer to the unordered map to be populated with robot IDs and Brain instances
     * @param agt_vec_ptr Pointer to the vector of AgentData instances
     * @param id_prefix ID prefix used to strip the robot ID integer value
     * @param id_base_num ID base number used to offset the robot ID integer value
     */
    ProcessRobotThought(const std::shared_ptr<RobotIdBrainMap> &id_brain_ptr,
                        std::vector<AgentData> *agt_vec_ptr,
                        const std::string &id_prefix, const int &id_base_num)
        : id_brain_map_ptr(id_brain_ptr), agt_data_vec_ptr(agt_vec_ptr), prefix(id_prefix), base_num(id_base_num) {}

    /**
     * @brief Overload the () operator (used to process each robot's values using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances

    std::vector<AgentData> *agt_data_vec_ptr; ///< Pointer to vector of agent data

    std::string prefix; ///< Prefix to robot ID

    int base_num; ///< Base number offset for robot ID
};

/**
 * @brief Class to implement loop functions for the collective perception simulation
 *
 */
class CollectivePerceptionLoopFunctions : public CBuzzLoopFunctions
{
public:
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
    inline void Reset() { SetupExperiment(); }

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

    /**
     * @brief Execute post experiment activities
     *
     */
    virtual void PostExperiment();

    /**
     * @brief Check if experiment is over
     *
     * @return true
     * @return false
     */
    inline bool IsExperimentFinished() { return finished_; }

private:
    void ComputeStats();

    void SetupExperiment();

    void CreateNewPacket();

    std::array<std::vector<Brain::ValuePair>, 3> GetAllSolverValues();

    std::pair<Brain::ValuePair, Brain::ValuePair> ComputeValuePairsSampleMeanAndStdDev(const std::vector<Brain::ValuePair> &input);

    template <typename T>
    std::vector<T> GenerateLinspace(const T &min, const T &max, const size_t &steps);

    bool finished_ = false;

    bool proto_datetime_;

    int trial_counter_ = 0;

    int id_base_num_;

    float arena_tile_size_;

    std::string verbose_level_;

    std::string id_prefix_;

    std::pair<uint32_t, uint32_t> arena_tile_count_;

    std::pair<float, float> arena_lower_lim_;

    std::vector<std::pair<float, float>> tfr_sp_ranges_;

    std::vector<std::pair<float, float>>::iterator curr_tfr_sp_range_itr_;

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr_ = std::make_shared<RobotIdBrainMap>(); ///< Pointer to unordered map containing robot IDs and Brain instances

    Arena arena_; ///< Arena object

    CFloorEntity *floor_entity_ptr_ = NULL; ///< Pointer to the floor entity class

    SimulationSet simulation_parameters_;

    SimulationStatsSet sim_stats_set_;

    SimulationAgentDataSet sim_agent_data_set_;

    InitializeRobot initialization_functor_;

    ProcessRobotThought process_thought_functor_;

    StatsPacket curr_stats_packet_;

    AgentDataPacket curr_agent_data_packet_;
};

#endif