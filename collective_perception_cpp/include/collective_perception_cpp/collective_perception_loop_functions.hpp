#ifndef COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP
#define COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP

#include <vector>
#include <unordered_map>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <buzz/buzzvm.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/plugins/simulator/entities/rab_equipped_entity.h>

// Local headers
#include "arena.hpp"
#include "brain.hpp"
#include "simulation_data_set.hpp"

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
     * @param ptr Pointer to the unordered map to be populated with robot IDs and Brain instances
     * @param sensor_prob Initial sensor probability to be assigned to robots
     */
    InitializeRobot(const std::shared_ptr<RobotIdBrainMap> &ptr, const float &sensor_prob)
        : id_brain_map_ptr((ptr)), b_prob(sensor_prob), w_prob(sensor_prob) {}

    /**
     * @brief Overload the () operator (used to initialize each robot using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    /**
     * @brief Update the sensor probabilities
     *
     * @param new_prob New probability value
     */
    inline void UpdateSensorProbability(const float &new_prob)
    {
        b_prob = new_prob;
        w_prob = new_prob;
    }

    float b_prob;

    float w_prob;

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances
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
     * @param ptr Pointer to the unordered map containing robot IDs and corresponding Brain instances
     */
    ProcessRobotThought(const std::shared_ptr<RobotIdBrainMap> &ptr) : id_brain_map_ptr(ptr) {}

    /**
     * @brief Overload the () operator (used to process each robot's values using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances
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

    virtual void PostExperiment();

    inline bool IsExperimentFinished() { return finished_; }

private:
    void ComputeStats();

    void SetupExperiment();

    void CreateNewSimPacket();

    void PopulateSimPacket();

    std::vector<Brain::ValuePair> GetAllLocalValues();

    std::vector<Brain::ValuePair> GetAllSocialValues();

    std::vector<Brain::ValuePair> GetAllInformedValues();

    std::pair<Brain::ValuePair, Brain::ValuePair> ComputeValuePairsSampleMeanAndStdDev(const std::vector<Brain::ValuePair> &input);

    template <typename T>
    std::vector<T> GenerateLinspace(const T &min, const T &max, const size_t &steps);

    Arena arena_; ///< Arena object

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr_ = std::make_shared<RobotIdBrainMap>(); ///< Pointer to unordered map containing robot IDs and Brain instances

    CFloorEntity *floor_entity_ptr_ = NULL; ///< Pointer to the floor entity class

    SimulationDataSet sim_data_set_;

    bool finished_ = false;

    int trial_counter_ = 0;

    float arena_tile_size_;

    std::pair<uint32_t, uint32_t> arena_tile_count_;

    std::pair<float, float> arena_lower_lim_;

    std::vector<std::pair<float, float>> tfr_sp_ranges_;

    std::vector<std::pair<float, float>>::iterator curr_tfr_sp_range_itr_;

    InitializeRobot initialization_functor_;

    SimPacket curr_sim_packet_;

    Stats curr_local_stats_;

    Stats curr_social_stats_;

    Stats curr_informed_stats_;
};

#endif