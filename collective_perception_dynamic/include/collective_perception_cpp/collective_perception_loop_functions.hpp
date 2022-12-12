#ifndef COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP
#define COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP

#include <vector>
#include <unordered_map>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>

// Buzz and ARGoS headers
#include <buzz/argos/buzz_loop_functions.h>
#include <buzz/buzzvm.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/plugins/simulator/entities/rab_equipped_entity.h>
#include <argos3/plugins/simulator/entities/box_entity.h>

// Local headers
#include "arena.hpp"
#include "brain.hpp"
#include "simulation_set.hpp"
#include "simulation_stats_set.hpp"
#include "simulation_agent_data_set.hpp"
#include "dac_plugin.hpp"

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
    inline InitializeRobot(const std::shared_ptr<RobotIdBrainMap> &id_brain_ptr, const double &sensor_prob, const float &spd, const bool &legacy)
        : id_brain_map_ptr(id_brain_ptr), b_prob(sensor_prob), w_prob(sensor_prob), spd(spd), legacy(legacy)
    {
        // Initialize generator
        std::random_device rd;
        generator = std::default_random_engine(rd());
    }

    /**
     * @brief Overload the () operator (used to initialize each robot using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    /**
     * @brief Generate a sensor probability at random based on a probability distribution
     *
     * @return float The sensor probability
     */
    float GenerateRandomSensorProbability();

    bool legacy; ///< Flag to indicate whether legacy equations are being used

    double b_prob; ///< Sensor quality in identifying black tiles

    double w_prob; ///< Sensor quality in identifying white tiles

    float spd; ///< Robot speed

    std::default_random_engine generator; ///< Random number generator

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances
};

/**
 * @brief Functor to process robot's estimates and confidences
 *
 */
struct ProcessRobotThought : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new ProcessRobotThought object
     *
     */
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
                        const std::string &id_prefix,
                        const int &id_base_num,
                        const std::vector<int> &disabled_ids);

    /**
     * @brief Overload the () operator (used to process each robot's values using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    /**
     * @brief Strips the string prefix in the robot ID to obtain the numeric part
     *
     * @param str_robot_id Robot ID
     * @return int Numeric part of the robot ID
     */
    inline int GetNumericId(std::string str_robot_id)
    {
        return std::stoi(str_robot_id.erase(0, prefix.length()));
    }

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances

    std::vector<AgentData> *agt_data_vec_ptr; ///< Pointer to vector of agent data

    std::string prefix; ///< Prefix to robot ID

    int base_num; ///< Base number offset for robot ID

    bool disabled = false; ///< Flag to activate disabling of robots

    std::unordered_map<int, bool> id_disabled_status_map; ///< Map that contains the IDs that need to be disabled and whether or not they're disabled
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
    /**
     * @brief Compute the swarm statistics
     *
     */
    void ComputeStats();

    /**
     * @brief Setup experiment
     *
     */
    void SetupExperiment();

    /**
     * @brief Create a new data packet objects
     *
     */
    void CreateNewPacket();

    /**
     * @brief Write data to disk
     *
     */
    void SaveData();

    /**
     * @brief Randomly draw robots to disable
     *
     */
    void SampleRobotsToDisable();

    /**
     * @brief Get the current time in string
     *
     * @return std::string in mmddyy_HHMMSS form
     */
    std::string GetCurrentTimeStr();

    /**
     * @brief Collect robot estimates and confidences
     *
     * @return std::array<std::vector<Brain::ValuePair>, 3> STL array of 3 vector of ValuePair objects; local, social, and informed respectively
     */
    std::array<std::vector<Brain::ValuePair>, 3> GetAllSolverValues();

    /**
     * @brief Compute the sample statistics
     *
     * @param input
     * @return std::pair<Brain::ValuePair, Brain::ValuePair> STL pair containing sample statistics
     */
    std::pair<Brain::ValuePair, Brain::ValuePair> ComputeValuePairsSampleMeanAndStdDev(const std::vector<Brain::ValuePair> &input);

    /**
     * @brief Generate a linearly spaced range of values
     *
     * @tparam T Any numeric type
     * @param min Minimum value of range
     * @param max Maximum value of range
     * @param steps Number of increments to take in range
     * @return std::vector<T> STL vector containing the linearly spaced values
     */
    template <typename T>
    std::vector<T> GenerateLinspace(const T &min, const T &max, const size_t &steps);

    bool finished_ = false; ///< Flag to indicate whether all simulation parameters have been executed

    bool legacy_; ///< Flag to activate usage of legacy social estimate computation

    bool proto_datetime_; ///< Flag to enable datetime in output data filename

    bool run_dac_plugin_; ///< Flag to enable usage of the DAC plugin

    int trial_counter_ = 0; ///< Counter to keep track of trials

    int id_base_num_;

    int disabled_time_in_ticks_; ///< Simulation time in ticks that robots are disabled

    int dac_plugin_write_period_in_ticks_; ///< Write period to DAC-SysML CSV file in ticks

    unsigned int disabled_robot_amount_; ///< Number of robots that are/should be disabled

    float arena_tile_size_; ///< Size of an arena tile (value is length, since the tile is square)

    float ticks_per_sec_; ///< Number of ticks in one second

    std::string verbose_level_; ///< Output verbosity level

    std::string id_prefix_; ///< Prefix in the robot IDs

    std::string output_folder_; ///< Folder to output data to

    std::pair<unsigned int, unsigned int> arena_tile_count_; ///< Tile count in x and y directions

    std::pair<float, float> arena_lower_lim_; ///< Coordinates to the bottom left corner of the arena

    std::vector<int> disabled_ids_;

    std::vector<std::pair<double, double>> tfr_sp_ranges_;

    std::vector<std::pair<double, double>>::iterator curr_tfr_sp_range_itr_;

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr_ = std::make_shared<RobotIdBrainMap>(); ///< Pointer to unordered map containing robot IDs and Brain instances

    Arena arena_; ///< Arena object

    DACPlugin dac_plugin_; ///< DAC plugin object

    CFloorEntity *floor_entity_ptr_ = NULL; ///< Pointer to the floor entity class

    CSpace *space_ptr_ = NULL; ///< Pointer to the space class

    SimulationSet simulation_parameters_;

    SimulationStatsSet sim_stats_set_;

    SimulationAgentDataSet sim_agent_data_set_;

    InitializeRobot initialization_functor_;

    ProcessRobotThought process_thought_functor_;

    StatsPacket curr_stats_packet_;

    AgentDataPacket curr_agent_data_packet_;
};

#endif