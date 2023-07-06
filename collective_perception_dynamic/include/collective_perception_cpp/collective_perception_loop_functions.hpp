#ifndef COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP
#define COLLECTIVE_PERCEPTION_LOOP_FUNCTIONS_HPP

#include <vector>
#include <unordered_map>
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
#ifdef ARGOS_EMANE
#include <argos3/plugins/simulator/entities/emane_equipped_entity.h>
#endif
#include <argos3/plugins/simulator/entities/box_entity.h>

// Local headers
#include "util.hpp"
#include "arena.hpp"
#include "brain.hpp"
#include "simulation_set.hpp"
#include "simulation_stats_set.hpp"
#include "simulation_agent_data_set.hpp"
#include "dac_plugin.hpp"
#include "robot_disability.hpp"

using namespace argos;

using RobotIdBrainMap = std::unordered_map<std::string, Brain>;

/**
 * @brief Functor to initialize robots
 *
 */
struct InitializeRobot : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new InitializeRobot functor
     *
     */
    InitializeRobot() : id_brain_map_ptr(nullptr), str_int_id_map_ptr(nullptr) {}

    /**
     * @brief Construct a new InitializeRobot functor
     *
     * @param id_brain_ptr Pointer to the unordered map to be populated with robot IDs and Brain instances
     * @param str_id_ptr Pointer to the unordered map to be populated with robot IDs and their corresponding integer indices
     * @param sensor_prob Initial sensor probability to be assigned to robots
     * @param spd Speed to be assigned to robots
     * @param legacy Flag to use legacy equations
     */
    inline InitializeRobot(const std::shared_ptr<RobotIdBrainMap> &id_brain_ptr,
                           const std::shared_ptr<std::unordered_map<std::string, int>> &str_id_ptr,
                           const double &sensor_prob,
                           const float &spd,
                           const bool &legacy)
        : id_brain_map_ptr(id_brain_ptr),
          str_int_id_map_ptr(str_id_ptr),
          b_prob(sensor_prob),
          w_prob(sensor_prob),
          spd(spd),
          legacy(legacy)
    {
        // Initialize generator
        std::random_device rd;
        generator = std::default_random_engine(rd());
        internal_counter = 0;
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

    int internal_counter; ///< Internal counter to assign indices based on robot string IDs

    std::default_random_engine generator; ///< Random number generator

    std::shared_ptr<std::unordered_map<std::string, int>> str_int_id_map_ptr; ///< Pointer to unordered map of string robot IDs to indices (so that vectors can store their corresponding data)

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances
};

/**
 * @brief Functor to process robot's estimates and confidences
 *
 */
struct ProcessRobotThought : public CBuzzLoopFunctions::COperation
{
    /**
     * @brief Construct a new ProcessRobotThought functor
     *
     */
    ProcessRobotThought() : id_brain_map_ptr(nullptr), str_int_id_map_ptr(nullptr), agt_data_vec_ptr(nullptr) {}

    /**
     * @brief Construct a new ProcessRobotThought functor
     *
     * @param id_brain_ptr Pointer to the unordered map connecting robot IDs and Brain instances
     * @param str_id_ptr Pointer to the unordered map to connecting robot IDs and their corresponding integer indices
     * @param agt_vec_ptr Pointer to the vector of AgentData instances
     * @param disabled_ids Vector containing robot IDs that are meant to be disabled
     * @param disability_type_map Unordered map connecting disability types and the activation flags
     */
    ProcessRobotThought(const std::shared_ptr<RobotIdBrainMap> &id_brain_ptr,
                        const std::shared_ptr<std::unordered_map<std::string, int>> &str_id_ptr,
                        std::vector<AgentData> *agt_vec_ptr,
                        const std::vector<std::pair<std::string, int>> &disabled_ids,
                        const std::unordered_map<DisabilityType, bool> &disability_type_map);

    /**
     * @brief Struct to store a single robot's disability status and types
     *
     */
    struct DisabilityStatusAndTypes
    {
        bool disability_activated;

        std::vector<DisabilityType> disability_types;
    };

    /**
     * @brief Overload the () operator (used to process each robot's values using BuzzForeachVM())
     *
     * @param str_robot_id Robot ID
     * @param t_vm Buzz VM
     */
    virtual void operator()(const std::string &str_robot_id, buzzvm_t t_vm);

    /**
     * @brief Check whether robot is supposed to have any disability (can be used even before disability activates)
     *
     * @param robot_id Robot ID string
     * @return true
     * @return false
     */
    inline bool HasDisability(const std::string &robot_id)
    {
        return id_disabled_status_map.find(robot_id.c_str()) != id_disabled_status_map.end();
    }

    /**
     * @brief Store observations into robot brain
     *
     * @param str_robot_id Robot ID string
     * @param t_vm Buzz VM object
     * @param robot_brain Robot brain instance corresponding to the current robot
     */
    void StoreObservations(const std::string &str_robot_id, buzzvm_t t_vm, Brain &robot_brain);

    /**
     * @brief Store the neighbor local values into robot brain
     *
     * @param str_robot_id Robot ID string
     * @param t_vm Buzz VM object
     * @param robot_brain Robot brain instance corresponding to the current robot
     */
    void StoreNeighborValues(const std::string &str_robot_id, buzzvm_t t_vm, Brain &robot_brain);

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr; ///< Pointer to unordered map containing robot IDs and corresponding Brain instances

    std::shared_ptr<std::unordered_map<std::string, int>> str_int_id_map_ptr; ///< Pointer to unordered map of string robot IDs to indices (so that vectors can store their corresponding data)

    std::vector<AgentData> *agt_data_vec_ptr; ///< Pointer to vector of agent data

    SwarmDisabilityStatus disability_status = SwarmDisabilityStatus::inactive;

    std::unordered_map<std::string, DisabilityStatusAndTypes> id_disabled_status_map; ///< Map that contains the IDs that need to be disabled and whether or not they're disabled
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
     * @brief Collect robot estimates and confidences
     *
     * @return std::array<std::vector<Brain::ValuePair>, 3> STL array of 3 vector of ValuePair objects; local, social, and informed respectively
     */
    std::array<std::vector<Brain::ValuePair>, 3> GetAllSolverValues();

    /**
     * @brief Compute the sample statistics
     *
     * @param input Values to compute with
     * @return std::pair<Brain::ValuePair, Brain::ValuePair> STL pair containing sample statistics
     */
    std::pair<Brain::ValuePair, Brain::ValuePair> ComputeValuePairsSampleMeanAndStdDev(const std::vector<Brain::ValuePair> &input);

    bool finished_ = false; ///< Flag to indicate whether all simulation parameters have been executed

    bool legacy_; ///< Flag to activate usage of legacy social estimate computation

    bool proto_datetime_; ///< Flag to enable datetime in output data filename

    bool run_dac_plugin_; ///< Flag to enable usage of the DAC plugin

    int trial_counter_ = 0; ///< Counter to keep track of trials

    int disabled_time_in_ticks_; ///< Simulation time in ticks that robots are disabled

    int dac_plugin_write_period_in_ticks_; ///< Write period to DAC-SysML CSV file in ticks

    unsigned int disabled_robot_amount_; ///< Number of robots that are/should be disabled

    float arena_tile_size_; ///< Size of an arena tile (value is length, since the tile is square)

    float ticks_per_sec_; ///< Number of ticks in one second

    std::string verbose_level_; ///< Output verbosity level

    std::string output_folder_; ///< Folder to output data to

    std::pair<unsigned int, unsigned int> arena_tile_count_; ///< Tile count in x and y directions

    std::pair<float, float> arena_lower_lim_; ///< Coordinates to the bottom left corner of the arena

    std::unordered_map<DisabilityType, bool> robot_disability_types_; ///< Map to indicate the robots' disability type

    std::vector<std::pair<std::string, int>> disabled_ids_; ///< Vector of robot IDs that are meant to be disabled; the IDs are paired with their contiguous integer indices

    std::vector<std::pair<double, double>> tfr_sp_ranges_; ///< Target fill ratio and sensor probability pairs

    std::vector<std::pair<double, double>>::iterator curr_tfr_sp_range_itr_; ///< Current target fill ratio and sensor probability configuration

    std::shared_ptr<RobotIdBrainMap> id_brain_map_ptr_ = std::make_shared<RobotIdBrainMap>(); ///< Pointer to unordered map containing robot IDs and Brain instances

    std::shared_ptr<std::unordered_map<std::string, int>> str_int_id_map_ptr_ = std::make_shared<std::unordered_map<std::string, int>>(); ///< Pointer to unordered map of string robot IDs to indices (so that vectors can store their corresponding data)

    Arena arena_; ///< Arena object

    DACPlugin dac_plugin_; ///< DAC plugin object

    CFloorEntity *floor_entity_ptr_ = NULL; ///< Pointer to the floor entity class

    CSpace *space_ptr_ = NULL; ///< Pointer to the space class

    SimulationSet simulation_parameters_; ///< SimulationSet object containing simulation parameters

    SimulationStatsSet sim_stats_set_; ///< SimulationStatsSet object containing simulation statistics

    SimulationAgentDataSet sim_agent_data_set_; ///< SimulationAgentDataSet object containing individual agent data

    InitializeRobot initialization_functor_; ///< Initialization functor to initialize Buzz controlled robots

    ProcessRobotThought process_thought_functor_; ///< ProcessRobotThought functor to process data of Buzz controlled robots

    StatsPacket curr_stats_packet_; ///< Current trial statistics

    AgentDataPacket curr_agent_data_packet_; ///< Current trial agent data
};

#endif