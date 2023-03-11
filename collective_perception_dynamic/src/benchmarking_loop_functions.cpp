#include "benchmarking_loop_functions.hpp"

void BenchmarkingLoopFunctions::Init(TConfigurationNode &t_tree)
{
    // Extract XML information
    try
    {
        // Call parent's Init
        CBuzzLoopFunctions::Init(t_tree);

        // Grab the reference to the XML node with the tag "benchmarking"
        TConfigurationNode &benchmarking_root_node = GetNode(t_tree, "benchmarking");

        // Grab algorithm information
        TConfigurationNode &algorithm_node = GetNode(benchmarking_root_node, "algorithm");

        GetNodeAttribute(algorithm_node, "type", algorithm_str_id_);

        // Initialize benchmark algorithm
        InitializeBenchmarkAlgorithm(algorithm_node);

        // Obtain the pointer to the benchmark data
        benchmark_data_ptr_ = std::make_shared<BenchmarkDataBase>(std::move(benchmark_algo_ptr_->GetData()));

        // Grab the constrained area to compute the true swarm density
        auto &box_map = GetSpace().GetEntitiesByType("box");

        // Get constrained x length
        CBoxEntity &wall_west = *any_cast<CBoxEntity *>(box_map["wall_west"]);
        CBoxEntity &wall_east = *any_cast<CBoxEntity *>(box_map["wall_east"]);

        float wall_west_thickness = wall_west.GetSize().GetX();
        float wall_east_thickness = wall_east.GetSize().GetX();

        float wall_west_pos_x = wall_west.GetEmbodiedEntity().GetOriginAnchor().Position.GetX();
        float wall_east_pos_x = wall_east.GetEmbodiedEntity().GetOriginAnchor().Position.GetX();

        assert(abs(wall_west_pos_x) == abs(wall_east_pos_x));         // ensure that walls are evenly separated
        assert(abs(wall_west_thickness) == abs(wall_east_thickness)); // ensure that walls are equally thick

        // Get constrained y length
        CBoxEntity &wall_north = *any_cast<CBoxEntity *>(box_map["wall_north"]);
        CBoxEntity &wall_south = *any_cast<CBoxEntity *>(box_map["wall_south"]);

        float wall_north_thickness = wall_north.GetSize().GetY();
        float wall_south_thickness = wall_south.GetSize().GetY();

        float wall_north_pos_y = wall_north.GetEmbodiedEntity().GetOriginAnchor().Position.GetY();
        float wall_south_pos_y = wall_south.GetEmbodiedEntity().GetOriginAnchor().Position.GetY();

        assert(abs(wall_north_pos_y) == abs(wall_south_pos_y));         // ensure that walls are evenly separated
        assert(abs(wall_north_thickness) == abs(wall_south_thickness)); // ensure that walls are equally thick

        // Compute constrained arena area
        float constrained_x_distance = (wall_east_pos_x - wall_west_pos_x) - wall_west_thickness;
        float constrained_y_distance = (wall_north_pos_y - wall_south_pos_y) - wall_north_thickness;

        float constrained_area = constrained_x_distance * constrained_y_distance;

        // Grab fill ratio range
        TConfigurationNode &fill_ratio_node = GetNode(benchmarking_root_node, "fill_ratio_range");

        double min, max;
        int steps;

        GetNodeAttribute(fill_ratio_node, "min", min);
        GetNodeAttribute(fill_ratio_node, "max", max);
        GetNodeAttribute(fill_ratio_node, "steps", steps);

        benchmark_data_ptr_->tfr_range = GenerateLinspace(min, max, steps);

        // Create pairings for target fill ratios and sensor probabilities
        std::vector<double> parameter_1_range = benchmark_data_ptr_->tfr_range;
        std::vector<double> parameter_2_range = benchmark_algo_ptr_->GetParameterRange();

        for (const double &p1 : parameter_1_range)
        {
            for (const double &p2 : parameter_2_range)
            {
                paired_parameter_ranges_.push_back(std::pair<double, double>(p1, p2));
            }
        }

        curr_paired_parameter_range_itr_ = paired_parameter_ranges_.begin();

        // Grab robot speeds
        GetNodeAttribute(GetNode(benchmarking_root_node, "speed"), "value", benchmark_data_ptr_->speed);

        // Grab number of agents and communications range
        auto &rab_map = GetSpace().GetEntitiesByType("rab");
        CRABEquippedEntity &random_rab = *any_cast<CRABEquippedEntity *>(rab_map.begin()->second);

        benchmark_data_ptr_->num_agents = rab_map.size();         // the number of range and bearing sensors is the same as the number of robots
        benchmark_data_ptr_->comms_range = random_rab.GetRange(); // all the range and bearing sensors have the same range
        benchmark_data_ptr_->density = benchmark_data_ptr_->num_agents *
                                       M_PI *
                                       std::pow(benchmark_data_ptr_->comms_range, 2) /
                                       constrained_area; // the density is the ratio of swarm communication area to total walkable area

        // Grab number of trials
        GetNodeAttribute(GetNode(benchmarking_root_node, "num_trials"), "value", benchmark_data_ptr_->num_trials);

        // Grab number of steps
        benchmark_data_ptr_->num_steps = GetSimulator().GetMaxSimulationClock();

        // Grab robot ID prefix and base number
        TConfigurationNode &robot_id_node = GetNode(benchmarking_root_node, "robot_id");

        GetNodeAttribute(robot_id_node, "prefix", benchmark_data_ptr_->id_prefix);
        GetNodeAttribute(robot_id_node, "base_num", benchmark_data_ptr_->id_base_num);

        // Grab JSON file save path
        TConfigurationNode &path_node = GetNode(benchmarking_root_node, "path");
        GetNodeAttribute(path_node, "folder", output_folder_);
        GetNodeAttribute(path_node, "name", output_filename_);
        GetNodeAttribute(path_node, "include_datetime", output_datetime_);

        if (verbose_level_ == "full" || verbose_level_ == "reduced")
        {
            LOG << "[INFO] Benchmarking loop functions verbose level = \"" << verbose_level_ << "\"" << std::endl;
            LOG << "[INFO] Specifying number of robots = " << benchmark_data_ptr_->num_agents << std::endl;
            LOG << "[INFO] Specifying robot speed = " << benchmark_data_ptr_->speed << " cm/s" << std::endl;
            LOG << "[INFO] Specifying number of trials = " << benchmark_data_ptr_->num_trials << std::endl;
            LOG << "[INFO] Specifying output folder = \"" << output_folder_ << "\"" << std::endl;
            LOG << "[INFO] Specifying output statistics filepath (" << ((output_datetime_) ? "with" : "without") << " datetime) = \"" << output_filename_ << "\"" << std::endl;

            LOG << "[INFO] Computed swarm density = " << benchmark_data_ptr_->density << std::endl;

            // Remove underscores from the keyword
            std::string str = benchmark_data_ptr_->parameter_keyword;
            std::replace(str.begin(), str.end(), '_', ' ');

            LOG << "[INFO] Running trial 1 with new parameters:"
                << " target fill ratio = " << curr_paired_parameter_range_itr_->first
                << " & " << str << " = " << curr_paired_parameter_range_itr_->second
                << std::endl;
        }
    }
    catch (CARGoSException &ex)
    {
        THROW_ARGOSEXCEPTION_NESTED("Error parsing loop functions!", ex);
    }

    // Create Packet to store data?
    benchmark_algo_ptr_->InitializeJson(*curr_paired_parameter_range_itr_);

    SetupExperiment();
}

void BenchmarkingLoopFunctions::InitializeBenchmarkAlgorithm(TConfigurationNode &t_tree)
{
    // Define a lambda function for disambiguating the BuzzForeachVM function
    // This is needed to pass into the BenchmarkAlgorithmBase class, which otherwise wouldn't have access
    auto buzz_foreach_vm_func = [this](CBuzzLoopFunctions::COperation &arg)
    { CBuzzLoopFunctions::BuzzForeachVM(arg); };

    // Determine algorithm type
    if (algorithm_str_id_ == CROSSCOMBE_2017)
    {
        // benchmark_algo_ptr_ =
        //     std::make_shared<BenchmarkCrosscombe2017>(t_tree,
        //                                               std::bind(&CBuzzLoopFunctions::BuzzForeachVM, this, std::placeholders::_1));
        benchmark_algo_ptr_ =
            std::make_shared<BenchmarkCrosscombe2017>(buzz_foreach_vm_func, t_tree);
    }
    else
    {
        THROW_ARGOSEXCEPTION("Unknown benchmark algorithm!");
    }
}

void BenchmarkingLoopFunctions::SetupExperiment()
{
    benchmark_algo_ptr_->SetupExperiment(*curr_paired_parameter_range_itr_);
}

REGISTER_LOOP_FUNCTIONS(BenchmarkingLoopFunctions, "benchmarking_loop_functions")