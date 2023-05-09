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

        // Grab verbosity level
        GetNodeAttribute(GetNode(benchmarking_root_node, "verbosity"), "level", verbose_level_);

        // Grab algorithm information
        TConfigurationNode &algorithm_node = GetNode(benchmarking_root_node, "algorithm");

        GetNodeAttribute(algorithm_node, "type", algorithm_str_id_);

        // Initialize benchmark algorithm
        InitializeBenchmarkAlgorithm(algorithm_node);

        // Obtain the pointer to the benchmark data
        BenchmarkDataBase &benchmark_data = benchmark_algo_ptr_->GetData();

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

        benchmark_data.tfr_range = GenerateLinspace(min, max, steps);

        // Create pairings for target fill ratios and the benchmark algorithm parameter of interest
        for (const double &p1 : benchmark_data.tfr_range)
        {
            for (const double &p2 : benchmark_data.benchmark_param_range)
            {
                paired_parameter_ranges_.push_back(std::pair<double, double>(p1, p2));
            }
        }

        curr_paired_parameter_range_itr_ = paired_parameter_ranges_.begin();

        // Grab robot speeds
        GetNodeAttribute(GetNode(benchmarking_root_node, "speed"), "value", benchmark_data.speed);

        // Grab number of agents and communications range
        auto &rab_map = GetSpace().GetEntitiesByType("rab");
        CRABEquippedEntity &random_rab = *any_cast<CRABEquippedEntity *>(rab_map.begin()->second);

        benchmark_data.num_agents = rab_map.size();         // the number of range and bearing sensors is the same as the number of robots
        benchmark_data.comms_range = random_rab.GetRange(); // all the range and bearing sensors have the same range
        benchmark_data.density = benchmark_data.num_agents *
                                 M_PI *
                                 std::pow(benchmark_data.comms_range, 2) /
                                 constrained_area; // the density is the ratio of swarm communication area to total walkable area

        // Grab number of trials
        GetNodeAttribute(GetNode(benchmarking_root_node, "num_trials"), "value", benchmark_data.num_trials);

        // Grab number of steps
        benchmark_data.num_steps = GetSimulator().GetMaxSimulationClock();

        // Grab JSON file save path
        TConfigurationNode &path_node = GetNode(benchmarking_root_node, "path");
        GetNodeAttribute(path_node, "folder", benchmark_data.output_folder);
        GetNodeAttribute(path_node, "name", benchmark_data.output_filename);
        GetNodeAttribute(path_node, "include_datetime", output_datetime_);

        if (verbose_level_ == "full" || verbose_level_ == "reduced")
        {
            LOG << "[INFO] Benchmarking loop functions verbose level = \"" << verbose_level_ << "\"" << std::endl;
            LOG << "[INFO] Specifying number of robots = " << benchmark_data.num_agents << std::endl;
            LOG << "[INFO] Specifying robot speed = " << benchmark_data.speed << " cm/s" << std::endl;
            LOG << "[INFO] Specifying number of trials = " << benchmark_data.num_trials << std::endl;
            LOG << "[INFO] Specifying output folder = \"" << benchmark_data.output_folder << "\"" << std::endl;
            LOG << "[INFO] Specifying output data filename (" << ((output_datetime_) ? "with" : "without") << " datetime) = \"" << benchmark_data.output_filename << "\"" << std::endl;
            LOG << "[INFO] Computed swarm density = " << benchmark_data.density << std::endl;

            // Remove underscores from the keyword
            std::string str = benchmark_algo_ptr_->GetParameterKeyword();
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

    SetupExperiment();
}

void BenchmarkingLoopFunctions::InitializeBenchmarkAlgorithm(TConfigurationNode &t_tree)
{
    // Get vector of all robot IDs
    std::vector<std::string> robot_id_vec;
    CSpace::TMapPerType &map = GetSpace().GetEntitiesByType("controller");

    for (auto itr = map.begin(); itr != map.end(); ++itr)
    {
        CControllableEntity &entity = *any_cast<CControllableEntity *>(itr->second);
        robot_id_vec.push_back(entity.GetRootEntity().GetId());
    }

    // Define a lambda function for disambiguating the BuzzForeachVM function
    // This is needed to pass into the BenchmarkAlgorithmBase class, which otherwise wouldn't have access
    auto buzz_foreach_vm_func = [this](CBuzzLoopFunctions::COperation &arg)
    { CBuzzLoopFunctions::BuzzForeachVM(arg); };

    // Determine algorithm type
    if (algorithm_str_id_ == CROSSCOMBE_2017)
    {
        benchmark_algo_ptr_ =
            std::make_shared<BenchmarkCrosscombe2017>(buzz_foreach_vm_func, t_tree, robot_id_vec);

        benchmark_algo_ptr_->Init();
    }
    else
    {
        THROW_ARGOSEXCEPTION("Unknown benchmark algorithm!");
    }
}

void BenchmarkingLoopFunctions::SetupExperiment()
{
    benchmark_algo_ptr_->SetupExperiment(curr_trial_ind_, *curr_paired_parameter_range_itr_);
}

void BenchmarkingLoopFunctions::PostStep()
{
    benchmark_algo_ptr_->PostStep();
}

void BenchmarkingLoopFunctions::PostExperiment()
{
    // Specific post experiment operation for benchmark algorithms
    benchmark_algo_ptr_->PostExperiment();

    BenchmarkDataBase &benchmark_data = benchmark_algo_ptr_->GetData();

    if (++curr_trial_ind_ % benchmark_data.num_trials == 0) // all trials for current param set is done
    {
        curr_trial_ind_ = 0; // reset counter

        ++curr_paired_parameter_range_itr_; // use next parameter set

        if (curr_paired_parameter_range_itr_ != paired_parameter_ranges_.end())
        {
            if (verbose_level_ == "full" || verbose_level_ == "reduced")
            {
                // Remove underscores from the keyword
                std::string str = benchmark_algo_ptr_->GetParameterKeyword();
                std::replace(str.begin(), str.end(), '_', ' ');

                LOG << "[INFO] Running trial 1 with new parameters:"
                    << " target fill ratio = " << curr_paired_parameter_range_itr_->first
                    << " & " << str << " = " << curr_paired_parameter_range_itr_->second
                    << std::endl;
            }
        }
        else // no more parameter sets
        {
            if (verbose_level_ == "full" || verbose_level_ == "reduced")
            {
                LOG << "[INFO] All simulation parameters executed." << std::endl;
            }

            SaveData();

            finished_ = true;
        }
    }
    else // more trials required for the current param set
    {
        // Repeat trial
        if (verbose_level_ == "full")
        {
            LOG << "[INFO] Running trial " << curr_trial_ind_ + 1 << " with same parameters." << std::endl;
        }
    }
}

void BenchmarkingLoopFunctions::SaveData()
{
    BenchmarkDataBase &benchmark_data = benchmark_algo_ptr_->GetData();
    std::string foldername_prefix =
        "t" + std::to_string(benchmark_data.num_trials) + "_" +
        "s" + std::to_string(benchmark_data.num_steps) + "_";

    if (output_datetime_)
    {
        // Get current time in string form
        std::string datetime_str = GetCurrentTimeStr();

        // Generate updated filename
        foldername_prefix = benchmark_data.output_folder + "/" + datetime_str + "_" + foldername_prefix;
    }
    else
    {
        foldername_prefix = benchmark_data.output_folder + "/_" + foldername_prefix;
    }

    // Create top-level data folder
    std::filesystem::create_directory(benchmark_data.output_folder);

    benchmark_algo_ptr_->SaveData(foldername_prefix);
}

REGISTER_LOOP_FUNCTIONS(BenchmarkingLoopFunctions, "benchmarking_loop_functions")