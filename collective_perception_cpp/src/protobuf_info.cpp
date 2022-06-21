#include <string>
#include <iostream>

#include "data_common.hpp"
#include "simulation_set.pb.h"

int main(int argc, char **argv)
{
    collective_perception_cpp::proto::SimulationStatsSet sim_stats_set_msg;

    try
    {
        LoadProtoFromDisk(sim_stats_set_msg, argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 0;
    }

    // Extract info
    std::string msg_type = sim_stats_set_msg.GetTypeName();

    auto sim_set_msg_ptr = sim_stats_set_msg.mutable_sim_set();
    std::string sim_type = sim_set_msg_ptr->sim_type();
    int num_agents = sim_set_msg_ptr->num_agents();
    int num_trials = sim_set_msg_ptr->num_trials();
    int num_steps = sim_set_msg_ptr->num_steps();
    float comms_range = sim_set_msg_ptr->comms_range();
    float density = sim_set_msg_ptr->density();
    float speed = sim_set_msg_ptr->speed();

    std::vector<float> tfr_range = {sim_set_msg_ptr->mutable_tfr_range()->begin(), sim_set_msg_ptr->mutable_tfr_range()->end()};
    std::vector<float> sp_range = {sim_set_msg_ptr->mutable_sp_range()->begin(), sim_set_msg_ptr->mutable_sp_range()->end()};

    // Display info
    std::cout << std::endl
              << std::string(40, '*') << " Protobuf Info " << std::string(40, '*') << std::endl
              << std::endl;
    std::cout << "\t\t"
              << "Protobuf type"
              << "\t\t\t: " << msg_type << std::endl;
    std::cout << "\t\t"
              << "Simulation type"
              << "\t\t\t: " << sim_type << std::endl;
    std::cout << "\t\t"
              << "Number of agents"
              << "\t\t: " << num_agents << std::endl;
    std::cout << "\t\t"
              << "Number of trials"
              << "\t\t: " << num_trials << std::endl;
    std::cout << "\t\t"
              << "Number of steps"
              << "\t\t\t: " << num_steps << std::endl;
    std::cout << "\t\t"
              << "Communication range"
              << "\t\t: " << comms_range << " m" << std::endl;
    std::cout << "\t\t"
              << "Swarm density"
              << "\t\t\t: " << density << std::endl;
    std::cout << "\t\t"
              << "Robot speed"
              << "\t\t\t: " << speed << " cm/s"  << std::endl;

    std::cout << "\t\t"
              << "Target fill ratio range"
              << "\t\t: "
              << "[ ";
    for (const float &tfr : tfr_range)
    {
        std::cout << tfr << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\t\t"
              << "Sensor probability range"
              << "\t: "
              << "[ ";
    for (const float &sp : sp_range)
    {
        std::cout << sp << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\t\t"
              << "Number of Packet objects"
              << "\t: " << sim_stats_set_msg.stats_packets_size() << std::endl;

    std::cout << std::endl
              << std::string(36, '*') << " End Protobuf Info " << std::string(40, '*') << std::endl
              << std::endl;

    google::protobuf::ShutdownProtobufLibrary(); // optional, delete any global objects allocated by the Protocol Buffer library
                                                 // needed if memory leak checks are

    return 0;
}