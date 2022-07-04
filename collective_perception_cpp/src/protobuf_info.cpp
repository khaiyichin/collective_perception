#include <string>
#include <iostream>

#include "data_common.hpp"
#include "simulation_set.hpp"
#include "simulation_set.pb.h"

template <typename T>
SimulationSet ExtractData(T &proto_msg, const std::string &filename)
{
    // Load serialized data
    LoadProtoFromDisk(proto_msg, filename);

    // Extract info
    SimulationSet s;
    s.Deserialize(proto_msg);

    return s;
}

int main(int argc, char **argv)
{
    collective_perception_cpp::proto::SimulationStatsSet sim_stats_set_msg;
    collective_perception_cpp::proto::SimulationAgentDataSet sim_agent_data_set_msg;

    SimulationSet s;
    std::string msg_type;
    int packet_size;

    try
    {
        s = ExtractData(sim_stats_set_msg, argv[1]);
        msg_type = sim_stats_set_msg.GetTypeName();
        packet_size = sim_stats_set_msg.stats_packets_size();
    }
    catch (const std::exception &e)
    {
        try
        {
            s = ExtractData(sim_agent_data_set_msg, argv[1]);
            msg_type = sim_agent_data_set_msg.GetTypeName();
            packet_size = sim_agent_data_set_msg.agent_data_packets_size();
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return 0;
        }
    }

    // Extract info

    // Display info
    std::cout << std::endl
              << std::string(40, '*') << " Protobuf Info " << std::string(40, '*') << std::endl
              << std::endl;
    std::cout << "\t\t"
              << "Protobuf type"
              << "\t\t\t: " << msg_type << std::endl;
    std::cout << "\t\t"
              << "Simulation type"
              << "\t\t\t: " << s.simulation_type_ << std::endl;
    std::cout << "\t\t"
              << "Number of agents"
              << "\t\t: " << s.num_agents_ << std::endl;
    std::cout << "\t\t"
              << "Number of trials"
              << "\t\t: " << s.num_trials_ << std::endl;
    std::cout << "\t\t"
              << "Number of steps"
              << "\t\t\t: " << s.num_steps_ << std::endl;
    std::cout << "\t\t"
              << "Communication range"
              << "\t\t: " << s.comms_range_ << " m" << std::endl;
    std::cout << "\t\t"
              << "Swarm density"
              << "\t\t\t: " << s.density_ << std::endl;
    std::cout << "\t\t"
              << "Robot speed"
              << "\t\t\t: " << s.speed_ << " cm/s" << std::endl;

    std::cout << "\t\t"
              << "Target fill ratio range"
              << "\t\t: "
              << "[ ";
    for (const float &tfr : s.tfr_range_)
    {
        std::cout << tfr << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\t\t"
              << "Sensor probability range"
              << "\t: "
              << "[ "
              << std::fixed;
    for (const float &sp : s.sp_range_)
    {
        std::cout << sp << ", ";
    }
    std::cout << "]" << std::scientific << std::endl;

    std::cout << "\t\t"
              << "Number of Packet objects"
              << "\t: " << packet_size << std::endl;

    std::cout << std::endl
              << std::string(36, '*') << " End Protobuf Info " << std::string(40, '*') << std::endl
              << std::endl;

    google::protobuf::ShutdownProtobufLibrary(); // optional, delete any global objects allocated by the Protocol Buffer library
                                                 // needed if memory leak checks are

    return 0;
}