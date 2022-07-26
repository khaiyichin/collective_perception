// Load 2 simulation agent data set protobuf files that i want to compare

#include "simulation_set.hpp"
#include "simulation_agent_data_set.hpp"
#include "simulation_set.pb.h"
#include "data_common.hpp"

#include <string>

int main(int argc, char **argv)
{
    collective_perception_cpp::proto::SimulationAgentDataSet sads_msg_large;

    std::string output_folder = argv[1];

    LoadProtoFromDisk(sads_msg_large, argv[2]);

    // deserialize them, then extract their lower level components
    SimulationAgentDataSet s_large(sads_msg_large);

    // write them into their lower level protobuf files and observe the storage space usage
    std::vector<collective_perception_cpp::proto::AgentDataPacket> adp_vec_large(sads_msg_large.agent_data_packets_size());

    std::vector<AgentDataPacket> agent_data_packet_vec_large = s_large.VectorizePackets(s_large.packets_);

    auto round_1000_int_to_str = [](const float &val)
    { return std::to_string(static_cast<int>(std::round(val * 1e3))); };

    int ind_large = 0;
    std::vector<std::string> large_filenames(sads_msg_large.agent_data_packets_size());
    for (auto &tfr : {0.05, 0.95})
    {
        for (auto &sp : {0.525, 0.75, 0.975})
        {
            AgentDataPacket adp = s_large.GetAgentDataPacket(tfr, sp);
            auto &adp_msg = adp_vec_large[ind_large];
            auto *packet_msg_ptr = adp_msg.mutable_packet();

            packet_msg_ptr->set_sim_type(s_large.simulation_type_);
            packet_msg_ptr->set_tfr(adp.target_fill_ratio);
            packet_msg_ptr->set_b_prob(adp.b_prob);
            packet_msg_ptr->set_w_prob(adp.w_prob);

            packet_msg_ptr->set_comms_range(adp.comms_range);
            packet_msg_ptr->set_num_agents(adp.num_agents);
            packet_msg_ptr->set_num_trials(adp.num_trials);
            packet_msg_ptr->set_num_steps(adp.num_steps);
            packet_msg_ptr->set_density(adp.density);
            packet_msg_ptr->set_speed(adp.speed);

            // Store RepeatedTrialAgentData object for repeated trials
            *(adp_msg.mutable_rtad()) = s_large.ExtractRepeatedTrialAgentDataMsg(adp.repeated_agent_data_vec);

            // // Break down further into a RepeatedTrialAgentData (RTADs)
            // // std
            // std::vector<collective_perception_cpp::proto::RepeatedTrialAgentData::MultiAgentData> rtad_mad_vec_large(adp_msg.mutable_rtad()->multiagents_size());

            // Store filenames
            large_filenames[ind_large] = output_folder + "/large_" + "sp" + round_1000_int_to_str(sp) + "_tfr" + round_1000_int_to_str(tfr) + ".pb";
            // large_filenames[ind_large] = output_folder + "/" + "sp" + round_1000_int_to_str(sp) + "_tfr" + round_1000_int_to_str(tfr);

            int ind_multiagt = 0;

            std::cout << "multiagents_size " << adp_msg.mutable_rtad()->multiagents_size() << std::endl;

            // for (auto itr = adp_msg.mutable_rtad()->mutable_multiagents()->begin();
            //      itr != adp_msg.mutable_rtad()->mutable_multiagents()->end();
            //      ++itr)
            // {
            //     std::string filename = large_filenames[ind_large] + "_" + std::to_string(++ind_multiagt) + ".pb";
            //     WriteProtoToDisk(*itr, filename);
            // }

            // Investigate the first trial
            auto &first_trial_mad_msg = *(adp_msg.mutable_rtad()->mutable_multiagents()->begin());

            std::cout << "agents_size " << first_trial_mad_msg.agents_size() << std::endl;

            // for (auto itr = first_trial_mad_msg.mutable_agents()->begin();
            //      itr != first_trial_mad_msg.mutable_agents()->end();
            //      ++itr)
            // {
            //     std::string filename = large_filenames[ind_large] + "_" + std::to_string(++ind_multiagt) + ".pb";
            //     WriteProtoToDisk(*itr, filename);
            // }

            // Investigate the first agent
            auto &first_agent_msg = *(first_trial_mad_msg.mutable_agents()->begin());

            std::cout << "tile_occurrences_size " << first_agent_msg.tile_occurrences_size() << std::endl;
            std::cout << "observations_size " << first_agent_msg.observations_size() << std::endl;
            
            WriteProtoToDisk(first_agent_msg, large_filenames[ind_large]);

            std::cout << round_1000_int_to_str(tfr) << " " << round_1000_int_to_str(sp) << " tile_occurrences ";
            for (auto &tile: first_agent_msg.tile_occurrences())
            {
                std::cout << tile << " ";
            }
            std::cout << std::endl;

            std::cout << round_1000_int_to_str(tfr) << " " << round_1000_int_to_str(sp) << " observations     ";
            for (auto &obs: first_agent_msg.observations())
            {
                std::cout << obs << " ";
            }
            std::cout << std::endl;

            // WriteProtoToDisk(adp_msg, large_filenames[ind_large]);

            ++ind_large;
        }
    }

    google::protobuf::ShutdownProtobufLibrary();
}