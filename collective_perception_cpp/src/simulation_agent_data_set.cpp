#include "simulation_agent_data_set.hpp"

SimulationAgentDataSet::SimulationAgentDataSet(collective_perception_cpp::proto::SimulationAgentDataSet &sim_agent_data_set_msg)
{
    // Load simulation parameters
    SimulationSet::Deserialize(sim_agent_data_set_msg);

    // Load agent data
    for (auto itr = sim_agent_data_set_msg.mutable_agent_data_packets()->begin();
         itr != sim_agent_data_set_msg.mutable_agent_data_packets()->end();
         ++itr)
    {
        AgentDataPacket packet;
        PacketMsg *packet_msg_ptr = itr->mutable_packet();

        packet.sim_type = packet_msg_ptr->sim_type();
        packet.comms_range = packet_msg_ptr->comms_range();
        packet.target_fill_ratio = packet_msg_ptr->tfr();
        packet.b_prob = packet_msg_ptr->b_prob();
        packet.w_prob = packet_msg_ptr->w_prob();
        packet.num_agents = packet_msg_ptr->num_agents();
        packet.num_trials = packet_msg_ptr->num_trials();
        packet.num_steps = packet_msg_ptr->num_steps();
        packet.speed = packet_msg_ptr->speed();

        // Deserialize AgentData messages
        RTAgentDataProtoMsg *repeated_trial_agt_data_msg_ptr = itr->mutable_rtad(); // pointer to RTAgentDataProtoMsg type

        // Iterate through different trials of AgentData objects
        for (auto itr_multi_agt = repeated_trial_agt_data_msg_ptr->mutable_multiagents()->begin();
             itr_multi_agt != repeated_trial_agt_data_msg_ptr->mutable_multiagents()->end();
             ++itr_multi_agt)
        {
            std::vector<AgentData> temp;

            // Iterate through different agents within the same trial
            for (auto itr_agt = itr_multi_agt->mutable_agents()->begin();
                 itr_agt != itr_multi_agt->mutable_agents()->end();
                 ++itr_agt)
            {
                AgentData a;

                // Store vector of tile occurrences
                std::copy(itr_agt->tile_occurrences().begin(),
                          itr_agt->tile_occurrences().end(),
                          std::back_insert_iterator(a.tile_occurrences));

                // Store vector of observations
                std::copy(itr_agt->observations().begin(),
                          itr_agt->observations().end(),
                          std::back_insert_iterator(a.observations));

                temp.push_back(a);
            }

            packet.repeated_agent_data_vec.push_back(temp);
        }

        // Insert populated AgentDataPacket object
        InsertAgentDataPacket(packet);
    }
}

void SimulationAgentDataSet::Serialize(collective_perception_cpp::proto::SimulationAgentDataSet &sim_agent_data_set_msg)
{
    // Serialize SimulationSet object
    SimulationSet::Serialize(sim_agent_data_set_msg);

    // Serialize StatsPacket objects for varying simulation parameters
    std::vector<AgentDataPacket> agent_data_packet_vec = SimulationSet::VectorizePackets(packets_);

    // Store StatsPacket objects for varying simulation parameters
    for (auto agent_data_packet_itr = agent_data_packet_vec.begin(); agent_data_packet_itr != agent_data_packet_vec.end(); ++agent_data_packet_itr)
    {
        auto agent_data_packet_msg_ptr = sim_agent_data_set_msg.add_agent_data_packets();
        PacketMsg *packet_msg_ptr = agent_data_packet_msg_ptr->mutable_packet();

        packet_msg_ptr->set_sim_type(simulation_type_);
        packet_msg_ptr->set_tfr(agent_data_packet_itr->target_fill_ratio);
        packet_msg_ptr->set_b_prob(agent_data_packet_itr->b_prob);
        packet_msg_ptr->set_w_prob(agent_data_packet_itr->w_prob);

        packet_msg_ptr->set_comms_range(comms_range_);
        packet_msg_ptr->set_num_agents(num_agents_);
        packet_msg_ptr->set_num_trials(num_trials_);
        packet_msg_ptr->set_num_steps(num_steps_);
        packet_msg_ptr->set_density(density_);
        packet_msg_ptr->set_speed(speed_);

        // Store RepeatedTrialAgentData object for repeated trials
        *(agent_data_packet_msg_ptr->mutable_rtad()) = ExtractRepeatedTrialAgentDataMsg(agent_data_packet_itr->repeated_agent_data_vec);
    }
}

RTAgentDataProtoMsg SimulationAgentDataSet::ExtractRepeatedTrialAgentDataMsg(const RepeatedTrials<std::vector<AgentData>> &vec)
{
    RTAgentDataProtoMsg rtad_msg;
    std::vector<MultiAgentDataProtoMsg> mad_msg_vec(num_trials_); // vector of MultiAgentData messages

    // Iterate through each trial
    for (auto itr = vec.begin(); itr != vec.end(); ++itr)
    {
        // itr is an iterator to a vector containing multiple AgentData objects within the same trial

        MultiAgentDataProtoMsg mad_msg;                         // MultiAgentData message
        std::vector<AgentDataProtoMsg> ad_msg_vec(num_agents_); // vector of AgentData messages

        // Define lambda function to populate AgentData message
        auto lambda = [](const AgentData &ad)
        {
            AgentDataProtoMsg output_ad_msg;
            *(output_ad_msg.mutable_tile_occurrences()) = {ad.tile_occurrences.begin(), ad.tile_occurrences.end()};
            *(output_ad_msg.mutable_observations()) = {ad.observations.begin(), ad.observations.end()};

            return output_ad_msg;
        };

        // Populate the data from all agents within the same trial
        std::transform(itr->begin(), itr->end(), ad_msg_vec.begin(), lambda);

        // Store data of multiple agents within the same trial
        *(mad_msg.mutable_agents()) = {ad_msg_vec.begin(), ad_msg_vec.end()};
        mad_msg_vec.push_back(mad_msg);
    }

    *(rtad_msg.mutable_multiagents()) = {mad_msg_vec.begin(), mad_msg_vec.end()};

    return rtad_msg;
}