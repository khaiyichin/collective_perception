#include "simulation_data_set.hpp"

SimulationDataSet::SimulationDataSet(collective_perception_cpp::proto::SimulationDataSet &sim_data_set_msg)
{
    // Store simulation parameters
    num_agents_ = sim_data_set_msg.num_agents();   // number of agents
    num_trials_ = sim_data_set_msg.num_trials();   // number of experiments
    num_steps_ = sim_data_set_msg.num_steps();     // number of simulation steps
    comms_range_ = sim_data_set_msg.comms_range(); // communication range
    density_ = sim_data_set_msg.density();         // swarm density

    std::copy(sim_data_set_msg.tfr_range().begin(),
              sim_data_set_msg.tfr_range().end(),
              std::back_inserter(tfr_range_)); // range of target fill ratios
    std::copy(sim_data_set_msg.sp_range().begin(),
              sim_data_set_msg.sp_range().end(),
              std::back_inserter(sp_range_)); // range of sensor probabilities

    // Store SimPacket objects
    for (auto itr = sim_data_set_msg.mutable_sim_packets()->begin();
         itr != sim_data_set_msg.mutable_sim_packets()->end();
         ++itr)
    {
        SimPacket packet;

        packet.comms_range = itr->comms_range();
        packet.target_fill_ratio = itr->tfr();
        packet.b_prob = itr->b_prob();
        packet.w_prob = itr->w_prob();
        packet.num_agents = itr->num_agents();
        packet.num_trials = itr->num_trials();
        packet.num_steps = itr->num_steps();
        packet.density = itr->density();

        // Load AgentData objects
        auto repeated_trial_agt_data_msg_ptr = itr->mutable_rtad(); // pointer to RTAgentDataProtoMsg type

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

        // Load Stats objects (local, social, and informed)
        auto repeated_trial_stats_msg_ptr = itr->mutable_rts(); // pointer to collective_perception_cpp::proto::RepeatedTrialStats type
        auto &local_vals_msg = repeated_trial_stats_msg_ptr->local_vals();
        auto &social_vals_msg = repeated_trial_stats_msg_ptr->social_vals();
        auto &informed_vals_msg = repeated_trial_stats_msg_ptr->informed_vals();

        // Iterate through different trials of values
        for (size_t i = 0; i < num_steps_; ++i)
        {
            Stats local, social, informed;

            // Copy local values
            std::copy(local_vals_msg.Get(i).x_mean().begin(),
                      local_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(local.x_sample_mean));
            std::copy(local_vals_msg.Get(i).conf_mean().begin(),
                      local_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(local.confidence_sample_mean));
            std::copy(local_vals_msg.Get(i).x_std().begin(),
                      local_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(local.x_sample_std));
            std::copy(local_vals_msg.Get(i).conf_std().begin(),
                      local_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(local.confidence_sample_std));

            packet.repeated_local_values.push_back(local);

            // Copy social values
            std::copy(social_vals_msg.Get(i).x_mean().begin(),
                      social_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(social.x_sample_mean));
            std::copy(social_vals_msg.Get(i).conf_mean().begin(),
                      social_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(social.confidence_sample_mean));
            std::copy(social_vals_msg.Get(i).x_std().begin(),
                      social_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(social.x_sample_std));
            std::copy(social_vals_msg.Get(i).conf_std().begin(),
                      social_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(social.confidence_sample_std));

            packet.repeated_social_values.push_back(social);

            // Copy informed values
            std::copy(informed_vals_msg.Get(i).x_mean().begin(),
                      informed_vals_msg.Get(i).x_mean().end(),
                      std::back_insert_iterator(informed.x_sample_mean));
            std::copy(informed_vals_msg.Get(i).conf_mean().begin(),
                      informed_vals_msg.Get(i).conf_mean().end(),
                      std::back_insert_iterator(informed.confidence_sample_mean));
            std::copy(informed_vals_msg.Get(i).x_std().begin(),
                      informed_vals_msg.Get(i).x_std().end(),
                      std::back_insert_iterator(informed.x_sample_std));
            std::copy(informed_vals_msg.Get(i).conf_std().begin(),
                      informed_vals_msg.Get(i).conf_std().end(),
                      std::back_insert_iterator(informed.confidence_sample_std));

            packet.repeated_informed_values.push_back(informed);
        }

        // Insert populated SimPacket object
        InsertSimPacket(packet);
    }
}

void SimulationDataSet::Serialize(collective_perception_cpp::proto::SimulationDataSet &sim_data_set_msg)
{
    sim_data_set_msg.set_sim_type(simulation_type_); // simulation type
    sim_data_set_msg.set_num_agents(num_agents_);    // number of agents
    sim_data_set_msg.set_num_trials(num_trials_);    // number of experiments
    sim_data_set_msg.set_num_steps(num_steps_);      // number of simulation steps
    sim_data_set_msg.set_comms_range(comms_range_);  // communication range
    sim_data_set_msg.set_density(density_);          // swarm density

    *(sim_data_set_msg.mutable_tfr_range()) = {tfr_range_.begin(), tfr_range_.end()};
    *(sim_data_set_msg.mutable_sp_range()) = {sp_range_.begin(), sp_range_.end()};

    std::vector<SimPacket> sim_packets_vec = GetAllSimPackets();

    // Store SimPacket objects for varying simulation parameters
    for (auto sim_pack_itr = sim_packets_vec.begin(); sim_pack_itr != sim_packets_vec.end(); ++sim_pack_itr)
    {
        auto packet_ptr = sim_data_set_msg.add_sim_packets();

        packet_ptr->set_sim_type(simulation_type_);
        packet_ptr->set_tfr(sim_pack_itr->target_fill_ratio);
        packet_ptr->set_b_prob(sim_pack_itr->b_prob);
        packet_ptr->set_w_prob(sim_pack_itr->w_prob);
        packet_ptr->set_num_agents(num_agents_);
        packet_ptr->set_num_trials(num_trials_);

        // Store RepeatedTrialAgentData object for repeated trials
        *(packet_ptr->mutable_rtad()) = ExtractRepeatedTrialAgentDataMsg(sim_pack_itr->repeated_agent_data_vec);

        // Store RepeatedTrialStats object for repeated trials
        *(packet_ptr->mutable_rts()) = ExtractRepeatedTrialStatsMsg({sim_pack_itr->repeated_local_values,
                                                                 sim_pack_itr->repeated_social_values,
                                                                 sim_pack_itr->repeated_informed_values});
    }
}

RTAgentDataProtoMsg SimulationDataSet::ExtractRepeatedTrialAgentDataMsg(const RepeatedTrials<std::vector<AgentData>> &vec)
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

RTStatsProtoMsg SimulationDataSet::ExtractRepeatedTrialStatsMsg(const std::array<RepeatedTrials<Stats>, 3> &arr)
{
    RTStatsProtoMsg rts_msg;
    std::vector<StatsProtoMsg> local_stats_msg_vec(num_trials_);
    std::vector<StatsProtoMsg> social_stats_msg_vec(num_trials_);
    std::vector<StatsProtoMsg> informed_stats_msg_vec(num_trials_);

    // Iterate through each trial
    for (size_t ind = 0; ind < num_trials_; ++ind)
    {
        // arr[0][j] is the Stats struct of the jth trial in the repeated_local_values vector (i.e., jth element in the vector)
        // arr[1][j] is the Stats struct of the jth trial in the repeated_social_values vector (i.e., jth element in the vector)
        // arr[2][j] is the Stats struct of the jth trial in the repeated_informed_values vector (i.e., jth element in the vector)

        StatsProtoMsg local, social, informed;

        *(local.mutable_x_mean()) = {arr[0][ind].x_sample_mean.begin(), arr[0][ind].x_sample_mean.end()};
        *(social.mutable_x_mean()) = {arr[1][ind].x_sample_mean.begin(), arr[1][ind].x_sample_mean.end()};
        *(informed.mutable_x_mean()) = {arr[2][ind].x_sample_mean.begin(), arr[2][ind].x_sample_mean.end()};

        local_stats_msg_vec.push_back(local);
        social_stats_msg_vec.push_back(social);
        informed_stats_msg_vec.push_back(informed);
    }

    *(rts_msg.mutable_local_vals()) = {local_stats_msg_vec.begin(), local_stats_msg_vec.end()};
    *(rts_msg.mutable_social_vals()) = {social_stats_msg_vec.begin(), social_stats_msg_vec.end()};
    *(rts_msg.mutable_informed_vals()) = {informed_stats_msg_vec.begin(), informed_stats_msg_vec.end()};

    return rts_msg;
}

SimPacket SimulationDataSet::GetSimPacket(const float &tfr, const float &sp)
{
    /* Internal units description

    The values for tfr and sp range from 0.000 to 1.000, with a resolution of 0.005.
    Therefore, after scaling to integers, the resolution for the internal units would be 5, and the new
    range is between 0 to 1000.
    */

    // Convert to internal units to use as keys (thus avoiding trailing values with floats)
    unsigned int tfr_internal, sp_internal;

    tfr_internal = RoundToMultiple(ConvertToInternalUnits(tfr), 5);
    sp_internal = RoundToMultiple(ConvertToInternalUnits(sp), 5);

    return sim_packets_[tfr_internal][sp_internal];
}

void SimulationDataSet::InsertSimPacket(const SimPacket &packet)
{
    int tfr_internal = RoundToMultiple(ConvertToInternalUnits(packet.target_fill_ratio), 5);
    int sp_internal = RoundToMultiple(ConvertToInternalUnits(packet.b_prob), 5);

    sim_packets_[tfr_internal][sp_internal] = packet;
}

std::vector<SimPacket> SimulationDataSet::GetAllSimPackets()
{
    std::vector<SimPacket> packets;

    for (auto &tfr_key_value : sim_packets_)
    {
        for (auto &sp_key_value : tfr_key_value.second)
        {
            packets.push_back(sp_key_value.second);
        }
    }

    return packets;
}

int SimulationDataSet::RoundToMultiple(const int &value, const int &base)
{
    int absBase = std::abs(base);
    int remainder = std::abs(value) % absBase;

    if (2 * remainder >= absBase)
    {
        return std::round(std::copysign(std::abs(value) + absBase - remainder, value));
    }
    else
    {
        return std::copysign(std::abs(value) - remainder, value);
    }
}